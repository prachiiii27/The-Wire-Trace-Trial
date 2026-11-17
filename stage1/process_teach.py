#!/usr/bin/env python3
"""
Group 6 - Stage 1 / Module 2: turn a teleop teach recording into a replayable trajectory.

OFFLINE ONLY - no ROS, publishes nothing.

Pipeline
  1. load data/teach_*.csv, drop duplicate joint_states samples and all dwell
     (rows where the arm did not move) -> pure geometric path in joint space
  2. re-parameterise by a pose arc-length u  (du^2 = |dp_tip|^2 + (ROT_L*dtheta)^2)
     and resample on a uniform u grid
  3. light Gaussian smoothing in u (joint space); sigma is shrunk automatically until
     the smoothed stick-tip path stays within --tol-mm / --tol-deg of the taught one
  4. time-scaling: fastest profile s(t) respecting per-joint |qd|, |qdd|, tip speed and
     tip angular speed (forward/backward pass), zero velocity at both ends
  5. sample at 100 Hz -> data/processed/<name>.npz (+ .csv, .json summary, .png plots)

The npz holds  t, q (N,6), qd (N,6) [joint1..6], tip_p (N,3), tip_quat (N,4 xyzw)
and is what replay_dls.py consumes.

Usage
  python3 stage1/process_teach.py data/teach_20261115_233621.csv
  python3 stage1/process_teach.py data/teach_20261115_233621.csv --qd-max 0.40 --v-max 0.04 --name fast
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from teleop_teach import JOINTS, Kinematics  # noqa: E402  (same KDL chain as teaching)

GROUP_DIR = Path(__file__).resolve().parent.parent
URDF_FILE = GROUP_DIR / "config" / "heal_robot_description.urdf"
OUT_DIR = GROUP_DIR / "data" / "processed"

ROT_L = 0.03          # m per rad: weight of rotation in the arc-length metric
DU = 0.0005           # resample spacing in u [m-equivalent]
MOVE_EPS = 1e-7       # rad: rows with no joint change at all are dwell
RATE_HZ = 100.0


def load_teach(csv_path: Path):
    df = pd.read_csv(csv_path, low_memory=False)
    n_raw = len(df)
    df = df.drop_duplicates("js_stamp").reset_index(drop=True)
    q = df[[f"q{i}" for i in range(1, 7)]].to_numpy(float)
    step = np.r_[np.inf, np.abs(np.diff(q, axis=0)).max(axis=1)]
    keep = step > MOVE_EPS
    keep[0] = True
    keep[-1] = True
    return df, q[keep], {"rows_raw": n_raw, "rows_unique": len(df), "rows_moving": int(keep.sum()),
                         "teach_duration_s": float(df.t_rel.iloc[-1]),
                         "teach_moving_s": float((step[1:] > MOVE_EPS).sum() / RATE_HZ),
                         "marks_t_rel": df.t_rel[df.mark == 1].round(2).tolist()}


def fk_all(kin: Kinematics, q: np.ndarray):
    P = np.empty((len(q), 3))
    Rm = np.empty((len(q), 3, 3))
    for k, qk in enumerate(q):
        P[k], Rm[k], _ = kin.forward(qk)
    return P, Rotation.from_matrix(Rm)


def arc_param(P: np.ndarray, R: Rotation) -> np.ndarray:
    dp = np.linalg.norm(np.diff(P, axis=0), axis=1)
    dth = (R[1:] * R[:-1].inv()).magnitude()
    return np.r_[0.0, np.cumsum(np.sqrt(dp ** 2 + (ROT_L * dth) ** 2))]


def resample(u: np.ndarray, q: np.ndarray, du: float):
    keep = np.r_[True, np.diff(u) > 1e-9]
    u, q = u[keep], q[keep]
    ug = np.arange(0.0, u[-1], du)
    if u[-1] - ug[-1] < 0.5 * du:
        ug[-1] = u[-1]              # no tiny last interval (it blows up dq/du at the end)
    else:
        ug = np.r_[ug, u[-1]]
    qg = np.column_stack([np.interp(ug, u, q[:, i]) for i in range(6)])
    return ug, qg


def smooth(qg: np.ndarray, sigma_pts: float) -> np.ndarray:
    if sigma_pts <= 0:
        return qg.copy()
    qs = gaussian_filter1d(qg, sigma_pts, axis=0, mode="nearest")
    # pin the exact taught start/end with an offset that fades out smoothly
    n = len(qg)
    w = int(min(n // 4, 6 * sigma_pts))
    if w > 1:
        fade = (0.5 + 0.5 * np.cos(np.linspace(0, math.pi, w)))[:, None]   # 1 -> 0
        qs[:w] += (qg[0] - qs[0]) * fade
        qs[-w:] += (qg[-1] - qs[-1]) * fade[::-1]
    return qs


def deviation(kin, q_new, Pg, Rg, win):
    """Max distance of the new path from the reference resampled path, comparing each
    point only with reference points within +-win samples (avoids matching another
    pass of the wire loop that happens to be close in space)."""
    P, R = fk_all(kin, q_new)
    n = len(P)
    dpos = np.full(n, np.inf)
    dang = np.full(n, np.inf)
    for off in range(-win, win + 1):
        j = np.clip(np.arange(n) + off, 0, n - 1)
        dpos = np.minimum(dpos, np.linalg.norm(P - Pg[j], axis=1))
        dang = np.minimum(dang, (R * Rg[j].inv()).magnitude())
    return 1000 * dpos.max(), math.degrees(dang.max()), P, R


def _profile(sd_max, a_s, du):
    sd = sd_max.copy()
    sd[0] = sd[-1] = 0.0
    n = len(sd)
    for k in range(n - 1):                    # forward (accelerate)
        sd[k + 1] = min(sd[k + 1], math.sqrt(sd[k] ** 2 + 2 * a_s[k] * du))
    for k in range(n - 1, 0, -1):             # backward (decelerate)
        sd[k - 1] = min(sd[k - 1], math.sqrt(sd[k] ** 2 + 2 * a_s[k] * du))
    return np.maximum(sd, 0.0)


def touch_scale(qs, ug, touch_files, radius, factor):
    """per-grid-point speed multiplier: `factor` within +-radius (path u) of every
    recorded wire touch (matched to the path by joint configuration)."""
    scale = np.ones(len(qs))
    n = 0
    for f in touch_files:
        df = pd.read_csv(f)
        for _, r in df.iterrows():
            qt = r[[f"q{i}" for i in range(1, 7)]].to_numpy(float)
            k = int(np.abs(qs - qt).max(axis=1).argmin())
            dist = float(np.abs(qs[k] - qt).max())
            if dist > 0.05:
                print(f"[process] touch in {Path(f).name} at {r.get('progress_pct', '?')}% is "
                      f"{dist:.3f} rad from this path -> ignored (different teach?)")
                continue
            scale[np.abs(ug - ug[k]) <= radius] = factor
            n += 1
    print(f"[process] slowing x{factor} within +-{1000*radius:.0f} mm around {n} touch(es), "
          f"{100*(scale < 1).mean():.1f}% of the path")
    return scale


def time_scale(qg, P, R, du, qd_max, qdd_max, v_max, w_max, iters=60, scale=None):
    """Path-velocity bound from qd/v/w limits + forward/backward accel pass, then
    iteratively lower the bound wherever the true joint accel (incl. the path-curvature
    term q''*sdot^2) exceeds qdd_max."""
    dq = np.gradient(qg, du, axis=0)                       # dq/du
    dp = np.linalg.norm(np.gradient(P, du, axis=0), axis=1)
    dth = np.r_[(R[1:] * R[:-1].inv()).magnitude(), 0.0] / du
    dth = np.maximum(dth, np.r_[0.0, dth[:-1]])
    absdq = np.maximum(np.abs(dq), 1e-9)
    sd_max = np.min(qd_max / absdq, axis=1)
    sd_max = np.minimum(sd_max, v_max / np.maximum(dp, 1e-9))
    sd_max = np.minimum(sd_max, w_max / np.maximum(dth, 1e-9))
    if scale is not None:
        sd_max = sd_max * scale
    a_s = 0.5 * np.min(qdd_max / absdq, axis=1)
    for it in range(iters):
        sd = _profile(sd_max, a_s, du)
        avg = 0.5 * (sd[1:] + sd[:-1])
        dt = du / np.maximum(avg, 1e-6)
        tg = np.r_[0.0, np.cumsum(dt)]
        qd = dq * sd[:, None]
        qdd = np.gradient(qd, tg, axis=0)
        ratio = np.abs(qdd).max(axis=1) / qdd_max
        bad = ratio > 1.0
        bad[0] = bad[-1] = False
        if not bad.any():
            break
        # shrink the speed bound around violating samples
        idx = np.flatnonzero(bad)
        f = 1.0 / np.sqrt(ratio[idx])
        for d in (-2, -1, 0, 1, 2):
            j = np.clip(idx + d, 0, len(sd) - 1)
            sd_max[j] = np.minimum(sd_max[j], np.maximum(sd[j] * f * 0.95, 1e-4))
    return tg, sd


def build(args):
    csv_path = Path(args.csv).resolve()
    kin = Kinematics(URDF_FILE.read_text(), "rq_tool_tip", tuple(args.tip_offset))
    df, q_raw, info = load_teach(csv_path)
    print(f"[process] {csv_path.name}: {info['rows_raw']} rows, {info['rows_unique']} unique, "
          f"{info['rows_moving']} moving ({info['teach_moving_s']:.0f} s of {info['teach_duration_s']:.0f} s)")

    P_raw, R_raw = fk_all(kin, q_raw)
    rec_tip = df[["tip_x", "tip_y", "tip_z"]].to_numpy(float)
    print(f"[process] FK check vs recorded tip: start {1000*np.linalg.norm(P_raw[0]-rec_tip[0]):.3f} mm, "
          f"end {1000*np.linalg.norm(P_raw[-1]-rec_tip[-1]):.3f} mm")
    u = arc_param(P_raw, R_raw)
    ug, qg = resample(u, q_raw, DU)
    Pg, Rg = fk_all(kin, qg)

    # largest smoothing that keeps the path within tolerance
    chosen = None
    for sigma_mm in [float(s) for s in args.sigma_mm]:
        qs = smooth(qg, sigma_mm / 1000 / DU)
        win = int(3 * sigma_mm / 1000 / DU) + 2
        dmm, ddeg, Ps, Rs = deviation(kin, qs, Pg, Rg, win)
        print(f"[process] smoothing sigma {sigma_mm:4.1f} mm -> max tip dev {dmm:.2f} mm, {ddeg:.2f} deg")
        if dmm <= args.tol_mm and ddeg <= args.tol_deg:
            chosen = (sigma_mm, qs, Ps, Rs, dmm, ddeg)
            break
    if chosen is None:
        qs = qg
        Ps, Rs = fk_all(kin, qs)
        chosen = (0.0, qs, Ps, Rs, 0.0, 0.0)
        print("[process] no smoothing level met tolerance -> using raw resampled path")
    sigma_mm, qs, Ps, Rs, dmm, ddeg = chosen

    margin = args.limit_margin
    lo_hit = qs.min(0) < kin.lower + margin
    hi_hit = qs.max(0) > kin.upper - margin
    if lo_hit.any() or hi_hit.any():
        raise SystemExit(f"[process] path comes within {margin} rad of a joint limit: "
                         f"min {np.round(qs.min(0),3)} max {np.round(qs.max(0),3)}")

    scale = None
    if args.slow_near:
        scale = touch_scale(qs, ug, args.slow_near, args.slow_radius_mm / 1000, args.slow_factor)
    tg, sd = time_scale(qs, Ps, Rs, DU, args.qd_max, args.qdd_max, args.v_max, args.w_max, scale=scale)
    T = float(tg[-1])
    t = np.arange(0.0, T, 1.0 / RATE_HZ)
    t = np.r_[t, T] if t[-1] < T else t
    spl = CubicSpline(ug, qs, axis=0)
    u_t = np.interp(t, tg, ug)
    sd_t = np.interp(t, tg, sd)
    q_t = spl(u_t)
    qd_t = spl(u_t, 1) * sd_t[:, None]
    qd_t[0] = qd_t[-1] = 0.0
    qdd_t = np.gradient(qd_t, t, axis=0)
    P_t, R_t = fk_all(kin, q_t)
    v_t = np.linalg.norm(np.gradient(P_t, t, axis=0), axis=1)

    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    name = csv_path.stem.replace("teach_", "traj_") + (f"_{args.name}" if args.name else "")
    npz = out_dir / f"{name}.npz"
    np.savez(npz, t=t, q=q_t, qd=qd_t, tip_p=P_t, tip_quat=R_t.as_quat(),
             joints=np.array(JOINTS), tip_offset=np.array(args.tip_offset),
             path_q=qs, path_u=ug, raw_q=q_raw,
             mark_q=df[df.mark == 1][[f"q{i}" for i in range(1, 7)]].to_numpy(float).reshape(-1, 6))
    cols = (["t"] + [f"q{i}" for i in range(1, 7)] + [f"qd{i}" for i in range(1, 7)]
            + ["tip_x", "tip_y", "tip_z", "tip_qx", "tip_qy", "tip_qz", "tip_qw"])
    pd.DataFrame(np.column_stack([t, q_t, qd_t, P_t, R_t.as_quat()]), columns=cols) \
        .to_csv(out_dir / f"{name}.csv", index=False, float_format="%.6f")

    summary = {
        "source": csv_path.name, **info,
        "path_length_m": float(np.linalg.norm(np.diff(P_raw, axis=0), axis=1).sum()),
        "total_rotation_deg": float(np.degrees((R_raw[1:] * R_raw[:-1].inv()).magnitude().sum())),
        "smoothing_sigma_mm": sigma_mm, "max_dev_mm": dmm, "max_dev_deg": ddeg,
        "limits": {"qd_max": args.qd_max, "qdd_max": args.qdd_max, "v_max": args.v_max,
                   "w_max": args.w_max},
        "duration_s": T,
        "slow_near": [str(x) for x in args.slow_near],
        "achieved": {"max_abs_qd": np.abs(qd_t).max(0).round(3).tolist(),
                     "max_abs_qdd": np.abs(qdd_t).max(0).round(3).tolist(),
                     "max_tip_speed_mps": float(v_t.max())},
        "q_min": q_t.min(0).round(4).tolist(), "q_max": q_t.max(0).round(4).tolist(),
        "start_q": q_t[0].round(6).tolist(), "end_q": q_t[-1].round(6).tolist(),
        "tip_offset_m": list(args.tip_offset),
    }
    (out_dir / f"{name}.json").write_text(json.dumps(summary, indent=2))

    print(f"[process] chosen sigma {sigma_mm} mm, max deviation {dmm:.2f} mm / {ddeg:.2f} deg")
    print(f"[process] duration {T:.1f} s  (teach motion was {info['teach_moving_s']:.0f} s)")
    print(f"[process] max |qd| per joint {summary['achieved']['max_abs_qd']}  (limit {args.qd_max})")
    print(f"[process] max |qdd| per joint {summary['achieved']['max_abs_qdd']}  (target {args.qdd_max})")
    print(f"[process] max tip speed {1000*v_t.max():.1f} mm/s")
    print(f"[process] wrote {npz}")

    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(14, 9))
            ax = fig.add_subplot(2, 2, 1, projection="3d")
            ax.plot(*P_raw.T, lw=0.6, label="taught")
            ax.plot(*P_t.T, lw=0.8, label="processed")
            ax.set_title("stick tip path [m]")
            ax.legend()
            ax2 = fig.add_subplot(2, 2, 2)
            ax2.plot(t, 1000 * v_t)
            ax2.set_title("tip speed [mm/s]")
            ax2.set_xlabel("t [s]")
            ax3 = fig.add_subplot(2, 2, 3)
            ax3.plot(t, q_t)
            ax3.set_title("q [rad]")
            ax3.legend(JOINTS, fontsize=7)
            ax4 = fig.add_subplot(2, 2, 4)
            ax4.plot(t, qd_t)
            ax4.axhline(args.qd_max, ls="--", c="k")
            ax4.axhline(-args.qd_max, ls="--", c="k")
            ax4.set_title("qd [rad/s]")
            fig.tight_layout()
            fig.savefig(out_dir / f"{name}.png", dpi=110)
            print(f"[process] plot {out_dir / f'{name}.png'}")
        except Exception as e:  # plotting is optional
            print(f"[process] plot skipped: {e}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--qd-max", type=float, default=0.30, help="per-joint speed [rad/s] (driver cap 0.5)")
    ap.add_argument("--qdd-max", type=float, default=0.6, help="per-joint accel [rad/s^2]")
    ap.add_argument("--v-max", type=float, default=0.03, help="stick-tip speed [m/s]")
    ap.add_argument("--w-max", type=float, default=0.6, help="stick angular speed [rad/s]")
    ap.add_argument("--tol-mm", type=float, default=1.0, help="max smoothed-path deviation [mm]")
    ap.add_argument("--tol-deg", type=float, default=1.5, help="max smoothed-path deviation [deg]")
    ap.add_argument("--sigma-mm", nargs="+", default=["6", "4", "3", "2", "1", "0.5"],
                    help="smoothing levels to try, largest first")
    ap.add_argument("--limit-margin", type=float, default=0.03, help="rad from URDF limits")
    ap.add_argument("--tip-offset", type=float, nargs=3, default=[0.0, 0.0, 0.095])
    ap.add_argument("--name", default="", help="suffix for output files")
    ap.add_argument("--slow-near", nargs="+", default=[], metavar="TOUCH_CSV",
                    help="data/touches/*.csv from earlier replays: slow down around those spots")
    ap.add_argument("--slow-factor", type=float, default=0.4, help="speed multiplier near touches")
    ap.add_argument("--slow-radius-mm", type=float, default=25.0, help="path half-width slowed per touch")
    ap.add_argument("--no-plot", action="store_true")
    raise SystemExit(build(ap.parse_args()))


if __name__ == "__main__":
    main()
