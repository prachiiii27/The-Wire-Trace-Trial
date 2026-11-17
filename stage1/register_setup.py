#!/usr/bin/env python3
"""
Group 6 - Stage 1: re-register a taught path after the wire board moved (no re-teach).

Idea: the P marks in the teach recording are reference spots on the wire.  After the board
moves, put the stick on the SAME spots again (teleop, press P at each), then fit the board's
rigid motion T (base frame) from old -> new mark poses, move the whole taught stick-tip path by T
and solve IK for every sample.  The output is a new teach CSV (same columns) that
process_teach.py turns into a trajectory as usual.

Models (--model):
  translation   3 DoF, needs >= 1 reference point
  planar        x, y, z + rotation about vertical z (board slid/turned on the table), >= 2 points
  rigid         full 6 DoF, >= 3 points (also uses the stick orientation at each mark, low weight)
  (default: planar with >= 2 points, translation with 1 point)
  NOTE rigid is unreliable for this wire: the marks lie almost on one line (along y), so roll about
  that line comes only from the hand-placed stick orientation (+-several deg).
The residuals tell you whether "the board moved rigidly" explains the new marks at all.
If no model fits (residual >> 1-2 mm) the stick probably moved in the gripper, or a point
was placed at a different spot -> check, or re-teach.

Matching: each new mark is paired with the NEAREST taught mark (marks are many cm apart,
the board moves only mm), so you may give any subset in any order.

Usage
  python3 stage1/register_setup.py data/teach_20261116_150937.csv data/teach_<ref recording>.csv [more ref csvs]
  python3 stage1/register_setup.py ... --model planar --dry-run       # only fit + report
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from teleop_teach import JOINTS, Kinematics, dls, rot_log  # noqa: E402

GROUP_DIR = Path(__file__).resolve().parent.parent
URDF_FILE = GROUP_DIR / "config" / "heal_robot_description.urdf"
QCOLS = [f"q{i}" for i in range(1, 7)]
ORI_ARM = 0.05      # m: lever used to turn the mark orientation into extra points (rigid model)
ORI_WEIGHT = 0.3    # weight of those orientation points vs. positions


def marks(df: pd.DataFrame, kin: Kinematics):
    rows = df[df.mark == 1]
    out = []
    for _, r in rows.iterrows():
        q = r[QCOLS].to_numpy(float)
        p, R, _ = kin.forward(q)
        out.append({"t_rel": float(r.t_rel), "q": q, "p": p, "R": R})
    return out


def weighted_kabsch(A, B, w):
    """R, t minimising sum w |R a + t - b|^2."""
    w = w / w.sum()
    ca, cb = (w[:, None] * A).sum(0), (w[:, None] * B).sum(0)
    H = ((A - ca) * w[:, None]).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cb - R @ ca


def fit(model, P_old, P_new, R_old=None, R_new=None):
    n = len(P_old)
    if model == "translation":
        return np.eye(3), (P_new - P_old).mean(0)
    if model == "planar":
        ca, cb = P_old[:, :2].mean(0), P_new[:, :2].mean(0)
        a, b = P_old[:, :2] - ca, P_new[:, :2] - cb
        th = math.atan2((a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum(), (a * b).sum())
        R = Rotation.from_euler("z", th).as_matrix()
        t = np.r_[cb - R[:2, :2] @ ca, (P_new[:, 2] - P_old[:, 2]).mean()]
        return R, t
    if model == "rigid":
        A, B, w = [P_old], [P_new], [np.ones(n)]
        for k in range(3):   # orientation as extra (down-weighted) points
            A.append(P_old + ORI_ARM * R_old[:, :, k])
            B.append(P_new + ORI_ARM * R_new[:, :, k])
            w.append(np.full(n, ORI_WEIGHT))
        return weighted_kabsch(np.vstack(A), np.vstack(B), np.concatenate(w))
    raise ValueError(model)


def residuals(R, t, P_old, P_new, R_old, R_new):
    e_p = 1000 * np.linalg.norm(P_old @ R.T + t - P_new, axis=1)
    e_r = np.array([math.degrees(np.linalg.norm(rot_log(R_new[i] @ (R @ R_old[i]).T)))
                    for i in range(len(P_old))])
    return e_p, e_r


def ik_path(kin, q_seed, P_tgt, R_tgt, tol_mm=0.02, tol_deg=0.02):
    """sequential DLS IK, warm-started from the previous solution (and the old q as seed)."""
    n = len(q_seed)
    Q = np.empty_like(q_seed)
    q = q_seed[0].copy()
    worst = (0.0, 0.0)
    for i in range(n):
        q = q + (q_seed[i] - q_seed[i - 1] if i else 0.0)    # follow the taught motion as predictor
        for _ in range(30):
            p, R, _ = kin.forward(q)
            e = np.concatenate([P_tgt[i] - p, rot_log(R_tgt[i] @ R.T)])
            if np.linalg.norm(e[:3]) * 1000 < tol_mm and math.degrees(np.linalg.norm(e[3:])) < tol_deg:
                break
            dq, _ = dls(kin.jacobian(q), e)
            q = q + np.clip(dq, -0.05, 0.05)
        worst = (max(worst[0], 1000 * np.linalg.norm(e[:3])), max(worst[1], math.degrees(np.linalg.norm(e[3:]))))
        Q[i] = q
    return Q, worst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("teach", help="original teach CSV (with P marks)")
    ap.add_argument("refs", nargs="+", help="new recording(s) with P marks at the same spots")
    ap.add_argument("--model", choices=["default", "translation", "planar", "rigid"], default="default")
    ap.add_argument("--tol-mm", type=float, default=1.5, help="auto: accept model if max residual below")
    ap.add_argument("--max-match-mm", type=float, default=40.0, help="max old<->new mark distance")
    ap.add_argument("--dry-run", action="store_true", help="fit and report only")
    ap.add_argument("--tip-offset", type=float, nargs=3, default=[0.0, 0.0, 0.095])
    a = ap.parse_args()

    kin = Kinematics(URDF_FILE.read_text(), "rq_tool_tip", tuple(a.tip_offset))
    teach_path = Path(a.teach).resolve()
    teach = pd.read_csv(teach_path, low_memory=False)
    old = marks(teach, kin)
    if not old:
        raise SystemExit("teach CSV has no P marks")
    new = []
    for f in a.refs:
        new += marks(pd.read_csv(f, low_memory=False), kin)
    if not new:
        raise SystemExit("reference recording(s) contain no P marks (press P at each spot while recording)")

    print(f"[register] taught marks: " + ", ".join(f"#{i} t={m['t_rel']:.0f}s {np.round(m['p'], 3)}"
                                                    for i, m in enumerate(old)))
    pairs = {}
    for nm in new:
        d = [np.linalg.norm(nm["p"] - om["p"]) for om in old]
        k = int(np.argmin(d))
        if d[k] * 1000 > a.max_match_mm:
            print(f"[register] new mark at {np.round(nm['p'], 3)} is {1000*d[k]:.0f} mm from every taught mark -> ignored")
            continue
        pairs[k] = nm            # last one wins if a spot was marked twice
    if not pairs:
        raise SystemExit("no new mark matches a taught mark")
    idx = sorted(pairs)
    P_old = np.array([old[k]["p"] for k in idx])
    P_new = np.array([pairs[k]["p"] for k in idx])
    R_old = np.array([old[k]["R"] for k in idx])
    R_new = np.array([pairs[k]["R"] for k in idx])
    print(f"[register] matched {len(idx)} reference point(s): taught marks {idx}")
    for j, k in enumerate(idx):
        d = 1000 * (P_new[j] - P_old[j])
        print(f"   mark #{k}: moved {np.round(d, 1)} mm (|{np.linalg.norm(d):.1f}| mm), "
              f"stick orientation differs {math.degrees(np.linalg.norm(rot_log(R_new[j] @ R_old[j].T))):.1f} deg")

    need = {"translation": 1, "planar": 2, "rigid": 3}
    results = {}
    for m in ("translation", "planar", "rigid"):
        if len(idx) < need[m]:
            continue
        R, t = fit(m, P_old, P_new, R_old, R_new)
        e_p, e_r = residuals(R, t, P_old, P_new, R_old, R_new)
        rv = np.degrees(Rotation.from_matrix(R).as_rotvec())
        results[m] = (R, t, e_p, e_r)
        dof = {"translation": 3, "planar": 4, "rigid": 6}[m]
        note = "  (exact fit - no check possible)" if 3 * len(idx) <= dof else ""
        print(f"[register] {m:11s}: t = {np.round(1000*t, 1)} mm, rot = {np.round(rv, 2)} deg | "
              f"residual max {e_p.max():.2f} mm, orient. max {e_r.max():.1f} deg{note}")

    if a.model == "default":
        choice = "planar" if "planar" in results else "translation"
    else:
        if a.model not in results:
            raise SystemExit(f"model {a.model} needs >= {need[a.model]} reference points")
        choice = a.model
    R, t, e_p, e_r = results[choice]
    print(f"[register] using model: {choice}")
    if len(idx) * 3 > {"translation": 3, "planar": 4, "rigid": 6}[choice] and e_p.max() > a.tol_mm:
        print(f"[register] WARNING: residual {e_p.max():.1f} mm > {a.tol_mm} mm - a rigid board move does not "
              f"explain the new marks. A spot was placed differently, or the stick moved in the gripper. "
              f"Check before using this; consider re-teaching.")
    if a.dry_run:
        return 0

    # ---- transform the whole taught path ----
    teach = teach.drop_duplicates("js_stamp").reset_index(drop=True)
    q_old = teach[QCOLS].to_numpy(float)
    step = np.r_[np.inf, np.abs(np.diff(q_old, axis=0)).max(axis=1)]
    keep = step > 1e-7
    keep[0] = keep[-1] = True
    marks_kept = teach.mark.to_numpy() == 1
    keep |= marks_kept
    sub = teach[keep].reset_index(drop=True)
    q_old = sub[QCOLS].to_numpy(float)
    print(f"[register] solving IK for {len(q_old)} path samples ...")
    P = np.empty((len(q_old), 3))
    Rm = np.empty((len(q_old), 3, 3))
    for i, q in enumerate(q_old):
        p, Ri, _ = kin.forward(q)
        P[i] = R @ p + t
        Rm[i] = R @ Ri
    Q, worst = ik_path(kin, q_old, P, Rm)
    print(f"[register] IK worst error {worst[0]:.3f} mm / {worst[1]:.3f} deg; "
          f"max joint change vs taught {np.abs(Q - q_old).max(0).round(3)} rad")
    margin = 0.03
    if (Q.min(0) < kin.lower + margin).any() or (Q.max(0) > kin.upper - margin).any():
        raise SystemExit(f"registered path hits a joint limit: min {Q.min(0).round(3)} max {Q.max(0).round(3)}")
    if worst[0] > 0.2 or worst[1] > 0.2:
        raise SystemExit("IK did not converge everywhere - not writing output")
    jumps = np.abs(np.diff(Q, axis=0)).max()
    if jumps > 0.05:
        raise SystemExit(f"IK solution jumps by {jumps:.3f} rad between samples (branch switch) - not writing")

    out = sub.copy()
    out[QCOLS] = Q
    for i in range(len(Q)):
        p, _, quat = kin.forward(Q[i])
        out.loc[i, ["tip_x", "tip_y", "tip_z"]] = p
        out.loc[i, ["tip_qx", "tip_qy", "tip_qz", "tip_qw"]] = quat
    out[["tool_x", "tool_y", "tool_z", "tool_qx", "tool_qy", "tool_qz", "tool_qw"]] = np.nan
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = teach_path.with_name(f"{teach_path.stem}_reg{stamp}.csv")
    out.to_csv(out_path, index=False, float_format="%.6f")
    meta = json.loads(teach_path.with_suffix(".json").read_text()) if teach_path.with_suffix(".json").exists() else {}
    meta.update({"registered_from": teach_path.name, "refs": [str(f) for f in a.refs], "model": choice,
                 "R": R.tolist(), "t_m": t.tolist(), "residual_mm": e_p.tolist(), "matched_marks": idx})
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"[register] wrote {out_path}")
    print(f"[register] next: python3 stage1/process_teach.py {out_path.relative_to(GROUP_DIR)} [same options as before]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
