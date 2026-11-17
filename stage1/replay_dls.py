#!/usr/bin/env python3
"""
Group 6 - Stage 1 / Module 3: replay a processed trajectory with DLS-IK velocity control.

Control law (100 Hz), stick tip = rq_tool_tip + tip offset:
    X_d(tau), V_ff(tau)   desired tip pose / twist from FK and J of the taught q(tau)
    V   = r * V_ff + [Kp*(p_d - p) ; Kr*log(R_d R^T)]       (correction clipped)
    qd  = DLS(J(q), V)  -> per-joint cap, accel ramp, joint-limit guard
    tau advances at rate r*speed; r drops smoothly when tracking error grows,
    r -> 0 on pause, abort (hold still) beyond the abort thresholds.

Modes
  --sim       no ROS at all: simulated arm (first-order lag) from the trajectory start.
              Gives the expected run time and tracking error.  Needs config/heal_robot_description.urdf
  --dry-run   live /joint_states, publishes NOTHING: checks start-pose match, then simulates
  (default)   LIVE: commands /velocity_controller/commands

  --reverse   play the path backwards (end -> start), e.g. to bring the stick back to START
  --list-marks          print where the teach P marks are on this trajectory (% of the forward path)
  --from-pct / --to-pct play only part of the path (forward %, also with --reverse), e.g. stop at a
                        mark to record a reference point for register_setup.py

Keys while LIVE (terminal, no ENTER needed):
  SPACE or T = mark a WIRE TOUCH here (press when the buzzer sounds) -> data/touches/*.csv
  P          = pause / resume
  Ctrl-C     = stop (zero velocity, exit)
Touches are stored with the path position in FORWARD trajectory time (also for --reverse),
joints, stick-tip pose, tracking error and joint efforts; process_teach.py --slow-near uses them.
Requires heal_server (robot) + velocity_control.launch.py (PC) running for live/dry-run.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import select
import sys
import termios
import threading
import tty
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from teleop_teach import JOINTS, Kinematics, clip_norm, dls, rot_log  # noqa: E402

GROUP_DIR = Path(__file__).resolve().parent.parent
URDF_FILE = GROUP_DIR / "config" / "heal_robot_description.urdf"
LOG_DIR = GROUP_DIR / "data" / "replays"
TOUCH_DIR = GROUP_DIR / "data" / "touches"
CMD_TOPIC = "/velocity_controller/commands"

RATE_HZ = 100.0
DT = 1.0 / RATE_HZ
QD_CAP_HARD = 0.45          # 0.9 x driver limit (driver zeroes everything above it); see --driver-limit
ACC_LIMIT = 4.0             # rad/s^2 ramp on the published command (trajectories use <= 3)
LIMIT_MARGIN = 0.02         # rad from URDF limits
JS_STALE_S = 0.10
LOOP_STALE_S = 0.10
KP = 3.0                    # 1/s position feedback
KR = 3.0                    # 1/s orientation feedback
CORR_V_MAX = 0.010          # m/s max feedback correction
CORR_W_MAX = 0.10           # rad/s
RATE_UP = 0.8               # 1/s: how fast the progress rate may increase
RATE_DOWN = 4.0             # 1/s: how fast it may decrease
RATE_STOP = 1.0             # 1/s^2: progress braking before a cut end (--to-pct / --from-pct)
START_TOL_RAD = 0.02        # max |q - q_d(0)| to start without an approach phase
APPROACH_TOL_RAD = 0.08     # beyond this: refuse
APPROACH_QD = 0.04          # rad/s during the approach phase
APPROACH_DONE_RAD = 0.008   # approach ends here; the run's Cartesian feedback removes the rest
APPROACH_TIMEOUT_S = 30.0
SETTLE_S = 2.0              # feedback-only time at the end
SETTLE_MM, SETTLE_DEG = 0.5, 0.3


# ----------------------------------------------------------------------------
class Trajectory:
    def __init__(self, path: Path, reverse: bool, from_pct=0.0, to_pct=100.0):
        d = np.load(path)
        if [str(j) for j in d["joints"]] != JOINTS:
            raise SystemExit(f"trajectory joint order {d['joints']} != {JOINTS}")
        self.t = d["t"]
        self.q = d["q"]
        self.qd = d["qd"]
        T0 = float(self.t[-1])
        self.T_full = T0
        self.t_off = 0.0           # forward time of the first sample of this (possibly cut) trajectory
        self.mark_pct = []
        if "mark_q" in d.files:
            for mq in d["mark_q"]:
                k = int(np.abs(self.q - mq).max(axis=1).argmin())
                self.mark_pct.append(100 * self.t[k] / T0)
        if from_pct > 0 or to_pct < 100:
            if not 0 <= from_pct < to_pct <= 100:
                raise SystemExit("need 0 <= --from-pct < --to-pct <= 100")
            k0 = int(np.searchsorted(self.t, from_pct / 100 * T0))
            k1 = int(np.searchsorted(self.t, to_pct / 100 * T0))
            k1 = min(max(k1, k0 + 2), len(self.t))
            self.t_off = float(self.t[k0])
            self.t = self.t[k0:k1] - self.t[k0]
            self.q = self.q[k0:k1].copy()
            self.qd = self.qd[k0:k1].copy()
        self.tip_offset = tuple(float(x) for x in d["tip_offset"])
        self.name = path.stem
        self.reverse = reverse
        # the end we play towards is a cut (mid-motion) -> controller must brake the progress rate
        self.cut_end = (from_pct > 0) if reverse else (to_pct < 100)
        if reverse:
            self.q = self.q[::-1].copy()
            self.qd = -self.qd[::-1].copy()
            self.t = self.t[-1] - self.t[::-1]
        self.T = float(self.t[-1])

    def at(self, tau: float):
        tau = min(max(tau, 0.0), self.T)
        k = int(np.searchsorted(self.t, tau, side="right")) - 1
        k = min(max(k, 0), len(self.t) - 2)
        a = (tau - self.t[k]) / (self.t[k + 1] - self.t[k])
        return (self.q[k] * (1 - a) + self.q[k + 1] * a,
                self.qd[k] * (1 - a) + self.qd[k + 1] * a)


class Controller:
    """Pure computation; shared by sim and live."""

    def __init__(self, kin: Kinematics, traj: Trajectory, args):
        self.kin, self.traj, self.a = kin, traj, args
        self.qd_cap = min(args.qd_cap, QD_CAP_HARD)
        self.tau = 0.0
        self.rate = 0.0
        self.qd_prev = np.zeros(6)
        self.paused = False
        self.phase = "approach"
        self.approach_t = 0.0
        self.settle_t = 0.0
        self.abort_reason = ""
        self.max_e = (0.0, 0.0)

    def step(self, q: np.ndarray):
        """returns (qd_cmd, info dict)."""
        a, kin, tr = self.a, self.kin, self.traj
        q_d, qd_ff = tr.at(self.tau)
        p, R, _ = kin.forward(q)
        p_d, R_d, _ = kin.forward(q_d)
        e_p = p_d - p
        e_r = rot_log(R_d @ R.T)
        e_mm, e_deg = 1000 * float(np.linalg.norm(e_p)), math.degrees(float(np.linalg.norm(e_r)))
        target = np.zeros(6)

        if self.phase == "approach":
            dq = q_d - q
            self.approach_t += DT
            if np.abs(dq).max() < max(APPROACH_DONE_RAD, START_TOL_RAD if self.approach_t <= DT else 0):
                self.phase = "run"          # small offsets: start directly, feedback absorbs them
            elif self.approach_t > APPROACH_TIMEOUT_S:
                self.phase, self.abort_reason = "abort", f"approach did not converge ({np.abs(dq).max():.4f} rad)"
            else:
                # at least 0.01 rad/s on the leading joint: the arm ignores tinier commands
                g = max(1.0, 0.01 / max(float(np.abs(dq).max()), 1e-6))
                target = np.clip(g * dq, -APPROACH_QD, APPROACH_QD)

        if self.phase == "run":
            self.max_e = (max(self.max_e[0], e_mm), max(self.max_e[1], e_deg))
            if e_mm > a.abort_mm or e_deg > a.abort_deg:
                self.phase = "abort"
                self.abort_reason = f"tracking error {e_mm:.1f} mm / {e_deg:.1f} deg"
            else:
                # progress-rate scaling from tracking error
                x = max((e_mm - a.warn_mm) / (a.abort_mm - a.warn_mm),
                        (e_deg - a.warn_deg) / (a.abort_deg - a.warn_deg), 0.0)
                r_target = 0.0 if self.paused else a.speed * max(0.0, 1.0 - x)
                if tr.cut_end:   # brake along the path so we stop exactly at the cut
                    r_target = min(r_target, math.sqrt(2 * RATE_STOP * max(tr.T - self.tau, 0.0)) + 0.02)
                dr = r_target - self.rate
                self.rate += min(max(dr, -RATE_DOWN * DT), RATE_UP * DT)
                self.rate = max(self.rate, 0.0)
                J_d = kin.jacobian(q_d)
                V = self.rate * (J_d @ qd_ff)
                V[:3] += clip_norm(KP * e_p, CORR_V_MAX)
                V[3:] += clip_norm(KR * e_r, CORR_W_MAX)
                target, _ = dls(kin.jacobian(q), V)
                self.tau += self.rate * DT
                if self.tau >= tr.T:
                    self.tau = tr.T
                    self.phase = "settle"

        if self.phase == "settle":
            self.settle_t += DT
            V = np.concatenate([clip_norm(KP * e_p, CORR_V_MAX), clip_norm(KR * e_r, CORR_W_MAX)])
            target, _ = dls(kin.jacobian(q), V)
            if (e_mm < SETTLE_MM and e_deg < SETTLE_DEG) or self.settle_t > SETTLE_S:
                self.phase = "done"

        if self.phase in ("abort", "done"):
            target = np.zeros(6)

        if not np.all(np.isfinite(target)):
            target = np.zeros(6)
            self.phase, self.abort_reason = "abort", "non-finite IK"
        m = float(np.abs(target).max())
        if m > self.qd_cap:
            target *= self.qd_cap / m
        for i in range(6):
            if (q[i] > kin.upper[i] - LIMIT_MARGIN and target[i] > 0) or \
               (q[i] < kin.lower[i] + LIMIT_MARGIN and target[i] < 0):
                target = np.zeros(6)
                if self.phase not in ("done",):
                    self.phase, self.abort_reason = "abort", f"{JOINTS[i]} at limit ({q[i]:.3f})"
                break
        lim = ACC_LIMIT * DT
        qd = self.qd_prev + np.clip(target - self.qd_prev, -lim, lim)
        qd[np.abs(qd) < 1e-6] = 0.0
        self.qd_prev = qd
        return qd, {"q_d": q_d, "e_mm": e_mm, "e_deg": e_deg, "p": p, "p_d": p_d}


def start_check(q_now, traj: Trajectory, kin: Kinematics):
    dq = np.abs(q_now - traj.q[0])
    p, R, _ = kin.forward(q_now)
    p0, R0, _ = kin.forward(traj.q[0])
    e_mm = 1000 * np.linalg.norm(p - p0)
    e_deg = math.degrees(np.linalg.norm(rot_log(R0 @ R.T)))
    print(f"[replay] start check: max |q - q_start| = {dq.max():.4f} rad ({JOINTS[int(dq.argmax())]}), "
          f"tip {e_mm:.1f} mm / {e_deg:.2f} deg from trajectory start")
    if dq.max() > APPROACH_TOL_RAD:
        print(f"[replay] REFUSED: robot is not at the trajectory start (> {APPROACH_TOL_RAD} rad). "
              f"If it is at the END of the wire, use --reverse.")
        return False
    if dq.max() > START_TOL_RAD:
        print(f"[replay] will first approach the start slowly (joint space, {APPROACH_QD} rad/s)")
    return True


class TouchRecorder:
    COLS = (["wall_time", "t_run", "traj", "direction", "speed", "tau_fwd", "progress_pct",
             "e_mm", "e_deg"] + [f"q{i}" for i in range(1, 7)]
            + ["tip_x", "tip_y", "tip_z", "tip_qx", "tip_qy", "tip_qz", "tip_qw",
               "des_x", "des_y", "des_z"] + [f"effort{i}" for i in range(1, 7)])

    def __init__(self, traj: "Trajectory", speed: float):
        TOUCH_DIR.mkdir(parents=True, exist_ok=True)
        self.traj, self.speed = traj, speed
        self.path = TOUCH_DIR / (f"touches_{traj.name}_{'rev' if traj.reverse else 'fwd'}_"
                                 f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        self.f = None
        self.n = 0
        self.last = -1.0

    def mark(self, t_run, c: "Controller", q, info, effort, kin):
        if t_run - self.last < 0.3:          # debounce
            return False
        self.last = t_run
        if self.f is None:
            self.f = open(self.path, "w", newline="")
            self.w = csv.writer(self.f)
            self.w.writerow(self.COLS)
        tr = self.traj
        tau_fwd = tr.t_off + (tr.T - c.tau if tr.reverse else c.tau)
        p, _, quat = kin.forward(q)
        self.w.writerow([f"{time.time():.3f}", f"{t_run:.3f}", tr.name,
                         "rev" if tr.reverse else "fwd", self.speed, f"{tau_fwd:.3f}",
                         f"{100 * tau_fwd / tr.T_full:.2f}", f"{info['e_mm']:.3f}", f"{info['e_deg']:.3f}"]
                        + [f"{x:.6f}" for x in q] + [f"{x:.6f}" for x in p]
                        + [f"{x:.6f}" for x in quat] + [f"{x:.6f}" for x in info["p_d"]]
                        + [f"{x:.4f}" for x in effort])
        self.f.flush()
        self.n += 1
        return True

    def close(self):
        if self.f:
            self.f.close()


class KeyReader:
    """single-key reads from the terminal (cbreak: Ctrl-C still works)."""

    def __init__(self):
        self.fd = sys.stdin.fileno() if sys.stdin.isatty() else None
        self.old = None

    def __enter__(self):
        if self.fd is not None:
            self.old = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        return self

    def keys(self):
        out = ""
        while self.fd is not None and select.select([self.fd], [], [], 0)[0]:
            ch = os.read(self.fd, 1).decode(errors="ignore")
            if not ch:
                break
            out += ch
        return out

    def __exit__(self, *exc):
        if self.old is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)


class Logger:
    def __init__(self, name: str):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.path = LOG_DIR / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.f = open(self.path, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(["t", "phase", "tau", "rate", "e_mm", "e_deg", "touch"]
                        + [f"q{i}" for i in range(1, 7)] + [f"qdes{i}" for i in range(1, 7)]
                        + [f"qd_cmd{i}" for i in range(1, 7)])

    def row(self, t, c: Controller, q, info, qd, touch=0):
        self.w.writerow([f"{t:.4f}", c.phase, f"{c.tau:.4f}", f"{c.rate:.3f}",
                         f"{info['e_mm']:.3f}", f"{info['e_deg']:.3f}", touch]
                        + [f"{x:.6f}" for x in q] + [f"{x:.6f}" for x in info["q_d"]]
                        + [f"{x:.6f}" for x in qd])

    def close(self):
        self.f.close()


# ----------------------------------------------------------------------------
def simulate(kin, traj, args, q0=None, lag=0.04, label="sim"):
    c = Controller(kin, traj, args)
    q = traj.q[0].copy() if q0 is None else q0.copy()
    qd_act = np.zeros(6)
    t = 0.0
    t_run0 = t_run1 = None
    while c.phase not in ("done", "abort") and t < 4 * traj.T / args.speed + 60:
        qd, info = c.step(q)
        if c.phase == "run" and t_run0 is None:
            t_run0 = t
        if c.phase in ("settle", "done") and t_run1 is None:
            t_run1 = t
        qd_act += (qd - qd_act) * (DT / (lag + DT))      # first-order actuator lag
        q = q + qd_act * DT
        t += DT
    print(f"[{label}] result: {c.phase} {c.abort_reason}")
    if t_run0 is not None and t_run1 is not None:
        print(f"[{label}] run time {t_run1 - t_run0:.1f} s (trajectory {traj.T:.1f} s, speed x{args.speed})")
    print(f"[{label}] max tracking error {c.max_e[0]:.2f} mm / {c.max_e[1]:.2f} deg "
          f"(model: {1000*lag:.0f} ms first-order lag)")
    return 0 if c.phase == "done" else 1


# ----------------------------------------------------------------------------
def run_ros(kin_urdf_from_topic: bool, traj: Trajectory, args):
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64MultiArray

    class ReplayNode(Node):
        def __init__(self, publish):
            super().__init__("group6_replay_dls")
            self.lock = threading.Lock()
            self.js = None
            self.dead = False
            self.eff = np.full(6, np.nan)
            self.pub = None
            self.armed = False
            self.heartbeat = time.monotonic()
            self.create_subscription(JointState, "/joint_states", self._js, qos_profile_sensor_data)
            if publish:
                self.pub = self.create_publisher(Float64MultiArray, CMD_TOPIC, 10)
                self.create_timer(0.02, self._watchdog)

        def _js(self, m):
            if len(m.name) == len(m.position):
                if not any(m.position) and not any(m.effort):
                    self.dead = True        # driver lost the robot: all zeros incl. effort
                    return
                eff = dict(zip(m.name, m.effort)) if len(m.effort) == len(m.name) else {}
                with self.lock:
                    self.js = (dict(zip(m.name, m.position)), time.monotonic())
                    self.eff = np.array([eff.get(n, float("nan")) for n in JOINTS])

        def q(self):
            with self.lock:
                js = self.js
            if js is None:
                return None, math.inf
            return np.array([js[0][n] for n in JOINTS]), time.monotonic() - js[1]

        def effort(self):
            with self.lock:
                return self.eff.copy()

        def publish(self, qd):
            if self.pub is None:
                return
            qd = [float(v) for v in qd]
            if len(qd) != 6 or not all(math.isfinite(v) and abs(v) <= QD_CAP_HARD for v in qd):
                qd = [0.0] * 6
            self.pub.publish(Float64MultiArray(data=qd))

        def _watchdog(self):
            if self.armed and time.monotonic() - self.heartbeat > LOOP_STALE_S:
                self.publish([0.0] * 6)

    live = not args.dry_run
    from rclpy.signals import SignalHandlerOptions
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)   # Ctrl-C -> our finally (zeros) first
    node = ReplayNode(publish=live)
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    kin = Kinematics(URDF_FILE.read_text(), "rq_tool_tip", traj.tip_offset)
    rc = 1
    log = None
    touches = None
    keyrd = None
    try:
        t0 = time.monotonic()
        while node.q()[0] is None and time.monotonic() - t0 < 5:
            time.sleep(0.05)
        q_now, age = node.q()
        if q_now is None:
            print("[replay] FAIL: no valid /joint_states" + (" - they are ALL ZERO (pos+effort): the driver "
                  "has lost the robot; restart heal_server + velocity_control" if node.dead else ""))
            return 1
        if live and node.count_subscribers(CMD_TOPIC) == 0:
            print(f"[replay] FAIL: nobody subscribes {CMD_TOPIC} (velocity_control.launch.py running?)")
            return 1
        if live and node.count_publishers(CMD_TOPIC) > 1:
            print(f"[replay] FAIL: another node publishes {CMD_TOPIC} (close teleop first - "
                  f"driver zeroes on >1 publisher)")
            return 1
        if not start_check(q_now, traj, kin):
            return 1
        if not live:
            print("[replay] DRY RUN: start pose OK, simulating from the current joint state")
            return simulate(kin, traj, args, q0=q_now, label="dry-run")

        print(f"[replay] LIVE: {traj.T / args.speed:.0f} s nominal, speed x{args.speed}, "
              f"qd cap {min(args.qd_cap, QD_CAP_HARD)} rad/s, abort at {args.abort_mm} mm / {args.abort_deg} deg")
        print("[replay] during the run:  SPACE/T = mark wire touch   P = pause/resume   Ctrl-C = stop")
        input("[replay] stick on the wire, area clear?  press ENTER to start (Ctrl-C to cancel) ")
        c = Controller(kin, traj, args)
        log = Logger("replay_rev" if args.reverse else "replay")
        touches = TouchRecorder(traj, args.speed)
        keyrd = KeyReader().__enter__()
        node.heartbeat = time.monotonic()
        node.armed = True
        t_start = time.monotonic()
        t_run0 = t_run1 = None
        next_tick = time.monotonic()
        last_print = 0.0
        while c.phase not in ("done", "abort"):
            node.heartbeat = time.monotonic()
            keys = keyrd.keys().lower()
            if "p" in keys:
                c.paused = not c.paused
                print("[replay] PAUSED (P to resume)" if c.paused else "[replay] resumed")
            want_touch = (" " in keys) or ("t" in keys)
            q, age = node.q()
            if age > JS_STALE_S:
                node.publish([0.0] * 6)
                c.phase, c.abort_reason = "abort", f"joint_states stale ({age*1000:.0f} ms)"
                break
            qd, info = c.step(q)
            node.publish(qd)
            now = time.monotonic()
            if c.phase == "run" and t_run0 is None:
                t_run0 = now
            if c.phase in ("settle", "done") and t_run1 is None:
                t_run1 = now
            touched = 0
            if want_touch and touches.mark(now - t_start, c, q, info, node.effort(), kin):
                touched = 1
                print(f"[replay] TOUCH #{touches.n} at {100*c.tau/traj.T:.1f}% "
                      f"tip {np.round(1000*info['p'], 1)} mm")
            log.row(now - t_start, c, q, info, qd, touched)
            if now - last_print > 1.0:
                last_print = now
                print(f"  {c.phase:8s} {100*c.tau/traj.T:5.1f}%  rate {c.rate:.2f}  "
                      f"err {info['e_mm']:.2f} mm {info['e_deg']:.2f} deg")
            next_tick += DT
            sl = next_tick - time.monotonic()
            if sl > 0:
                time.sleep(sl)
            else:
                next_tick = time.monotonic()
        # ramp down whatever is commanded
        qd = c.qd_prev.copy()
        while np.abs(qd).max() > 0:
            qd = np.sign(qd) * np.maximum(np.abs(qd) - ACC_LIMIT * 2 * DT, 0.0)
            node.publish(qd)
            time.sleep(DT)
        print(f"[replay] {c.phase.upper()} {c.abort_reason}")
        if t_run0 is not None and t_run1 is not None:
            print(f"[replay] run time {t_run1 - t_run0:.2f} s")
        print(f"[replay] max tracking error {c.max_e[0]:.2f} mm / {c.max_e[1]:.2f} deg")
        print(f"[replay] wire touches marked: {touches.n}")
        rc = 0 if c.phase == "done" else 1
    except KeyboardInterrupt:
        print("\n[replay] Ctrl-C -> stopping")
    finally:
        if keyrd is not None:
            keyrd.__exit__(None, None, None)
        if touches is not None:
            touches.close()
            if touches.n:
                print(f"[replay] touches saved: {touches.path}")
        node.armed = False
        try:
            for _ in range(10):
                node.publish([0.0] * 6)
                time.sleep(0.01)
        except Exception as e:
            print(f"[replay] WARNING could not publish final zeros: {e}")
        if log:
            log.close()
            print(f"[replay] log {log.path}")
        ex.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return rc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traj", help="data/processed/traj_*.npz")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--sim", action="store_true")
    g.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reverse", action="store_true", help="play end -> start")
    ap.add_argument("--from-pct", type=float, default=0.0, help="start at this forward %% of the path")
    ap.add_argument("--to-pct", type=float, default=100.0, help="stop at this forward %% of the path")
    ap.add_argument("--list-marks", action="store_true", help="print the teach marks' %% positions and exit")
    ap.add_argument("--speed", type=float, default=1.0, help="time scale on the trajectory (0.05 .. 2.0)")
    ap.add_argument("--qd-cap", type=float, default=0.45, help="per-joint cap [rad/s] (<= 0.9 x --driver-limit)")
    ap.add_argument("--driver-limit", type=float, default=0.5,
                    help="velocity limit the driver was LAUNCHED with (HEAL_COMMISSIONING_VELOCITY_LIMIT, "
                         "default 0.5); only raise if you really set that env var")
    ap.add_argument("--warn-mm", type=float, default=1.5)
    ap.add_argument("--warn-deg", type=float, default=1.5)
    ap.add_argument("--abort-mm", type=float, default=6.0)
    ap.add_argument("--abort-deg", type=float, default=5.0)
    args = ap.parse_args()
    if not 0.05 <= args.speed <= 2.0:
        raise SystemExit("--speed must be in [0.05, 2.0]")
    if not 0.5 <= args.driver_limit <= 2.0:
        raise SystemExit("--driver-limit must be in [0.5, 2.0]")
    global QD_CAP_HARD
    QD_CAP_HARD = 0.9 * args.driver_limit
    traj = Trajectory(Path(args.traj), args.reverse, args.from_pct, args.to_pct)
    if args.list_marks:
        if not traj.mark_pct:
            print("[replay] no marks stored in this trajectory (re-run process_teach.py to add them)")
        for i, pct in enumerate(traj.mark_pct):
            print(f"[replay] mark #{i}: {pct:6.2f} % of the forward path")
        raise SystemExit(0)
    print(f"[replay] {Path(args.traj).name}{' (REVERSED)' if args.reverse else ''}: {traj.T:.1f} s, "
          f"{len(traj.t)} samples")
    if args.sim:
        kin = Kinematics(URDF_FILE.read_text(), "rq_tool_tip", traj.tip_offset)
        raise SystemExit(simulate(kin, traj, args))
    raise SystemExit(run_ros(True, traj, args))


if __name__ == "__main__":
    main()
