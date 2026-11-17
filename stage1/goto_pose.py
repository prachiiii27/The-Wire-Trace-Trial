#!/usr/bin/env python3
"""
Group 6 - Stage 1 helper: move the HEAL arm in joint space to the trajectory START (grip) pose,
optionally backed off along the stick/tool axis, so the stick can be re-gripped.

Typical use after homing (gripper OPEN, stick hanging on the wire at START):
  1. python3 stage1/goto_pose.py data/processed/traj_20261115_233621.npz --backoff 0.06 --dry-run
  2. python3 stage1/goto_pose.py data/processed/traj_20261115_233621.npz --backoff 0.06
  3. python3 stage1/goto_pose.py data/processed/traj_20261115_233621.npz          # final, exact grip pose
  4. hold the stick in the open gripper at the SAME depth as before, close the gripper

Motion: quintic joint-space profile (zero vel/acc at both ends), peak joint speed <= --qd-max,
plus P feedback; aborts (zero velocity) if the arm lags the plan by > 0.05 rad.
--dry-run prints the plan and publishes nothing.  ENTER confirms, Ctrl-C stops.
--end   targets the trajectory END instead (stick at the end of the wire).
"""
from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from teleop_teach import JOINTS, Kinematics, dls, rot_log  # noqa: E402

GROUP_DIR = Path(__file__).resolve().parent.parent
URDF_FILE = GROUP_DIR / "config" / "heal_robot_description.urdf"
CMD_TOPIC = "/velocity_controller/commands"
DT = 0.01
KP = 2.0
TRACK_ABORT = 0.05
JS_STALE_S = 0.10


def backoff_q(kin: Kinematics, q_goal, dist):
    """IK: same orientation, tool origin moved -dist along its own z axis."""
    p, R, _ = kin.forward(q_goal)
    p_t = p - R[:, 2] * dist
    q = q_goal.copy()
    for _ in range(500):
        pc, Rc, _ = kin.forward(q)
        e = np.concatenate([p_t - pc, rot_log(R @ Rc.T)])
        if np.linalg.norm(e[:3]) < 1e-5 and np.linalg.norm(e[3:]) < 1e-4:
            return q
        dq, _ = dls(kin.jacobian(q), e)
        q = q + np.clip(dq, -0.05, 0.05)
    raise SystemExit("backoff IK did not converge - try a smaller --backoff")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traj")
    ap.add_argument("--end", action="store_true", help="go to the trajectory END pose")
    ap.add_argument("--backoff", type=float, default=0.0, help="m to stay back along tool -z")
    ap.add_argument("--qd-max", type=float, default=0.15, help="peak joint speed [rad/s]")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not 0.0 <= a.backoff <= 0.15:
        raise SystemExit("--backoff must be in [0, 0.15] m")
    qd_max = min(a.qd_max, 0.3)

    d = np.load(a.traj)
    q_goal = d["q"][-1 if a.end else 0].astype(float)
    # kinematics of the bare tool (rq_tool_tip) for the backoff direction
    kin = Kinematics(URDF_FILE.read_text(), "rq_tool_tip", (0.0, 0.0, 0.0))
    if a.backoff > 0:
        q_goal = backoff_q(kin, q_goal, a.backoff)
    for i in range(6):
        if not kin.lower[i] + 0.03 < q_goal[i] < kin.upper[i] - 0.03:
            raise SystemExit(f"target {JOINTS[i]}={q_goal[i]:.3f} too close to its limit")

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64MultiArray

    from rclpy.signals import SignalHandlerOptions
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)   # Ctrl-C -> our finally (zeros) first
    node = Node("group6_goto_pose")
    lock = threading.Lock()
    js = {}

    def cb(m):
        if len(m.name) == len(m.position):
            if not any(m.position) and not any(m.effort):
                js["dead"] = time.monotonic()      # driver lost the robot: all zeros incl. effort
                return
            with lock:
                js["q"] = (np.array([dict(zip(m.name, m.position))[n] for n in JOINTS]), time.monotonic())

    node.create_subscription(JointState, "/joint_states", cb, qos_profile_sensor_data)
    pub = None if a.dry_run else node.create_publisher(Float64MultiArray, CMD_TOPIC, 10)
    from rclpy.executors import SingleThreadedExecutor
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()

    def send(qd):
        if pub is not None:
            qd = [float(x) for x in qd]
            if not all(math.isfinite(x) and abs(x) <= 0.35 for x in qd):
                qd = [0.0] * 6
            pub.publish(Float64MultiArray(data=qd))

    def get_q():
        with lock:
            return js.get("q", (None, -math.inf))

    rc = 1
    try:
        t0 = time.monotonic()
        while get_q()[0] is None and time.monotonic() - t0 < 5:
            time.sleep(0.05)
        q0, _ = get_q()
        if q0 is None:
            print("[goto] FAIL: no valid /joint_states" + (" - they are ALL ZERO (pos+effort): the driver has "
                  "lost the robot; restart heal_server + velocity_control" if "dead" in js else ""))
            return 1
        delta = q_goal - q0
        T = max(1.875 * float(np.abs(delta).max()) / qd_max, 2.0)
        print(f"[goto] target ({'END' if a.end else 'START'}, backoff {a.backoff*1000:.0f} mm):")
        print(f"         q_now  = {np.round(q0, 3)}")
        print(f"         q_goal = {np.round(q_goal, 3)}")
        print(f"         delta  = {np.round(delta, 3)}  -> {T:.1f} s, peak {qd_max} rad/s")
        p0, _, _ = kin.forward(q0)
        pg, _, _ = kin.forward(q_goal)
        print(f"         tool tip moves {1000*np.linalg.norm(pg-p0):.0f} mm")
        if a.dry_run:
            print("[goto] DRY RUN - nothing published")
            return 0
        if node.count_subscribers(CMD_TOPIC) == 0:
            print(f"[goto] FAIL: nobody subscribes {CMD_TOPIC}")
            return 1
        if node.count_publishers(CMD_TOPIC) > 1:
            print(f"[goto] FAIL: another node publishes {CMD_TOPIC}")
            return 1
        input("[goto] path clear, gripper/stick not snagged?  ENTER to move, Ctrl-C to cancel ")
        t_start = time.monotonic()
        nxt = t_start
        while True:
            t = time.monotonic() - t_start
            tau = min(t / T, 1.0)
            s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
            sd = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / T
            q_des = q0 + delta * s
            q, stamp = get_q()
            if time.monotonic() - stamp > JS_STALE_S:
                print("[goto] ABORT: joint_states stale" + (" / ALL ZERO (driver lost robot)" if "dead" in js else ""))
                break
            err = q_des - q
            if np.abs(err).max() > TRACK_ABORT:
                print(f"[goto] ABORT: lagging plan by {np.abs(err).max():.3f} rad")
                break
            send(np.clip(delta * sd + KP * err, -qd_max - 0.05, qd_max + 0.05))
            if tau >= 1.0 and np.abs(q_goal - q).max() < 0.002:
                print(f"[goto] reached, residual {np.abs(q_goal - q).max():.4f} rad")
                rc = 0
                break
            if t > T + 5:
                print(f"[goto] timeout, residual {np.abs(q_goal - q).max():.4f} rad")
                break
            nxt += DT
            time.sleep(max(0.0, nxt - time.monotonic()))
    except KeyboardInterrupt:
        print("\n[goto] Ctrl-C -> stop")
    finally:
        for _ in range(10):
            send([0.0] * 6)
            time.sleep(0.01)
        ex.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
