#!/usr/bin/env python3
"""
Group 6 - Stage 1 / Module 1: keyboard teleop + teach recorder for the HEAL arm.

Jog the STICK TIP (rq_tool_tip + tip offset) in Cartesian space with the keyboard.
Damped-least-squares IK converts the twist into joint velocities published on
/velocity_controller/commands.  While recording, joint states + tip pose are
written to data/teach_<timestamp>.csv for the processing module.

Requires heal_server (robot) + velocity_control.launch.py (PC) running.

Modes
  --selftest   no window, no publisher: checks URDF/KDL/FK against /tool_end_effector_pose
  --dry-run    window + keys, computes everything, publishes NOTHING
  (default)    live

Keys (focus the pygame window; motion only while a key is HELD)
  W/S  +x/-x     A/D  +y/-y     R/F  +z/-z          (translation)
  J/L  +rx/-rx   I/K  +ry/-ry   U/O  +rz/-rz        (rotation about stick tip)
  1/2/3 speed level   TAB base/tool frame   SPACE brake (hold)
  ENTER start/stop recording   P mark waypoint   ESC quit
No gripper keys on purpose (dropping the stick).

Safety: dead-man keys, stop on window focus loss, stop if /joint_states is stale,
per-joint |qd| <= 0.30 rad/s (driver rejects > 0.5), joint-limit guard,
accel-limited ramp, watchdog thread publishes zero if the UI loop freezes
(the driver holds the last command forever otherwise), zero on exit.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import PyKDL as kdl
import rclpy
from geometry_msgs.msg import PoseStamped
from kdl_parser_py.urdf import treeFromUrdfModel
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy,
                       qos_profile_sensor_data)
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String
from urdf_parser_py.urdf import URDF

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
JOINTS = [f"joint{i}" for i in range(1, 7)]   # command order of velocity_controller
BASE_LINK = "base_link"
DEFAULT_TIP_LINK = "rq_tool_tip"
DEFAULT_TIP_OFFSET = (0.0, 0.0, 0.095)          # m, in rq_tool_tip frame (stick loop centre)
CMD_TOPIC = "/velocity_controller/commands"

RATE_HZ = 100.0
DT = 1.0 / RATE_HZ
QD_MAX = 0.30            # rad/s per joint (driver hard limit 0.5)
ACC_UP = 0.6             # rad/s^2 when speeding up
ACC_DOWN = 3.0           # rad/s^2 when slowing down
LIMIT_MARGIN = 0.08      # rad from URDF joint limits
JS_STALE_S = 0.10        # joint_states older than this -> stop
LOOP_STALE_S = 0.10      # UI loop silent longer than this -> watchdog zeros
SPEEDS = [(0.005, 0.03), (0.010, 0.06), (0.020, 0.10)]   # (m/s, rad/s) levels 1..3
DLS_EPS = 0.05           # sigma_min below which damping ramps in
DLS_LAMBDA_MAX = 0.08
HOLD_KP = 2.0            # 1/s, position hold while only rotating
HOLD_KR = 2.0            # 1/s, orientation hold while only translating
HOLD_V_MAX = 0.003       # m/s
HOLD_W_MAX = 0.05        # rad/s

GROUP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = GROUP_DIR / "data"


# ----------------------------------------------------------------------------
# Kinematics
# ----------------------------------------------------------------------------
def rot_log(R: np.ndarray) -> np.ndarray:
    """SO(3) log map -> rotation vector (angle*axis)."""
    c = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    angle = math.acos(c)
    if angle < 1e-9:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = math.sin(angle)
    if s < 1e-6:  # angle ~ pi
        w, V = np.linalg.eigh((R + R.T) / 2.0)
        return V[:, np.argmax(w)] * angle
    return v / (2.0 * s) * angle


def clip_norm(v: np.ndarray, vmax: float) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v if n <= vmax else v * (vmax / n)


class Kinematics:
    def __init__(self, urdf_xml: str, tip_link: str, tip_offset):
        model = URDF.from_xml_string(urdf_xml)
        ok, tree = treeFromUrdfModel(model)
        if not ok:
            raise RuntimeError("kdl_parser failed on robot_description")
        chain = tree.getChain(BASE_LINK, tip_link)
        chain.addSegment(kdl.Segment("stick_tip", kdl.Joint(),
                                     kdl.Frame(kdl.Vector(*map(float, tip_offset)))))
        names = []
        for i in range(chain.getNrOfSegments()):
            j = chain.getSegment(i).getJoint()
            if j.getTypeName() not in ("None", "Fixed"):
                names.append(j.getName())
        if names != JOINTS:
            raise RuntimeError(f"KDL chain joints {names} != expected {JOINTS}")
        self.chain = chain
        self.fk = kdl.ChainFkSolverPos_recursive(chain)
        self.jac = kdl.ChainJntToJacSolver(chain)
        self.lower = np.array([model.joint_map[n].limit.lower for n in JOINTS])
        self.upper = np.array([model.joint_map[n].limit.upper for n in JOINTS])

    @staticmethod
    def _q(q):
        a = kdl.JntArray(6)
        for i in range(6):
            a[i] = float(q[i])
        return a

    def forward(self, q):
        f = kdl.Frame()
        self.fk.JntToCart(self._q(q), f)
        p = np.array([f.p[0], f.p[1], f.p[2]])
        R = np.array([[f.M[r, c] for c in range(3)] for r in range(3)])
        quat = np.array(f.M.GetQuaternion())  # x, y, z, w
        return p, R, quat

    def jacobian(self, q) -> np.ndarray:
        J = kdl.Jacobian(6)
        self.jac.JntToJac(self._q(q), J)
        return np.array([[J[r, c] for c in range(6)] for r in range(6)])


def dls(J: np.ndarray, x: np.ndarray):
    s = np.linalg.svd(J, compute_uv=False)
    smin = float(s[-1])
    lam2 = 0.0 if smin >= DLS_EPS else DLS_LAMBDA_MAX ** 2 * (1.0 - (smin / DLS_EPS) ** 2)
    lam2 = max(lam2, 1e-8)
    qd = J.T @ np.linalg.solve(J @ J.T + lam2 * np.eye(6), x)
    return qd, smin


# ----------------------------------------------------------------------------
# ROS node
# ----------------------------------------------------------------------------
class TeleopNode(Node):
    def __init__(self, publish: bool):
        super().__init__("group6_teleop_teach")
        self.lock = threading.Lock()
        self.js = None            # (pos dict, vel dict, stamp_sec, rx_monotonic)
        self.tool_pose = None     # (xyz, quat_xyzw)
        self.urdf = None
        self.heartbeat = time.monotonic()
        self.armed = False        # watchdog only active once the UI loop is running
        self.pub_lock = threading.Lock()
        self.pub = None
        self.watchdog_zeroing = False

        self.create_subscription(JointState, "/joint_states", self._js_cb, qos_profile_sensor_data)
        self.create_subscription(PoseStamped, "/tool_end_effector_pose", self._pose_cb,
                                 qos_profile_sensor_data)
        latched = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE,
                             durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, "/robot_description", self._urdf_cb, latched)
        if publish:
            self.pub = self.create_publisher(Float64MultiArray, CMD_TOPIC, 10)
            self.create_timer(0.02, self._watchdog)

    def _js_cb(self, msg: JointState):
        if len(msg.name) != len(msg.position):
            return
        if not any(msg.position) and not any(msg.effort):
            return      # driver lost the robot (all zeros incl. effort) -> treated as stale
        pos = dict(zip(msg.name, msg.position))
        vel = dict(zip(msg.name, msg.velocity)) if len(msg.velocity) == len(msg.name) else {}
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        with self.lock:
            self.js = (pos, vel, stamp, time.monotonic())

    def _pose_cb(self, msg: PoseStamped):
        p, o = msg.pose.position, msg.pose.orientation
        with self.lock:
            self.tool_pose = (np.array([p.x, p.y, p.z]), np.array([o.x, o.y, o.z, o.w]))

    def _urdf_cb(self, msg: String):
        with self.lock:
            self.urdf = msg.data

    def snapshot(self):
        with self.lock:
            return self.js, self.tool_pose

    def publish(self, qd):
        if self.pub is None:
            return
        qd = [float(v) for v in qd]
        if len(qd) != 6 or not all(math.isfinite(v) for v in qd):
            qd = [0.0] * 6
        msg = Float64MultiArray()
        msg.data = qd
        with self.pub_lock:
            self.pub.publish(msg)

    def _watchdog(self):
        if not self.armed:
            return
        if time.monotonic() - self.heartbeat > LOOP_STALE_S:
            if not self.watchdog_zeroing:
                self.get_logger().error("UI loop stalled -> publishing zero velocity")
            self.watchdog_zeroing = True
            self.publish([0.0] * 6)
        else:
            self.watchdog_zeroing = False


def wait_for(node: TeleopNode, timeout: float = 10.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        js, _ = node.snapshot()
        if node.urdf and js is not None:
            return True
        time.sleep(0.05)
    return False


def q_from_js(js):
    pos = js[0]
    return np.array([float(pos[n]) for n in JOINTS])


# ----------------------------------------------------------------------------
# Self test (no window, no publisher)
# ----------------------------------------------------------------------------
def selftest(node: TeleopNode, tip_link: str, tip_offset):
    print("[selftest] waiting for /robot_description and /joint_states ...")
    if not wait_for(node):
        print("[selftest] FAIL: no robot_description or joint_states within 10 s")
        return 1
    kin0 = Kinematics(node.urdf, tip_link, (0.0, 0.0, 0.0))
    kin = Kinematics(node.urdf, tip_link, tip_offset)
    print(f"[selftest] chain joints OK: {JOINTS}")
    print(f"[selftest] joint limits lower={np.round(kin.lower, 3)} upper={np.round(kin.upper, 3)}")
    time.sleep(0.3)
    js, tool = node.snapshot()
    q = q_from_js(js)
    print(f"[selftest] q (joint1..6) = {np.round(q, 4)}")
    p0, _, quat0 = kin0.forward(q)
    if tool is None:
        print("[selftest] WARN: no /tool_end_effector_pose received, skipping FK cross-check")
    else:
        err_mm = 1000 * np.linalg.norm(p0 - tool[0])
        qd = abs(float(np.dot(quat0, tool[1])))
        ang_deg = math.degrees(2 * math.acos(min(1.0, qd)))
        print(f"[selftest] FK(rq_tool_tip) vs /tool_end_effector_pose: pos err {err_mm:.2f} mm, "
              f"rot err {ang_deg:.2f} deg -> {'OK' if err_mm < 2 and ang_deg < 1 else 'MISMATCH'}")
    p, R, _ = kin.forward(q)
    print(f"[selftest] stick tip (offset {tip_offset}) = {np.round(p, 4)} m")
    print(f"[selftest] stick direction (tool z in base) = {np.round(R[:, 2], 3)}")
    # Jacobian vs finite difference
    J = kin.jacobian(q)
    Jn = np.zeros((3, 6))
    for i in range(6):
        dq = np.zeros(6)
        dq[i] = 1e-6
        Jn[:, i] = (kin.forward(q + dq)[0] - kin.forward(q - dq)[0]) / 2e-6
    print(f"[selftest] Jacobian linear part vs finite diff max err = {np.abs(J[:3] - Jn).max():.2e}")
    s = np.linalg.svd(J, compute_uv=False)
    print(f"[selftest] singular values = {np.round(s, 4)}")
    for (v, w) in SPEEDS:
        m_lin = m_rot = 0.0
        for ax in range(3):
            x = np.zeros(6)
            x[ax] = v
            m_lin = max(m_lin, float(np.abs(dls(J, x)[0]).max()))
            x = np.zeros(6)
            x[3 + ax] = w
            m_rot = max(m_rot, float(np.abs(dls(J, x)[0]).max()))
        print(f"[selftest] level {v*1000:.0f} mm/s / {w} rad/s: worst-axis max |qd| "
              f"translate {m_lin:.3f}, rotate {m_rot:.3f} rad/s (cap {QD_MAX})")
    print("[selftest] done (nothing was published)")
    return 0


# ----------------------------------------------------------------------------
# Teleop UI loop
# ----------------------------------------------------------------------------
KEYMAP = None  # filled after pygame import


def run_ui(node: TeleopNode, tip_link: str, tip_offset, dry_run: bool):
    import pygame
    print("[teleop] waiting for /robot_description and /joint_states ...")
    if not wait_for(node):
        print("[teleop] FAIL: no robot_description or joint_states within 10 s")
        return 1
    kin = Kinematics(node.urdf, tip_link, tip_offset)

    pygame.init()
    screen = pygame.display.set_mode((820, 560))
    pygame.display.set_caption("Group6 HEAL teleop" + (" [DRY RUN]" if dry_run else " [LIVE]"))
    font = pygame.font.SysFont("monospace", 17)
    clock = pygame.time.Clock()
    K = pygame

    speed_idx = 0
    frame = "base"
    qd_prev = np.zeros(6)
    p_hold = None
    R_hold = None
    rec_file = None
    rec_writer = None
    rec_path = None
    rec_t0 = None
    n_marks = 0
    mark_pending = False
    status = ""
    running = True
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    def start_recording():
        nonlocal rec_file, rec_writer, rec_path, rec_t0, n_marks
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rec_path = DATA_DIR / f"teach_{stamp}.csv"
        rec_file = open(rec_path, "w", newline="")
        rec_writer = csv.writer(rec_file)
        rec_writer.writerow(
            ["t_wall", "t_rel", "js_stamp", "mark"]
            + [f"q{i}" for i in range(1, 7)] + [f"qd_meas{i}" for i in range(1, 7)]
            + [f"qd_cmd{i}" for i in range(1, 7)]
            + ["tip_x", "tip_y", "tip_z", "tip_qx", "tip_qy", "tip_qz", "tip_qw"]
            + ["tool_x", "tool_y", "tool_z", "tool_qx", "tool_qy", "tool_qz", "tool_qw"]
            + ["speed_level", "frame", "keys", "fault"])
        meta = {"tip_link": tip_link, "tip_offset_m": list(tip_offset), "joints": JOINTS,
                "rate_hz": RATE_HZ, "speeds": SPEEDS, "dry_run": dry_run,
                "started": stamp}
        rec_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        rec_t0 = time.time()
        n_marks = 0

    def stop_recording():
        nonlocal rec_file, rec_writer
        if rec_file:
            rec_file.close()
        rec_file = rec_writer = None

    lin_keys = [(K.K_w, K.K_s), (K.K_a, K.K_d), (K.K_r, K.K_f)]
    rot_keys = [(K.K_j, K.K_l), (K.K_i, K.K_k), (K.K_u, K.K_o)]
    key_names = "wsadrfjlikuo"
    key_codes = [K.K_w, K.K_s, K.K_a, K.K_d, K.K_r, K.K_f, K.K_j, K.K_l, K.K_i, K.K_k, K.K_u, K.K_o]

    try:
        node.heartbeat = time.monotonic()
        node.armed = True
        while running:
            node.heartbeat = time.monotonic()
            for ev in pygame.event.get():
                if ev.type == K.QUIT:
                    running = False
                elif ev.type == K.KEYDOWN:
                    if ev.key == K.K_ESCAPE:
                        running = False
                    elif ev.key in (K.K_1, K.K_2, K.K_3):
                        speed_idx = ev.key - K.K_1
                    elif ev.key == K.K_TAB:
                        frame = "tool" if frame == "base" else "base"
                        p_hold = R_hold = None
                    elif ev.key in (K.K_RETURN, K.K_KP_ENTER):
                        if rec_writer is None:
                            start_recording()
                            status = f"RECORDING -> {rec_path.name}"
                        else:
                            stop_recording()
                            status = f"saved {rec_path.name} ({n_marks} marks)"
                    elif ev.key == K.K_p:
                        if rec_writer is not None:
                            mark_pending = True
                        else:
                            status = "P ignored: not recording"

            keys = pygame.key.get_pressed()
            focused = pygame.key.get_focused()
            brake = keys[K.K_SPACE]
            js, tool = node.snapshot()
            now = time.monotonic()

            fault = ""
            if js is None or now - js[3] > JS_STALE_S:
                fault = "joint_states stale"
            q = q_from_js(js) if js is not None else np.zeros(6)
            p, R, quat = kin.forward(q)
            J = kin.jacobian(q)

            lin = np.array([keys[a] - keys[b] for a, b in lin_keys], dtype=float)
            rot = np.array([keys[a] - keys[b] for a, b in rot_keys], dtype=float)
            moving_lin = bool(np.any(lin))
            moving_rot = bool(np.any(rot))
            smin = float(np.linalg.svd(J, compute_uv=False)[-1])

            target = np.zeros(6)
            if fault or brake or not focused or not (moving_lin or moving_rot):
                p_hold = R_hold = None
            else:
                v_s, w_s = SPEEDS[speed_idx]
                v = lin / max(np.linalg.norm(lin), 1.0) * v_s
                w = rot / max(np.linalg.norm(rot), 1.0) * w_s
                if frame == "tool":
                    v, w = R @ v, R @ w
                if moving_lin:
                    p_hold = None
                else:
                    if p_hold is None:
                        p_hold = p.copy()
                    v = v + clip_norm(HOLD_KP * (p_hold - p), HOLD_V_MAX)
                if moving_rot:
                    R_hold = None
                else:
                    if R_hold is None:
                        R_hold = R.copy()
                    w = w + clip_norm(HOLD_KR * rot_log(R_hold @ R.T), HOLD_W_MAX)
                qd, smin = dls(J, np.concatenate([v, w]))
                m = float(np.abs(qd).max())
                if m > QD_MAX:
                    qd *= QD_MAX / m
                for i in range(6):
                    if (q[i] > kin.upper[i] - LIMIT_MARGIN and qd[i] > 0) or \
                       (q[i] < kin.lower[i] + LIMIT_MARGIN and qd[i] < 0):
                        fault = f"{JOINTS[i]} near limit ({q[i]:.2f})"
                if not np.all(np.isfinite(qd)):
                    fault = "non-finite IK"
                if not fault:
                    target = qd

            if "stale" in fault or fault.startswith("non-finite"):
                qd_out = np.zeros(6)      # hard stop
            else:
                up = np.abs(target) > np.abs(qd_prev)
                lim = np.where(up, ACC_UP, ACC_DOWN) * DT
                qd_out = qd_prev + np.clip(target - qd_prev, -lim, lim)
                qd_out[np.abs(qd_out) < 1e-5] = 0.0
            qd_prev = qd_out
            if not dry_run:
                node.publish(qd_out)

            if rec_writer is not None and js is not None:
                vel = js[1]
                qd_meas = [float(vel.get(n, float("nan"))) for n in JOINTS]
                tp = tool if tool is not None else (np.full(3, np.nan), np.full(4, np.nan))
                pressed = "".join(c for c, kc in zip(key_names, key_codes) if keys[kc])
                rec_writer.writerow(
                    [f"{time.time():.6f}", f"{time.time() - rec_t0:.4f}", f"{js[2]:.6f}",
                     int(mark_pending)]
                    + [f"{x:.6f}" for x in q] + [f"{x:.6f}" for x in qd_meas]
                    + [f"{x:.6f}" for x in qd_out]
                    + [f"{x:.6f}" for x in p] + [f"{x:.6f}" for x in quat]
                    + [f"{x:.6f}" for x in tp[0]] + [f"{x:.6f}" for x in tp[1]]
                    + [speed_idx + 1, frame, pressed, fault])
                if mark_pending:
                    n_marks += 1
                    status = f"RECORDING -> {rec_path.name}  (mark {n_marks} saved)"
                    mark_pending = False

            # ---- HUD ----
            screen.fill((20, 20, 24))
            v_s, w_s = SPEEDS[speed_idx]
            lines = [
                ("DRY RUN - nothing published" if dry_run else "LIVE - commanding robot",
                 (255, 200, 0) if dry_run else (255, 80, 80)),
                (f"speed {speed_idx + 1}: {v_s * 1000:.0f} mm/s, {w_s:.2f} rad/s     frame: {frame.upper()}",
                 (220, 220, 220)),
                (f"stick tip  x={p[0]:+.4f} y={p[1]:+.4f} z={p[2]:+.4f} m", (220, 220, 220)),
                (f"q   = {np.array2string(q, precision=3, suppress_small=True)}", (180, 180, 180)),
                (f"qd  = {np.array2string(qd_out, precision=3, suppress_small=True)}", (180, 180, 180)),
                (f"sigma_min = {smin:.3f}" + ("  (near singularity)" if smin < DLS_EPS else ""),
                 (255, 160, 0) if smin < DLS_EPS else (180, 180, 180)),
                (f"FAULT: {fault}" if fault else ("window not focused - stopped" if not focused else "ok"),
                 (255, 80, 80) if (fault or not focused) else (120, 220, 120)),
                (status or "not recording", (255, 80, 80) if rec_writer is not None else (180, 180, 180)),
                ("", (0, 0, 0)),
                ("W/S +x/-x  A/D +y/-y  R/F +z/-z     (hold to move)", (150, 150, 255)),
                ("J/L rx   I/K ry   U/O rz   (about stick tip)", (150, 150, 255)),
                ("1/2/3 speed  TAB frame  SPACE brake", (150, 150, 255)),
                ("ENTER rec start/stop   P mark waypoint   ESC quit", (150, 150, 255)),
            ]
            for i, (txt, col) in enumerate(lines):
                screen.blit(font.render(txt, True, col), (16, 16 + 30 * i))
            pygame.display.flip()
            clock.tick(RATE_HZ)
    finally:
        node.armed = False
        for _ in range(10):
            node.publish([0.0] * 6)
            time.sleep(0.01)
        stop_recording()
        pygame.quit()
        if rec_path:
            print(f"[teleop] last recording: {rec_path}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--selftest", action="store_true")
    g.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tip-link", default=DEFAULT_TIP_LINK)
    ap.add_argument("--tip-offset", type=float, nargs=3, default=list(DEFAULT_TIP_OFFSET),
                    metavar=("X", "Y", "Z"), help="stick tip offset in tip-link frame [m]")
    args = ap.parse_args()

    from rclpy.signals import SignalHandlerOptions
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)   # Ctrl-C -> our finally (zeros) first
    live = not (args.selftest or args.dry_run)
    node = TeleopNode(publish=live)
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    spin = threading.Thread(target=ex.spin, daemon=True)
    spin.start()
    rc = 1
    try:
        if args.selftest:
            rc = selftest(node, args.tip_link, tuple(args.tip_offset))
        else:
            rc = run_ui(node, args.tip_link, tuple(args.tip_offset), dry_run=args.dry_run)
    except KeyboardInterrupt:
        rc = 0
    finally:
        node.publish([0.0] * 6)
        time.sleep(0.05)
        ex.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
