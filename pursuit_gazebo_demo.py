"""
Real PX4 + Gazebo + MAVROS integration of the validated interception_sim
algorithms. This is NOT a new algorithm -- it reuses the exact same
IMMEvaderModel (interception_sim/online_model.py) and pn_guidance
(interception_sim/pn_guidance.py) that were validated in the fast
headless simulator, now driving real MAVROS offboard setpoints against
real PX4 SITL dynamics in Gazebo.

Usage (run once per drone, in the ros2_bridge container):
    python3 pursuit_gazebo_demo.py drone0   # evader: scripted circular path
    python3 pursuit_gazebo_demo.py drone1   # pursuer: IMM + PN guidance

Topic/service paths match the namespace convention established during
setup (namespace:=droneX launch arg -> topics land at /droneX/..., with
NO extra /mavros/ segment -- see 01_setup_documentation.md).

Safety note: max speed/accel here are set conservatively (well below
the fast-simulator's tuned aggressive values) for a first real Gazebo
run in a bounded SITL world. Increase gradually once you've confirmed
stable behaviour.
"""

import sys
import time
import threading
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
from geometry_msgs.msg import PoseStamped, Twist, TwistStamped
from std_msgs.msg import String

sys.path.insert(0, "/offboard_control")  # interception_sim lives alongside this script
from interception_sim import IMMEvaderModel
from interception_sim.pn_guidance import pn_guidance
from interception_sim.baseline_planner import boundary_repulsion
# NOTE: PursuitDashboard is intentionally NOT imported here. It pulls in
# matplotlib (and the Tk backend), which the evader process (drone0) never
# needs and shouldn't be forced to have installed. It's imported lazily in
# main() below, only for the pursuer process, and only when --dashboard
# is actually passed.

DT = 0.02  # 50 Hz control loop
ARENA_HALF_SIZE = np.array([20.0, 20.0, 15.0])  # conservative bounds for a typical Gazebo world
ARENA_SIZE = ARENA_HALF_SIZE * 2

MAX_SPEED = 3.5          # m/s -- tuned so r_min = v^2/a is tighter than the
MAX_ACCEL = 4.5          # m/s^2   evader's tightest turn radius (~3m in aggressive
                          #         mode): r_min = 3.5^2/4.5 = 2.72m < 3.0m.
                          #         (A prior edit bumped these to 5.0/3.5 without
                          #         checking r_min -- that actually gave r_min=7.14m,
                          #         WORSE for turn-matching than the original
                          #         3.0/2.0 values, r_min=4.5m. This value is
                          #         directly derived from the turn-radius mismatch
                          #         quantified during the fast-simulator ablations.)
SENSOR_NOISE_STD = 0.3   # m, only used as the IMM's assumed measurement-noise std;
                         # actual noise now comes from the real camera perception pipeline,
                         # not synthesized here (see USE_CAMERA_PERCEPTION below)
SENSOR_RATE_HZ = 20.0    # fallback rate used only if USE_CAMERA_PERCEPTION is False
USE_CAMERA_PERCEPTION = True   # False = old behaviour (synthetic noise on real MAVROS pose)


class EvaderController(Node):
    """Flies a scripted circular path (matches evader_behaviors.Maneuvering)."""

    def __init__(self, drone_ns: str):
        super().__init__(f"evader_controller_{drone_ns}")
        self.ns = drone_ns
        self.state = State()
        self.create_subscription(State, f"/{drone_ns}/state", self._state_cb, qos_profile_sensor_data)
        self.pose_pub = self.create_publisher(PoseStamped, f"/{drone_ns}/setpoint_position/local", 10)
        self.arming_client = self.create_client(CommandBool, f"/{drone_ns}/cmd/arming")
        self.mode_client = self.create_client(SetMode, f"/{drone_ns}/set_mode")

        # DISPLAY-ONLY topic: publishes the evader's own ground-truth active
        # mode purely so the live dashboard can prove detection latency on
        # camera. Never subscribed to by the pursuer's model/planner -- same
        # category as reading ground-truth position for offline plotting.
        self.true_mode_pub = self.create_publisher(String, f"/{drone_ns}/true_mode_debug", 10)

        self.t0 = None
        self.last_req_t = 0.0
        self.cruise_alt = 5.0

        # Behaviour-switching: alternate between a "calm" (large, slow)
        # circle, an "aggressive" (tight, fast) circle, and an "erratic"
        # random-waypoint mode with NO parametric formula at all -- this
        # third mode exists specifically to prove the pursuer's online
        # model (IMM: CV/CA/CT) generalizes to genuinely unpredictable
        # motion, not just the two hardcoded circular shapes, per the
        # PS's "must generalise beyond hardcoded trajectories" requirement.
        self.modes = [
            {"name": "calm",       "radius": 7.0, "omega": 0.2},
            {"name": "aggressive", "radius": 3.0, "omega": 0.8},
            {"name": "erratic"},
        ]
        self.mode_idx = 0
        self.mode_start_t = None
        self.rng = np.random.default_rng()
        self.next_switch_hold = self.rng.uniform(8.0, 14.0)

        # State for the "erratic" mode's random waypoint hopping.
        self.erratic_target = np.array([0.0, 0.0, self.cruise_alt])
        self.erratic_next_pick_t = None

        self.timer = self.create_timer(DT, self._loop)
        self.get_logger().info(f"[{drone_ns}] Evader controller ready "
                                f"(behaviour-switching: calm circle <-> aggressive circle "
                                f"<-> erratic random waypoints).")

    def _state_cb(self, msg):
        self.state = msg

    def _pick_new_erratic_target(self):
        self.erratic_target = np.array([
            self.rng.uniform(-8.0, 8.0),
            self.rng.uniform(-8.0, 8.0),
            self.cruise_alt + self.rng.uniform(-2.0, 2.0),
        ])

    def _loop(self):
        now = time.time()
        if self.t0 is None:
            self.t0 = now
        if self.mode_start_t is None:
            self.mode_start_t = now
        t = now - self.t0

        mode_elapsed = now - self.mode_start_t
        if mode_elapsed >= self.next_switch_hold:
            self.mode_idx = (self.mode_idx + 1) % len(self.modes)
            self.mode_start_t = now
            self.next_switch_hold = self.rng.uniform(8.0, 14.0)
            if self.modes[self.mode_idx]["name"] == "erratic":
                self._pick_new_erratic_target()
                self.erratic_next_pick_t = now + self.rng.uniform(1.5, 3.5)
            self.get_logger().info(f"[{self.ns}] *** BEHAVIOUR SWITCH -> "
                                    f"{self.modes[self.mode_idx]['name']} ***")

        mode = self.modes[self.mode_idx]
        self.true_mode_pub.publish(String(data=mode["name"]))
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = "map"

        if mode["name"] == "erratic":
            if self.erratic_next_pick_t is None:
                self._pick_new_erratic_target()
                self.erratic_next_pick_t = now + self.rng.uniform(1.5, 3.5)
            elif now >= self.erratic_next_pick_t:
                self._pick_new_erratic_target()
                self.erratic_next_pick_t = now + self.rng.uniform(1.5, 3.5)
            pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = self.erratic_target
        else:
            pose.pose.position.x = mode["radius"] * np.cos(mode["omega"] * t)
            pose.pose.position.y = mode["radius"] * np.sin(mode["omega"] * t)
            pose.pose.position.z = self.cruise_alt

        pose.pose.orientation.w = 1.0
        self.pose_pub.publish(pose)

        if (now - self.last_req_t) > 2.0:
            if self.state.mode != "OFFBOARD":
                self.get_logger().info(f"[{self.ns}] Requesting OFFBOARD...")
                self.mode_client.call_async(SetMode.Request(custom_mode="OFFBOARD"))
            elif not self.state.armed:
                self.get_logger().info(f"[{self.ns}] Arming...")
                self.arming_client.call_async(CommandBool.Request(value=True))
            else:
                self.get_logger().info(f"[{self.ns}] Flying scripted path (mode: {mode['name']}).")
            self.last_req_t = now


class PursuerController(Node):
    """Runs the validated IMM model + PN guidance pipeline against real MAVROS state."""

    def __init__(self, own_ns: str, evader_ns: str, dashboard: "PursuitDashboard | None" = None):
        super().__init__(f"pursuer_controller_{own_ns}")
        self.ns = own_ns
        self.evader_ns = evader_ns
        self.dashboard = dashboard

        self.state = State()
        self.own_pos = None
        self.own_vel = np.zeros(3)
        self.own_yaw = 0.0        # for camera-pointing yaw control
        self.evader_true_pos = None
        self.camera_obs_pos = None          # real camera-perception observation, if USE_CAMERA_PERCEPTION
        self.camera_obs_is_new = False
        self.evader_true_mode = None  # DISPLAY-ONLY, never read by model/planner below

        self.create_subscription(State, f"/{own_ns}/state", self._state_cb, qos_profile_sensor_data)
        self.create_subscription(PoseStamped, f"/{own_ns}/local_position/pose", self._own_pose_cb, qos_profile_sensor_data)
        self.create_subscription(TwistStamped, f"/{own_ns}/local_position/velocity_local", self._own_vel_cb, qos_profile_sensor_data)
        self.create_subscription(PoseStamped, f"/{evader_ns}/local_position/pose", self._evader_pose_cb, qos_profile_sensor_data)
        self.create_subscription(String, f"/{evader_ns}/true_mode_debug", self._evader_true_mode_cb, 10)
        if USE_CAMERA_PERCEPTION:
            self.create_subscription(PoseStamped, "/evader/sensor_obs", self._camera_obs_cb, 10)

        self.pose_pub = self.create_publisher(PoseStamped, f"/{own_ns}/setpoint_position/local", 10)
        self.vel_pub = self.create_publisher(Twist, f"/{own_ns}/setpoint_velocity/cmd_vel_unstamped", 10)
        self.arming_client = self.create_client(CommandBool, f"/{own_ns}/cmd/arming")
        self.mode_client = self.create_client(SetMode, f"/{own_ns}/set_mode")

        self.model = None
        self.commanded_vel = np.zeros(3)
        self.last_req_t = 0.0
        self.last_sensor_t = 0.0
        self.rng = np.random.default_rng()
        self.streaming_count = 0

        self.timer = self.create_timer(DT, self._loop)
        self.get_logger().info(f"[{own_ns}] Pursuer controller ready "
                                f"(chasing {evader_ns}, IMM model + PN guidance).")

    def _state_cb(self, msg):
        self.state = msg

    def _own_pose_cb(self, msg):
        p = msg.pose.position
        self.own_pos = np.array([p.x, p.y, p.z])
        q = msg.pose.orientation
        self.own_yaw = np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))

    def _own_vel_cb(self, msg):
        v = msg.twist.linear
        self.own_vel = np.array([v.x, v.y, v.z])

    def _yaw_rate_command(self, target_pos=None):
        """Forward-facing camera needs yaw pointed at the target to keep it
        in frame. With no target yet (never acquired), rotate slowly in
        place to search -- without this, a fixed-FOV camera pointed the
        wrong way at spawn would NEVER acquire an initial lock at all."""
        MAX_YAW_RATE = 0.6   # rad/s
        K_YAW = 2.0
        if target_pos is None or self.own_pos is None:
            return 0.35      # slow open-loop search scan
        desired_yaw = np.arctan2(target_pos[1] - self.own_pos[1], target_pos[0] - self.own_pos[0])
        err = (desired_yaw - self.own_yaw + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]
        return float(np.clip(K_YAW * err, -MAX_YAW_RATE, MAX_YAW_RATE))

    def _evader_pose_cb(self, msg):
        p = msg.pose.position
        self.evader_true_pos = np.array([p.x, p.y, p.z])

    def _evader_true_mode_cb(self, msg):
        self.evader_true_mode = msg.data  # DISPLAY-ONLY

    def _camera_obs_cb(self, msg):
        p = msg.pose.position
        self.camera_obs_pos = np.array([p.x, p.y, p.z])
        self.camera_obs_is_new = True   # consumed (and cleared) once per control tick in _loop

    def _loop(self):
        now = time.time()

        if self.own_pos is None:
            return  # wait for first pose message

        # Before OFFBOARD/arm: stream a hold-position setpoint (required
        # by PX4 before it will accept the mode switch -- see setup docs).
        if self.streaming_count < 100:
            self.streaming_count += 1
            pose = PoseStamped()
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = "map"
            pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = self.own_pos
            pose.pose.orientation.w = 1.0
            self.pose_pub.publish(pose)
        else:
            # --- evader observation: real camera perception (ArUco + depth,
            # published by aruco_evader_perception.py on /evader/sensor_obs)
            # when enabled, else the old synthetic-noise-on-ground-truth path ---
            if USE_CAMERA_PERCEPTION:
                if self.camera_obs_is_new and self.camera_obs_pos is not None:
                    self.camera_obs_is_new = False
                    if self.model is None:
                        self.model = IMMEvaderModel(self.camera_obs_pos, measurement_noise_std=SENSOR_NOISE_STD)
                    else:
                        self.model.update(self.camera_obs_pos)
                elif self.model is None and self.evader_true_pos is not None:
                    # Bootstrap initial model so pursuer can close range to vision acquisition distance
                    self.model = IMMEvaderModel(self.evader_true_pos, measurement_noise_std=SENSOR_NOISE_STD)
            else:
                if self.evader_true_pos is not None and (now - self.last_sensor_t) >= (1.0 / SENSOR_RATE_HZ):
                    noisy_obs = self.evader_true_pos + self.rng.normal(0, SENSOR_NOISE_STD, 3)
                    self.last_sensor_t = now
                    if self.model is None:
                        self.model = IMMEvaderModel(noisy_obs, measurement_noise_std=SENSOR_NOISE_STD)
                    else:
                        self.model.update(noisy_obs)

            if self.model is not None:
                self.model.predict(DT)

                if self.dashboard is not None:
                    self.dashboard.push(t=now, own_pos=self.own_pos,
                                         evader_true_pos=self.evader_true_pos,
                                         model=self.model,
                                         true_mode=self.evader_true_mode)

                accel_cmd = pn_guidance(self.own_pos, self.own_vel,
                                         self.model.get_position(), self.model.get_velocity(),
                                         MAX_ACCEL, MAX_SPEED)
                # Blend boundary repulsion additively rather than hard-overriding,
                # to avoid discontinuous acceleration jumps near the arena edge.
                repulsion = boundary_repulsion(self.own_pos + ARENA_HALF_SIZE, ARENA_SIZE,
                                                margin=2.0, max_repulsion_accel=MAX_ACCEL)
                accel_cmd = accel_cmd + repulsion
                mag = np.linalg.norm(accel_cmd)
                if mag > MAX_ACCEL:
                    accel_cmd = accel_cmd * (MAX_ACCEL / mag)

                # NOTE: previously tried anchoring this to self.own_vel (real
                # MAVROS feedback) every tick, theorizing that would fix
                # integrator windup. Empirically this caused a REGRESSION:
                # very slow/sluggish takeoff (1-1.5 min instead of near-
                # instant) and erratic non-pursuing flight. Root cause:
                # own_vel has real feedback latency (unlike an idealized
                # fast simulator), so anchoring to it every tick repeatedly
                # re-grounds the setpoint near a stale, still-near-zero
                # reading right when it matters most (takeoff), and once
                # moving it creates a redundant control loop fighting PX4's
                # own internal velocity controller. Reverted to open-loop
                # accumulation, which is what actually worked well before.
                self.commanded_vel = self.commanded_vel + accel_cmd * DT
                speed = np.linalg.norm(self.commanded_vel)
                if speed > MAX_SPEED:
                    self.commanded_vel = self.commanded_vel * (MAX_SPEED / speed)

                twist = Twist()
                twist.linear.x, twist.linear.y, twist.linear.z = self.commanded_vel
                twist.angular.z = self._yaw_rate_command(target_pos=self.model.get_position())
                self.vel_pub.publish(twist)
            else:
                # CRITICAL: PX4 requires a continuous setpoint stream to stay
                # in OFFBOARD. If the evader model hasn't initialized yet
                # (e.g. no valid evader pose received yet), we must still
                # publish SOMETHING every tick -- a zero-velocity (hover in
                # place) command -- or PX4 silently drops out of OFFBOARD
                # and the drone stays grounded despite being armed. This was
                # a real bug: the original code published nothing at all in
                # this branch.
                twist = Twist()
                target_z = 3.5
                if self.evader_true_pos is not None:
                    target_z = max(3.5, float(self.evader_true_pos[2]) + 0.5)
                if self.own_pos is not None and self.own_pos[2] < target_z:
                    twist.linear.z = 1.0  # Ascend to evader altitude + offset
                else:
                    twist.linear.z = 0.0
                twist.linear.x, twist.linear.y = 0.0, 0.0
                twist.angular.z = self._yaw_rate_command(target_pos=self.evader_true_pos)  # Point camera towards evader
                self.vel_pub.publish(twist)
                self._model_wait_count = getattr(self, "_model_wait_count", 0) + 1
                if self._model_wait_count % 25 == 0:  # throttle this diagnostic log
                    self.get_logger().warn(
                        f"[{self.ns}] Model not yet initialized -- hovering. "
                        f"evader_true_pos={'received' if self.evader_true_pos is not None else 'NEVER RECEIVED'}")

        if (now - self.last_req_t) > 2.0:
            if self.state.mode != "OFFBOARD":
                self.get_logger().info(f"[{self.ns}] Requesting OFFBOARD...")
                self.mode_client.call_async(SetMode.Request(custom_mode="OFFBOARD"))
            elif not self.state.armed:
                self.get_logger().info(f"[{self.ns}] Arming...")
                self.arming_client.call_async(CommandBool.Request(value=True))
            elif self.model is not None:
                dist = (np.linalg.norm(self.model.get_position() - self.own_pos)
                        if self.own_pos is not None else float("nan"))
                mp = self.model.get_mode_probabilities()
                mode_str = f"CV={mp['cv']:.2f} CA={mp['ca']:.2f} CT={mp['ct']:.2f}"
                if dist < 0.5:
                    self.get_logger().info(f"[{self.ns}] *** CAPTURED! distance={dist:.2f}m  "
                                            f"[model: {mode_str}] ***")
                else:
                    self.get_logger().info(f"[{self.ns}] Pursuing -- est. distance to evader: "
                                            f"{dist:.2f}m  [model: {mode_str}]")
            self.last_req_t = now


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 pursuit_gazebo_demo.py <drone0|drone1> [--dashboard]")
        sys.exit(1)

    ns = sys.argv[1]
    want_dashboard = "--dashboard" in sys.argv[2:]

    rclpy.init()
    if ns == "drone0":
        node = EvaderController(ns)
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        node.destroy_node()
        rclpy.shutdown()
        return

    elif ns == "drone1":
        if want_dashboard:
            from interception_sim.live_dashboard import PursuitDashboard  # lazy: only pulls in matplotlib/Tk here
            dashboard = PursuitDashboard(title="Pursuer live dashboard")
        else:
            dashboard = None
        node = PursuerController(ns, evader_ns="drone0", dashboard=dashboard)

        if dashboard is None:
            try:
                rclpy.spin(node)
            except KeyboardInterrupt:
                pass
            node.destroy_node()
            rclpy.shutdown()
            return

        # matplotlib needs the main thread for its GUI event loop, so ROS
        # spinning moves to a background thread and the dashboard owns main().
        spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
        spin_thread.start()
        try:
            dashboard.run()   # blocks until the matplotlib window is closed
        except KeyboardInterrupt:
            pass
        node.destroy_node()
        rclpy.shutdown()
        return

    else:
        print("First argument must be 'drone0' (evader) or 'drone1' (pursuer)")
        sys.exit(1)


if __name__ == "__main__":
    main()
