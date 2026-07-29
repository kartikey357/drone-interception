#!/usr/bin/env python3
"""
aruco_evader_perception.py
--------------------------------------------------------------
REPLACES evader_sensor_sim.py. Instead of taking the evader's ground-truth
position and adding synthetic noise, this node genuinely PERCEIVES the
evader through a simulated depth camera mounted on the pursuer:

  RGB image --> detect ArUco marker on the evader --> pixel centroid
  depth image --> sample depth at that pixel (robust median over a patch)
  pinhole back-projection --> 3D point in the camera's OPTICAL frame
  tf2 transform (camera_optical -> map) --> 3D point in WORLD frame
  --> published as PoseStamped on /evader/sensor_obs   (SAME topic/type
      the rest of the pipeline -- estimator/IMM/planner -- already expects,
      so nothing downstream needs to change)

Frame chain (standard ROS/REP-103/105 conventions):
    map  --(dynamic, from pursuer's own MAVROS pose)-->  <ns>_base_link
         --(static, camera physically mounted facing forward)--> <ns>_camera_link
         --(static, REP-103 optical-frame rotation)--> <ns>_camera_optical_frame

If your camera is NOT mounted facing straight forward with no tilt, edit
CAMERA_MOUNT_XYZ / CAMERA_MOUNT_RPY below to match your actual SDF mount pose
-- these two files must agree, or your 3D estimate will be geometrically wrong
even though nothing will error out. See the bottom of this file for a quick
calibration self-check you can run before trusting this for control.
--------------------------------------------------------------
"""

import sys
import numpy as np
import cv2
from cv2 import aruco

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped, TransformStamped, PointStamped
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import tf2_ros
import tf2_geometry_msgs  # noqa: F401  (registers PointStamped transform support)
from scipy.spatial.transform import Rotation as R


# ============================== CONFIG ==============================
PURSUER_NS = sys.argv[1] if len(sys.argv) > 1 else "drone1"

RGB_TOPIC    = f"/{PURSUER_NS}/camera/image"
DEPTH_TOPIC  = f"/{PURSUER_NS}/camera/depth_image"
INFO_TOPIC   = f"/{PURSUER_NS}/camera/camera_info"
POSE_TOPIC   = f"/{PURSUER_NS}/mavros/local_position/pose"
OUTPUT_TOPIC = "/evader/sensor_obs"          # unchanged -- drop-in replacement

ARUCO_DICT   = aruco.DICT_4X4_50
MARKER_ID    = 0                              # only trust this specific marker ID
DEPTH_PATCH  = 5                              # NxN pixel patch around centroid for robust depth sampling
MAX_RANGE_M  = 40.0                           # discard implausible depth readings (Gazebo depth noise/holes)

# Camera mount pose relative to the pursuer's body frame (base_link, FLU convention).
# EDIT THESE to match wherever you actually mount the camera in your SDF.
CAMERA_MOUNT_XYZ = (0.12, 0.0, 0.0)           # 12cm forward of body origin
CAMERA_MOUNT_RPY = (0.0, 0.0, 0.0)            # facing straight forward, no tilt
# ======================================================================


def yaw_from_quat(x, y, z, w):
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class ArucoEvaderPerception(Node):
    def __init__(self):
        super().__init__("aruco_evader_perception")
        self.bridge = CvBridge()
        self.aruco_dict = aruco.getPredefinedDictionary(ARUCO_DICT)
        # OpenCV >=4.7 uses the class-based ArucoDetector API; older builds
        # (e.g. the OpenCV 4.5.x that ships in Ubuntu 22.04's apt package)
        # use the old functional API. Support both rather than assuming one.
        if hasattr(aruco, "ArucoDetector"):
            self._new_api = True
            self.aruco_params = aruco.DetectorParameters()
            self.detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        else:
            self._new_api = False
            self.aruco_params = aruco.DetectorParameters_create()

        self.K = None            # camera intrinsics matrix, filled from CameraInfo
        self.latest_depth = None
        self.own_pose = None     # latest pursuer PoseStamped, for the dynamic TF

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.static_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        self._publish_static_camera_transform()

        self.create_subscription(CameraInfo, INFO_TOPIC, self.on_info, 10)
        self.create_subscription(Image, DEPTH_TOPIC, self.on_depth, qos_profile_sensor_data)
        self.create_subscription(PoseStamped, POSE_TOPIC, self.on_own_pose, qos_profile_sensor_data)
        self.create_subscription(Image, RGB_TOPIC, self.on_rgb, qos_profile_sensor_data)

        self.pub = self.create_publisher(PoseStamped, OUTPUT_TOPIC, 10)

        self.n_frames = 0
        self.n_detections = 0
        self.create_timer(5.0, self._log_stats)

        self.get_logger().info(f"[{PURSUER_NS}] ArUco camera perception node started. "
                                f"Watching for marker ID {MARKER_ID}.")

    # -------- static camera mount transform (base_link -> camera_link -> camera_optical_frame) --------
    def _publish_static_camera_transform(self):
        t1 = TransformStamped()
        t1.header.frame_id = f"{PURSUER_NS}_base_link"
        t1.child_frame_id = f"{PURSUER_NS}_camera_link"
        t1.transform.translation.x, t1.transform.translation.y, t1.transform.translation.z = CAMERA_MOUNT_XYZ
        q = R.from_euler("xyz", CAMERA_MOUNT_RPY).as_quat()  # [x,y,z,w]
        t1.transform.rotation.x, t1.transform.rotation.y, t1.transform.rotation.z, t1.transform.rotation.w = q

        # REP-103 standard: camera_link (x-forward,y-left,z-up) -> camera_optical_frame (x-right,y-down,z-forward)
        t2 = TransformStamped()
        t2.header.frame_id = f"{PURSUER_NS}_camera_link"
        t2.child_frame_id = f"{PURSUER_NS}_camera_optical_frame"
        t2.transform.rotation.x = -0.5
        t2.transform.rotation.y = 0.5
        t2.transform.rotation.z = -0.5
        t2.transform.rotation.w = 0.5

        self.static_broadcaster.sendTransform([t1, t2])

    # -------- callbacks --------
    def on_info(self, msg: CameraInfo):
        self.K = np.array(msg.k).reshape(3, 3)  # fx 0 cx / 0 fy cy / 0 0 1

    def on_depth(self, msg: Image):
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

    def on_own_pose(self, msg: PoseStamped):
        self.own_pose = msg
        # broadcast map -> pursuer_base_link so tf2 can chain it to the camera frames above
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "map"
        t.child_frame_id = f"{PURSUER_NS}_base_link"
        t.transform.translation.x = msg.pose.position.x
        t.transform.translation.y = msg.pose.position.y
        t.transform.translation.z = msg.pose.position.z
        t.transform.rotation = msg.pose.orientation
        self.tf_broadcaster.sendTransform(t)

    def on_rgb(self, msg: Image):
        self.n_frames += 1
        if self.K is None or self.latest_depth is None:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._new_api:
            corners, ids, _ = self.detector.detectMarkers(gray)
        else:
            corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        if ids is None:
            return  # no marker visible this frame -- natural "dropout", nothing published
        ids = ids.flatten()
        if MARKER_ID not in ids:
            return
        idx = int(np.where(ids == MARKER_ID)[0][0])
        c = corners[idx][0]  # 4x2 pixel corners
        u, v = c[:, 0].mean(), c[:, 1].mean()

        depth = self._robust_depth(int(round(u)), int(round(v)))
        if depth is None or not (0.05 < depth < MAX_RANGE_M):
            return

        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        x_cam = (u - cx) * depth / fx
        y_cam = (v - cy) * depth / fy
        z_cam = depth

        pt = PointStamped()
        pt.header.stamp = msg.header.stamp if msg.header.stamp.sec > 0 else self.get_clock().now().to_msg()
        pt.header.frame_id = f"{PURSUER_NS}_camera_optical_frame"
        pt.point.x, pt.point.y, pt.point.z = float(x_cam), float(y_cam), float(z_cam)

        try:
            pt_world = self.tf_buffer.transform(pt, "map", timeout=rclpy.duration.Duration(seconds=0.05))
        except Exception as e:
            self.get_logger().warn(f"TF transform failed (camera->map): {e}", throttle_duration_sec=2.0)
            return

        out = PoseStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = "map"
        out.pose.position = pt_world.point
        out.pose.orientation.w = 1.0  # orientation not estimated -- position-only measurement
        self.pub.publish(out)
        self.n_detections += 1

    def _robust_depth(self, u, v):
        h, w = self.latest_depth.shape[:2]
        r = DEPTH_PATCH // 2
        u0, u1 = max(0, u - r), min(w, u + r + 1)
        v0, v1 = max(0, v - r), min(h, v + r + 1)
        patch = self.latest_depth[v0:v1, u0:u1].astype(np.float32).flatten()
        patch = patch[np.isfinite(patch) & (patch > 0)]
        if patch.size == 0:
            return None
        return float(np.median(patch))

    def _log_stats(self):
        rate = (self.n_detections / max(1, self.n_frames)) * 100
        self.get_logger().info(f"[{PURSUER_NS}] {self.n_frames} frames, "
                                f"{self.n_detections} marker detections ({rate:.0f}% lock rate)")


def main():
    rclpy.init()
    node = ArucoEvaderPerception()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

# --------------------------------------------------------------------
# QUICK CALIBRATION SELF-CHECK (do this before trusting this for control):
# 1. Hover the pursuer stationary, facing the evader, at a KNOWN measured
#    distance (e.g. hover 5m away, evader hovering still too).
# 2. echo /evader/sensor_obs and compare against the evader's own
#    /mavros/local_position/pose. They should agree within ~10-20cm.
# 3. If they're consistently offset in one direction, your
#    CAMERA_MOUNT_XYZ/RPY likely don't match the real SDF mount pose --
#    fix the mismatch there, not by fudging the output afterward.
# --------------------------------------------------------------------
