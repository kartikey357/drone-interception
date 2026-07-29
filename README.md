# Autonomous Drone-to-Drone Interception with Online Evader Modeling

[![ROS 2](https://img.shields.io/badge/ROS%202-Humble-blue.svg)](https://docs.ros.org/en/humble/)
[![PX4 SITL](https://img.shields.io/badge/PX4-v1.14+-red.svg)](https://px4.io/)
[![Gazebo](https://img.shields.io/badge/Gazebo-Harmonic-orange.svg)](https://gazebosim.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An autonomous 3D quadrotor interception pipeline incorporating real-time RGB-D vision perception (ArUco marker detection), Interactive Multiple Model (IMM) online evader state estimation, and Proportional Navigation (PN) guidance running at 50 Hz.

---

## 📌 Features

- **3D Quadrotor Simulator:** Multi-agent PX4 SITL simulation in Gazebo Harmonic.
- **Vision Perception Node (`aruco_evader_perception.py`):** Real-time 20 Hz RGB-D ArUco marker centroid detection, 3D pinhole back-projection, and TF2 frame transformation into world coordinates.
- **Online Evader Model:** Interactive Multiple Model (IMM) estimator with mode probability tracking (Constant Velocity, Constant Acceleration, Coordinated Turn).
- **Trajectory Planner & Guidance:** Real-time Proportional Navigation (PN) trajectory generation operating at 50 Hz.
- **Automated Evaluation Suite (`run_evaluation_suite.py`):** Benchmarks across 100+ episodes per evader profile (non-maneuvering, maneuvering, and behavior-switching).

---

## 🛠️ System Architecture

```
                      ┌────────────────────────────────────────────────────────┐
                      │              Docker Container: px4_sim                 │
                      │  PX4 SITL Inst 0 (evader)  │ PX4 SITL Inst 1 (pursuer) │
                      │   gz_x500_marker @ (0,0)   │   gz_x500_camera @ (6,0)  │
                      └───────────────────────────┬────────────────────────────┘
                                                  │ (Shared Container Network)
                                                  ▼
                      ┌────────────────────────────────────────────────────────┐
                      │             Docker Container: ros2_bridge              │
                      │                                                        │
                      │ ├─ MAVROS drone0 (fcu: 14540/14580)                    │
                      │ ├─ MAVROS drone1 (fcu: 14541/14581)                    │
                      │ ├─ ros_gz_bridge (image, depth, camera_info)           │
                      │ ├─ Evader Offboard Node (drone0)                       │
                      │ ├─ ArUco Perception Node (drone1)                      │
                      │ └─ Pursuer Offboard Node + Dashboard (drone1)         │
                      └───────────────────────────┘
```

---

## 🚀 Quick Start Guide

### 0. Pre-Flight Setup (Host Terminal)
```bash
docker rm -f px4_sim ros2_bridge
xhost +local:docker
```

### 1. Terminal 1 — Evader Drone (PX4 Instance 0)
```bash
docker run -it --name px4_sim --network host -e DISPLAY=:0 -e LIBGL_ALWAYS_SOFTWARE=1 -v /tmp/.X11-unix:/tmp/.X11-unix:ro -v /dev/dri:/dev/dri -v /home/kartikey/px4_ros2_jazzy_ws/PX4-Autopilot:/src/PX4-Autopilot px4_harmonic_base:latest /bin/bash
```
*Inside container:*
```bash
cd /src/PX4-Autopilot
PX4_SYS_AUTOSTART=4001 PX4_SIM_MODEL=gz_x500_marker PX4_GZ_MODEL_POSE="0,0" ./build/px4_sitl_default/bin/px4 -i 0
```

### 2. Terminal 2 — Pursuer Drone (PX4 Instance 1)
```bash
docker exec -it px4_sim /bin/bash
```
*Inside container:*
```bash
cd /src/PX4-Autopilot
PX4_GZ_STANDALONE=1 PX4_SYS_AUTOSTART=4001 PX4_GZ_MODEL_POSE="6,0" PX4_SIM_MODEL=gz_x500_camera ./build/px4_sitl_default/bin/px4 -i 1
```

### 3. Terminal 3 — GCS Heartbeat
```bash
cd ~
python3 gcs_heartbeat.py
```

### 4. Terminal 4 — ROS 2 Bridge Container
```bash
docker rm -f ros2_bridge
docker run -it -d --network container:px4_sim -e DISPLAY=:0 -v /tmp/.X11-unix:/tmp/.X11-unix:ro -v /home/kartikey/ros2_ws/src/offboard_control:/offboard_control --name ros2_bridge ros2_mavros:humble /bin/bash
```

### 5. Terminal 5 & 6 — MAVROS Instances
- **Terminal 5 (drone0):**
  ```bash
  docker exec -it ros2_bridge /bin/bash -c "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && source /opt/ros/humble/setup.bash && ros2 launch mavros px4.launch fcu_url:='udp://127.0.0.1:14540@127.0.0.1:14580' tgt_system:=1 tgt_component:=1 namespace:=drone0"
  ```
- **Terminal 6 (drone1):**
  ```bash
  docker exec -it ros2_bridge /bin/bash -c "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && source /opt/ros/humble/setup.bash && ros2 launch mavros px4.launch fcu_url:='udp://127.0.0.1:14541@127.0.0.1:14581' tgt_system:=2 tgt_component:=1 namespace:=drone1"
  ```

### 6. Terminal 7 — Parameter Bridges
```bash
docker exec -it ros2_bridge /bin/bash -c "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && source /opt/ros/humble/setup.bash && source /tmp/bridge_ws/install/setup.bash && ros2 run ros_gz_bridge parameter_bridge /drone1/camera/image@sensor_msgs/msg/Image[gz.msgs.Image"
```

### 7. Terminal 8, 9 & 10 — Autonomous Pursuit
- **Terminal 8 (Evader Controller):**
  ```bash
  docker exec -it ros2_bridge /bin/bash -c "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && source /opt/ros/humble/setup.bash && python3 /offboard_control/pursuit_gazebo_demo.py drone0"
  ```
- **Terminal 9 (ArUco Perception):**
  ```bash
  docker exec -it ros2_bridge /bin/bash -c "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && source /opt/ros/humble/setup.bash && python3 /offboard_control/aruco_evader_perception.py drone1"
  ```
- **Terminal 10 (Pursuer Controller + Dashboard):**
  ```bash
  docker exec -it ros2_bridge /bin/bash -c "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && source /opt/ros/humble/setup.bash && python3 /offboard_control/pursuit_gazebo_demo.py drone1 --dashboard"
  ```

---

## 📊 Benchmark Results

| Evaluation Profile | Interception Rate | Mean Closest Approach | Status |
| :--- | :---: | :---: | :---: |
| **Non-Maneuvering Evader** | **95.0%** | **$0.47\,\text{m}$** | PASSED (Target: $\ge 95\%$) |
| **Maneuvering Evader** | **42.0%** | **$1.08\,\text{m}$** | PASSED (Min: $\ge 40\%$) |
| **Behavior-Switching Evader** | **97.0%** | **$0.46\,\text{m}$** | PASSED (Target: $\ge 55\%$) |

- **Technical Report:** See [`technical_report.md`](technical_report.md) for full architecture, mathematical formulation, and ablation analysis.

---

## 📜 License
This project is released under the [MIT License](LICENSE).
