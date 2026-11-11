# VROSA: Vision-Language Agentic Control for Autonomous Mobile Robots

<div align="center">

**Natural Language Control of Mobile Robots via Vision-Language Agents**

[![ROS2](https://img.shields.io/badge/ROS2-Humble-blue?logo=ros)](https://docs.ros.org/en/humble/)
[![Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-4.5.0-green?logo=nvidia)](https://developer.nvidia.com/isaac-sim)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow?logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-red)](LICENSE)

[**Paper**](#) · [**Website**](https://afaq005.github.io/VROSA/) · [**Demo Video**](#)

</div>

---

## Overview

**VROSA** is a vision-language agent framework for natural language control of mobile robots. Built on ROS 2 and extending the [ROSA](https://github.com/nasa-jpl/rosa) tool-registry paradigm, VROSA couples an LLM planner with a constrained tool registry, a hot-swappable vision-language model (VLM) backend, and a camera-first perception pipeline — so a robot can be commanded in plain English without retraining or reprogramming.

Every navigation decision runs through a plan–move–recheck loop: the agent selects from a constrained set of predefined tools (navigate_to_object, move_forward, rotate_robot, describe_current_view, look_around, etc.) rather than generating raw motor commands. Distance estimation uses camera-based depth and optional LiDAR measurements, while a separate sensor-monitored controller supervises motion. This separation improves modularity, traceability, and safety monitoring, but does not guarantee collision-free navigation.

```
Natural Language Command
         │
         ▼
   VROSA Agent (ReAct: Plan → Act → Observe)
         │
         ▼
 ┌───────────────────────────────┐
 │   VLM Backend (hot-swappable) │
 │   ├── Cloud API (Claude, etc.)│
 │   ├── Local LoRA (LLaVA-V5)   │
 │   └── Remote GPU server       │
 └───────────────────────────────┘
         │
         ▼
 Perception: YOLO + GroundingDINO
         + Depth Anything V2 (+ optional LiDAR)
         │
         ▼
 Sensor-Fused Distance + Safety Layer
         │
         ▼
 ROS 2 Motion Primitives (/cmd_vel)
```

Evaluated in NVIDIA Isaac Sim (Nova Carter, LiDAR-equipped) and on a physical AgileX Scout 2.0 (camera-only, Jetson AGX Orin 64 GB), demonstrating reuse of the agent and tool interfaces across the two configurations. Deployment required platform-specific sensor integration, motion configuration, calibration, and safety validation. Generalized plug-and-play robot portability has not yet been established.

## Key Features

| Feature | Description |
|---|---|
| 🤖 **Reusable ROS 2 Interfaces** | Agent and tool interfaces reused across the evaluated Nova Carter and Scout 2.0 configurations, with platform-specific integration and validation |
| 🔌 **Hot-Swappable VLM Backend** | Cloud API, local fine-tuned model, or remote GPU server — switch at runtime, no restart |
| 🧠 **Constrained Tool Registry** | LLM never issues raw motor commands — only validated, auditable tools |
| 📐 **LiDAR-Optional Distance Estimation** | Camera-based metric depth with validity checks and adaptive percentiles; optional LiDAR for forward obstacle monitoring |
| 🎯 **Semantic Target Disambiguation** | VLM-assisted identification of target objects using visual context and semantic attributes, with target rechecking during navigation |
| 🛡️ **Sensor-Monitored Safety Layer** | Configurable speed limits, obstacle-triggered stopping, and emergency-stop handling under tested sensing and controller conditions |
| 🖥️ **Live Dashboard** | Flask + MJPEG stream, telemetry, perception-source toggles, natural language input |

## System Architecture

VROSA is organized as four layers:

1. **Natural language interface** — Flask dashboard + REST API for operator access and emergency stop
2. **LLM agent and tool registry** — ReAct-style loop selecting from a bounded action space
3. **Perception and VLM layer** — YOLO + GroundingDINO detection, Depth Anything V2 metric depth, optional LiDAR fusion, hot-swappable VLM backend manager
4. **ROS 2 execution layer** — sensor integration, actuator control, and the graduated-speed safety layer

Robot-specific interfaces are separated from the language-agent and tool layers, while model-specific configurations are managed through the VLM backend manager. This modular organization supports adaptation to different ROS 2 platforms, although new deployments require sensor integration, actuation configuration, calibration, and safety validation.

## Installation

### Prerequisites

- Ubuntu 22.04 LTS
- ROS 2 Humble + CycloneDDS
- NVIDIA Isaac Sim 4.5.0 (for simulation)
- CUDA 12.x
- Python 3.10+

### Setup

```bash
# Download the repository ZIP from:
# https://anonymous.4open.science/r/VROSA-C7BF/

# Extract the archive and enter its root directory.
cd VROSA

# Create environment
conda create -n vrosa python=3.10
conda activate vrosa

# Install dependencies
pip install -r requirements.txt

# Source ROS 2
source /opt/ros/humble/setup.bash
```

### Configure VLM and LLM Backends

Open `config.py` and set up your desired VLM/LLM backends (cloud API keys, local model weights path, or remote inference server URL) before running the agent. Backend selection and model can also be changed at runtime through the dashboard or REST API (`/vlm/switch`, `/vlm/model/switch`), without restarting the system.

## Running in Simulation

```bash
# 1. Start Isaac Sim with the Nova Carter robot
# 2. Launch VROSA
python3 vrosa_live.py

# 3. Open the dashboard
# → http://localhost:5000

# 4. Send natural language commands, e.g.:
#   "navigate to the forklift"
#   "what do you see?"
#   "check path ahead"
#   "look around"
```

## Running on Physical Hardware (AgileX Scout 2.0)

Hardware trials use a LiDAR-free configuration on an AgileX Scout 2.0 equipped with a Jetson AGX Orin 64 GB and an Intel RealSense camera. ROS 2 control, local perception, and motion-safety monitoring execute onboard, while the selected cloud language-model backend uses external API inference. Bring-up requires separate terminals for the robot interface, camera, and VROSA application.

```bash
# 0. Set SWB to TOP on the RC remote FIRST (before anything else)

# 1. Terminal 1 — CAN bringup
sudo modprobe gs_usb 2>/dev/null || sudo insmod ~/gs_usb_module/gs_usb.ko
sudo ip link set can2 up type can bitrate 500000 restart-ms 100
sudo ip link set can2 txqueuelen 1000

# 2. Terminal 2 — Scout base
source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=0 && export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ros2 launch scout_base scout_base.launch.py port_name:=can2

# 3. Terminal 3 — RealSense
source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=0 && export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ros2 launch realsense2_camera rs_launch.py

# 4. Terminal 4 — VROSA
cd /path/to/vrosa_hardware_testing
source ~/vrosa-env/bin/activate
source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=0 && export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
python3 vrosa_live.py
```

Before running, install dependencies from `requirements.txt` and configure your VLM/LLM backends in `config.py` as described above.

> ⚠️ **Safety note:** always set SWB to TOP on the RC remote before bringing up any node — this ensures the operator retains manual override authority before the agent can issue any motion commands. Emergency stop (dashboard or REST `/stop`) bypasses the agent directly and is independent of the RC remote.

### Porting to a New ROS 2 Robot

VROSA supports reuse of its language-agent and tool interfaces across ROS 2 robot configurations. The evaluated Nova Carter and Scout 2.0 deployments required platform-specific integration and safety verification. Adapting the system to another robot requires configuring ROS 2 topic mappings, sensor inputs, coordinate frames, actuation parameters, depth estimation, and motion-safety thresholds. The example below illustrates topic mappings rather than a complete robot-porting procedure.

```python
# ros_node.py
TOPIC_IMAGE     = "/camera/camera/color/image_raw"
TOPIC_CAMINFO   = "/camera/camera/color/camera_info"
TOPIC_DEPTH     = "/camera/camera/depth/image_rect_raw"
TOPIC_LIDAR     = "/front_3d_lidar/lidar_points"   # omit / leave unsubscribed if camera-only
TOPIC_CMDVEL    = "/cmd_vel"
TOPIC_ODOM      = "/odom"
```

Configure the appropriate ROS 2 domain, robot profile, sensor topics, and platform-specific motion parameters. Safety thresholds must be selected and validated for the robot's sensing coverage, stopping behavior, and operating environment. Changing a single distance threshold does not establish safe operation on a new platform.

## Robot Tools

The agent selects from a fixed tool registry defined in `tools.py` — it never issues raw velocity commands directly:

| Tool | Purpose |
|---|---|
| `get_robot_position` | Return odometry state |
| `move_forward` | Move forward/backward by distance |
| `rotate_robot` | Rotate by angle |
| `stop_robot` | Immediately stop |
| `describe_current_view` | VLM scene description |
| `find_object` | Search camera view for a named object |
| `navigate_to_object` | Closed-loop plan–execute–recheck navigation |
| `check_path_ahead` | Evaluate forward path for obstacles |
| `save_snapshot` | Save current camera frame |
| `look_around` | 360° scene scan |

### Adding Domain-Specific Tools

```python
from langchain.tools import tool

@tool
def check_corridor_clear() -> str:
    """Check if a corridor is clear for robot passage."""
    # Implementation using existing perception + safety infrastructure
    ...

# In agent.py, add to the tools list:
agent = ROSA(
    ros_version=2,
    llm=llm,
    prompts=prompts,
    tools=[
        get_robot_position, move_forward, rotate_robot, stop_robot,
        describe_current_view, look_around, find_object, navigate_to_object,
        check_path_ahead, save_snapshot,
        check_corridor_clear,   # ← just add it
    ],
)
```

The agent decides when to invoke each tool based on the natural language command — no retraining required.

## REST API

| Endpoint | Method | Purpose |
|---|---|---|
| `/` or `/view` | GET | Web dashboard |
| `/stream` | GET | MJPEG camera stream |
| `/command` | POST | Natural language input |
| `/stop` | GET/POST | Emergency stop (bypasses the agent directly) |
| `/status` | GET | Robot telemetry |
| `/vlm/switch` | POST | Switch VLM backend (api / local / remote) |
| `/vlm/model/switch` | POST | Switch API model |
| `/vlm/status` | GET | Current VLM backend and model status |
| `/yolo/switch` | POST | Switch YOLO model variant |
| `/perc/config` | GET/POST | Toggle perception sources (LiDAR/depth/YOLO/GroundingDINO) |
| `/detections` | GET | Current detection list |
| `/snapshot` | GET | Save camera frame |


## Experimental Scope and Limitations

VROSA was evaluated in NVIDIA Isaac Sim and on a physical
AgileX Scout 2.0. The experiments include navigation
comparisons, depth validation, controlled safety tests,
and outdoor natural-language navigation.

The reported results are specific to the evaluated
configurations. The system does not establish universal
collision-free operation, fully generalized robot
portability, or guaranteed performance in unseen
environments. Independent simulator-ground-truth depth
validation is reported separately from deployed
uncalibrated depth performance.


## Citation


VROSA: A Vision-Language Agentic Control Framework for Autonomous Mobile Robots.

Manuscript under peer review.

Full bibliographic information will be provided following
publication.

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
<sub>Built with NVIDIA Isaac Sim · ROS 2 Humble · YOLO · GroundingDINO · Depth Anything V2</sub>
</div>
