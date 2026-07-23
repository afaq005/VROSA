# VROSA: Vision-Language Agentic Control for Autonomous Mobile Robots

<div align="center">

**Natural Language Control of Mobile Robots via Vision-Language Agents**

[![ROS2](https://img.shields.io/badge/ROS2-Humble-blue?logo=ros)](https://docs.ros.org/en/humble/)
[![Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-4.5.0-green?logo=nvidia)](https://developer.nvidia.com/isaac-sim)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow?logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-red)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-IEEE%20T--ASE-orange)](https://arxiv.org)

[**Paper**](#) · [**Website**](#) · [**Demo Video**](#)

</div>

---

## Overview

**VROSA** is a robot-agnostic, vision-language agent framework for natural language control of mobile robots. Built on ROS 2 and extending the [ROSA](https://github.com/nasa-jpl/rosa) tool-registry paradigm, VROSA couples an LLM planner with a constrained tool registry, a hot-swappable vision-language model (VLM) backend, and a camera-first perception pipeline — so a robot can be commanded in plain English without retraining or reprogramming.

Every navigation decision runs through a **plan–execute–recheck loop**: the agent never issues raw velocity commands directly. It selects from a fixed set of tools (`navigate_to_object`, `move_forward`, `rotate_robot`, `describe_current_view`, `look_around`, etc.), and geometric sensors — not the VLM — handle distance estimation and safety stops. This keeps the system auditable, debuggable, and safe by construction.

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

Validated in **NVIDIA Isaac Sim** (Nova Carter, LiDAR-equipped) and on a **physical AgileX Scout 2.0** (camera-only, Jetson AGX Orin 64GB), demonstrating cross-platform transfer with no change to agent, tool, perception, or motion code — only ROS 2 topic constants.

## Key Features

| Feature | Description |
|---|---|
| 🤖 **Robot-Agnostic** | ROS 2-compatible platforms — transfer requires only topic-constant changes |
| 🔌 **Hot-Swappable VLM Backend** | Cloud API, local fine-tuned model, or remote GPU server — switch at runtime, no restart |
| 🧠 **Constrained Tool Registry** | LLM never issues raw motor commands — only validated, auditable tools |
| 📐 **LiDAR-Optional Distance Estimation** | Sensor-fused priority: LiDAR → depth-patch median → depth percentile fallback |
| 🎯 **Sector-Filtered Disambiguation** | Correctly grounds "the bench" or "the green e-bike" among multiple same-class objects |
| 🛡️ **Graduated Safety Layer** | Slowdown zone + hard stop, independent of which sensing channel is active |
| 🖥️ **Live Dashboard** | Flask + MJPEG stream, telemetry, perception-source toggles, natural language input |

## System Architecture

VROSA is organized as four layers:

1. **Natural language interface** — Flask dashboard + REST API for operator access and emergency stop
2. **LLM agent and tool registry** — ReAct-style loop selecting from a bounded action space
3. **Perception and VLM layer** — YOLO + GroundingDINO detection, Depth Anything V2 metric depth, optional LiDAR fusion, hot-swappable VLM backend manager
4. **ROS 2 execution layer** — sensor integration, actuator control, and the graduated-speed safety layer

Robot-specific details are isolated in the ROS 2 layer; model-specific details are isolated in the VLM backend manager — new perception modules or robot platforms can be added without redesigning the control pipeline.

## Installation

### Prerequisites

- Ubuntu 22.04 LTS
- ROS 2 Humble + CycloneDDS
- NVIDIA Isaac Sim 4.5.0 (for simulation)
- CUDA 12.x
- Python 3.10+

### Setup

```bash
git clone https://github.com/afaq005/VROSA.git
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

Hardware trials run in **camera-only mode** (no LiDAR) on a Jetson AGX Orin 64GB, with all perception and agent computation running onboard. Bring-up requires four terminals.

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
cd ~/afaq/vrosa_hardware_testing
source ~/vrosa-env/bin/activate
source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=0 && export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
python3 vrosa_live.py
```

Before running, install dependencies from `requirements.txt` and configure your VLM/LLM backends in `config.py` as described above.

> ⚠️ **Safety note:** always set SWB to TOP on the RC remote before bringing up any node — this ensures the operator retains manual override authority before the agent can issue any motion commands. Emergency stop (dashboard or REST `/stop`) bypasses the agent directly and is independent of the RC remote.

### Porting to a New ROS 2 Robot

Transfer to a new platform (as demonstrated between the simulated Nova Carter and the physical Scout 2.0 in this work) requires only changing ROS 2 topic constants — no agent, tool, perception, or motion code changes:

```yaml
# robots/my_robot.yaml
robot:
  name: "My Robot"
  topics:
    camera:    /camera/color/image_raw
    lidar:     /scan/points          # optional — omit if camera-only
    odometry:  /odom
    cmd_vel:   /cmd_vel
    cam_info:  /camera/camera_info
  safe_distance_m: 0.75
  max_speed_ms:    0.4
```

## Adding Domain-Specific Tools

```python
from langchain.tools import tool

@tool
def check_corridor_clear() -> str:
    """Check if a corridor is clear for robot passage."""
    # Implementation using existing perception + safety infrastructure
    ...

agent = VROSAAgent(
    ros_version=2,
    llm=llm,
    tools=[
        *base_tools,
        check_corridor_clear,   # ← just add it
    ]
)
```

The agent decides when to invoke each tool based on the natural language command — no retraining required.

## Citation

If you use VROSA in your research, please cite:

```bibtex
@article{vrosa2026,
  title   = {VROSA: Vision-Language Agentic Control for Autonomous Mobile Robots},
  author  = {Ahmed, Afaq and Eesaar, Hassan and Lee, Deok Jin},
  journal = {IEEE Transactions on Automation Science and Engineering},
  year    = {2026}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
<sub>Built with NVIDIA Isaac Sim · ROS 2 Humble · YOLO · GroundingDINO · Depth Anything V2</sub>
</div>
