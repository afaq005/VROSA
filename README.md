# V-ROSA: Visual Robot Operating System Agent

<div align="center">

**Integrating Vision-Language Models with Real-Time Robot Navigation**

[![ROS2](https://img.shields.io/badge/ROS2-Humble-blue?logo=ros)](https://docs.ros.org/en/humble/)
[![Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-4.5.0-green?logo=nvidia)](https://developer.nvidia.com/isaac-sim)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow?logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-red)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-IEEE%20T--RO-orange)](https://arxiv.org)

[**Paper**](#) · [**Website**](#) · [**Demo Video**](#) · [**HuggingFace Model**](#)

</div>

---

## Overview

**V-ROSA** is a robot-agnostic, semantically-driven navigation agent that combines fine-tuned Vision-Language Models (VLM) with real-time LiDAR fusion. Unlike end-to-end VLA systems, V-ROSA uses a **ReAct (Reason + Act) loop** — every navigation decision is transparent, debuggable, and extensible without retraining.

```
Natural Language Command
         ↓
   ROSA Agent (ReAct)
         ↓
  ┌──────────────────────────────┐
  │  VLM Backend (hot-swappable) │
  │  ├── Cloud API (Claude/GPT)  │
  │  ├── Local LoRA (LLaVA-1.5) │
  │  └── Remote GPU  (A100)      │
  └──────────────────────────────┘
         ↓
  LiDAR-VLM Directional Fusion
         ↓
  Robot Action (ROS 2 /cmd_vel)
```

## Key Features

| Feature | Description |
|---|---|
| 🤖 **Robot-Agnostic** | Any ROS 2 compatible robot — swap via `robots/` config |
| 🔌 **Plug-and-Replace VLM** | Cloud API, local LoRA, or remote GPU — hot-swap at runtime |
| 🧠 **Extensible Tool Registry** | Add domain-specific tools without rewriting the agent |
| 📐 **Metric Distance Output** | "approximately 2.8 meters" — no depth sensor needed at inference |
| 🔀 **LiDAR-VLM Fusion** | Directional point cloud filtering for object-specific distance |
| 🚀 **One-Shot Navigation** | Full-distance moves vs. slow step-wise approaches |
| 🖥️ **Live Dashboard** | Flask + MJPEG stream, real-time telemetry, natural language input |

## Novelty

V-ROSA is the **first open VLN system on an extensible ROSA agent harness**, with six primary contributions:

1. **Robot-agnostic architecture** — ROS 2 topic abstraction via `TopicManager`
2. **Plug-and-replace `VLMBackend` interface** — cloud/local/remote, zero code changes
3. **First VLN system on ROSA** — extends JPL ROSA from text-only to visual navigation
4. **Depth Anything V2 as teacher** — auto-labels any RGB image with metric depth in natural language
5. **Directional LiDAR-VLM fusion** — no semantic segmentation or depth estimation at inference
6. **ReAct semantic navigation** — transparent, debuggable vs. end-to-end VLA black boxes

## Comparison to Related Work

| System | Fine-tuned VLM | Metric Distance | Real-time Nav | Extensible Tools | Open Source |
|--------|:-:|:-:|:-:|:-:|:-:|
| **V-ROSA (ours)** | ✅ | ✅ metres | ✅ LiDAR fused | ✅ | ✅ |
| SayCan (Google) | ❌ | ❌ | Tabletop only | ❌ | Partial |
| OpenVLA | ✅ | ❌ | Manipulation | ❌ | ✅ |
| RT-2 (Google) | ❌ | ❌ | Tabletop only | ❌ | ❌ |
| NavGPT | ❌ | Categorical | ✅ | ❌ | ❌ |
| LM-Nav | ❌ | ❌ | Outdoor only | ❌ | ✅ |

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        V-ROSA System                            │
│                                                                 │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────────┐  │
│  │  Hardware   │    │  ROS 2       │    │  Intelligence     │  │
│  │  Layer      │───▶│  Middleware  │───▶│  Layer            │  │
│  │             │    │              │    │                   │  │
│  │ Nova Carter │    │ CycloneDDS   │    │ ROSA Agent        │  │
│  │ Isaac Sim   │    │ TopicManager │    │ ├── VLMManager    │  │
│  │ RTX 3080    │    │ 5 topics     │    │ │   ├── API       │  │
│  │ A100 (srv)  │    │              │    │ │   ├── Local     │  │
│  └─────────────┘    └──────────────┘    │ │   └── Remote    │  │
│                                         │ └── Tool Registry │  │
│                                         └───────────────────┘  │
│                                                  │              │
│                                         ┌────────▼───────────┐ │
│                                         │  Flask Dashboard   │ │
│                                         │  MJPEG @ 15 FPS    │ │
│                                         │  NL Command Input  │ │
│                                         └────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Ubuntu 22.04 LTS
- ROS 2 Humble + CycloneDDS
- NVIDIA Isaac Sim 4.5.0.2 (for simulation)
- CUDA 12.8
- Python 3.10+

### Setup

```bash
git clone https://github.com/afaq005/vrosa.git
cd vrosa

# Create environment
conda create -n vrosa python=3.10
conda activate vrosa

# Install dependencies
pip install -r requirements.txt

# Source ROS 2
source /opt/ros/humble/setup.bash
```

### VLM Backends

**Option 1 — Cloud API (no GPU needed, fastest start):**
```bash
python vrosa/vrosa_live.py --vlm api \
  --api-key YOUR_KEY \
  --api-base https://api.anthropic.com/v1 \
  --api-model claude-sonnet-4-6
```

**Option 2 — Fine-tuned Local Model:**
```bash
# Download weights from HuggingFace
huggingface-cli download YOUR_HF_REPO vrosa_vlm_v4 --local-dir ./weights/vrosa_vlm_v4

python vrosa/vrosa_live.py --vlm local \
  --local-model ./weights/vrosa_vlm_v4
```

**Option 3 — Remote  Inference Server:**
```bash
# On the server:
python inference/vrosa_inference_server.py

# On your robot machine:
python vrosa/vrosa_live.py --vlm remote \
  --remote-url http://SERVER_IP:7860
```

### Hot-swap VLM at Runtime

```bash
# From CLI
curl -X POST http://localhost:5000/vlm/switch \
     -H 'Content-Type: application/json' \
     -d '{"mode": "local"}'

# Or use the dashboard dropdown at http://localhost:5000
```

## Usage

### Quick Start (Simulation)

```bash
# 1. Start Isaac Sim with Nova Carter
# 2. Launch V-ROSA
python vrosa/vrosa_live.py --vlm api

# 3. Open dashboard
# → http://localhost:5000

# 4. Send natural language commands:
#   "navigate to the forklift"
#   "what do you see?"
#   "check path ahead"
#   "move forward 2 meters"
```

### Adding Your Own Robot

```yaml
# robots/my_robot.yaml
robot:
  name: "My Robot"
  topics:
    camera:    /camera/image_raw
    lidar:     /scan/points
    odometry:  /odom
    cmd_vel:   /cmd_vel
    cam_info:  /camera/camera_info
  safe_distance_m: 1.0
  max_speed_ms:    0.4
```

### Adding Domain-Specific Tools

```python
# Example: Medical robot tool
from langchain.tools import tool

@tool
def check_corridor_clear() -> str:
    """Check if hospital corridor is clear for robot passage."""
    # Your implementation using existing VLM + LiDAR infrastructure
    ...

# Add to agent
agent = ROSA(
    ros_version=2,
    llm=llm,
    tools=[
        *base_tools,
        check_corridor_clear,   # ← just add it
        identify_patient,
        alert_staff,
    ]
)
```

The agent automatically decides when to use each tool based on natural language commands — no retraining required.

## Training Your Own VLM

### Data Collection (Isaac Sim)

```bash
# Auto-drive robot to collect training frames with LiDAR ground truth
python training/data_collection/isaac_collector.py \
  --output ./data/isaac_sim_frames \
  --duration 7200  # 2 hours
```

### Auto-Label with Depth Anything V2

```bash
# Apply metric depth labels to any RGB dataset
python training/depth_labeling/depth_teacher.py \
  --input ./data/raw_images \
  --output ./data/labeled \
  --scale-prior 3.0  # median depth → 3m (warehouse prior)
```

### Fine-tune LLaVA-1.5

```bash
python training/train.py \
  --config training/configs/v4_config.yaml \
  --data ./data/vrosa_v4_48k.jsonl \
  --output ./weights/vrosa_vlm_v4
```

### Training Config

```yaml
# training/configs/v4_config.yaml
base_model: llava-hf/llava-1.5-7b-hf
lora:
  rank: 16
  alpha: 32
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
training:
  precision: bfloat16
  batch_size: 2
  grad_accumulation: 16
  sequence_length: 1024
  lr_schedule: cosine
  eval_batch_size: 1
  save_steps: 200
```

## Training Results

| Version | Train Loss | Eval Loss | Samples | Duration | Quality |
|---------|:---------:|:---------:|:-------:|:--------:|---------|
| V1 | 0.069 | 0.030 | 19,800 | 7.4h | ❌ Overfit, templates |
| V3 | 0.132 | 0.090 | 24,000 | 9.5h | ✅ Natural metric distances |
| V3+ | ~0.12 | ~0.08 | 30,600 | ~10h | ✅ Object naming + distances |
| **V4** | TBD | TBD | **48,600** | ~12h | Real-world generalization |

> The healthy loss gap (train > eval) in V3+ indicates genuine generalization, not memorization. V1's near-zero eval loss indicated overfitting.

## Navigation Performance (V3 Model, Isaac Sim)

| Test | Command | Result | Notes |
|------|---------|--------|-------|
| Object discovery | "Do you see any forklift?" | Found after 2 rotations | Identified at 1.9m right |
| Navigate to object | "Move towards the cart" | Stopped at 0.66m | LiDAR safety working |
| Scene description | "What do you see?" | Blue cart + yellow cart + distances | Metric distances accurate |
| Path check | "Check path ahead" | CLEAR, nearest 2.76m | LiDAR + VLM both reported |
| One-shot nav | Forklift at 1.9m | 1 move of 1.6m | vs. 7 moves previously |
| Obstacle stop | Forward movement | Stopped at 0.97m | LiDAR threshold at 1.0m |

## Repository Structure

```
vrosa/
├── vrosa/
│   ├── vrosa_live.py          # Main agent — run this
│   ├── agent/
│   │   └── tools/             # Extensible tool definitions
│   ├── vlm/
│   │   ├── base.py            # VLMBackend abstract class
│   │   ├── api_backend.py     # Claude / GPT-4V
│   │   ├── local_backend.py   # Fine-tuned LLaVA + LoRA
│   │   └── remote_backend.py  # Flask inference server
│   └── navigation/
│       ├── lidar_fusion.py    # Directional LiDAR filtering
│       └── one_shot_nav.py    # Full-distance navigation
├── training/
│   ├── data_collection/
│   │   └── isaac_collector.py # Isaac Sim auto-driver
│   ├── depth_labeling/
│   │   └── depth_teacher.py   # DA-V2 auto-labeling
│   ├── train.py
│   └── configs/
│       ├── v3_config.yaml
│       └── v4_config.yaml
├── inference/
│   └── vrosa_inference_server.py  # A100 Flask server
├── robots/                    # Robot platform configs
│   ├── nova_carter.yaml
│   └── README.md
├── docs/                      # Project website (GitHub Pages)
├── requirements.txt
└── README.md
```

## Citation

If you use V-ROSA in your research, please cite:

```bibtex
@article{vrosa2026,
  title     = {V-ROSA: A Robot-Agnostic Visual Navigation Agent with
               Extensible Tool Harness and Metric Depth VLM},
  author    = {YOUR NAME},
  journal   = {IEEE Transactions on Robotics},
  year      = {2026},
  doi       = {}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
<sub>Built with NVIDIA Isaac Sim · ROS 2 Humble · LLaVA-1.5 · JPL ROSA · Depth Anything V2</sub>
</div>
