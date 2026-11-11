VROSA: A Vision-Language Agentic Control Framework for Autonomous Mobile Robots
Natural-Language Control of Mobile Robots Through Constrained Vision-Language Agents
![ROS 2 Humble](https://img.shields.io/badge/ROS_2-Humble-22314E?logo=ros&logoColor=white) ![NVIDIA Isaac Sim](https://img.shields.io/badge/NVIDIA-Isaac_Sim-76B900?logo=nvidia&logoColor=white) ![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white) ![MIT License](https://img.shields.io/badge/License-MIT-A31F34)
Anonymous peer-review artifact; no identifying website or paper link.
---
Overview
VROSA is a ROS 2-based framework for natural-language control of mobile robots. Building on the ROSA tool-registry paradigm, it connects an LLM planner to a constrained set of perception and robot-motion tools, an interchangeable vision-language model (VLM) backend, and camera-first perception using YOLO, GroundingDINO, and Depth Anything V2.
Navigation follows a plan–move–recheck cycle. The agent selects predefined tools rather than writing low-level velocity commands directly. Tool implementations issue ROS 2 motion commands under sensor-monitored motion constraints. This improves modularity, traceability, and interruptibility, but does not constitute a formal safety guarantee.
```text
Natural-language instruction / dashboard / REST API
                         |
                         v
             LLM planner + tool registry
                         |
               select a constrained tool
                         |
             +-----------+-----------+
             |                       |
             v                       v
   VLM backend manager       ROS 2 perception and state
   (API/local/remote)         (RGB, YOLO, GroundingDINO,
                              DA-V2, optional LiDAR,
                              camera information, odometry)
             |                       |
             +-----------+-----------+
                         |
                         v
             Target grounding and distance
                         |
                         v
               Plan–move–recheck control
                         |
                         v
          Sensor-monitored motion safety layer
                         |
                         v
             ROS 2 motion primitives (/cmd_vel)
```
VROSA was evaluated with Nova Carter in NVIDIA Isaac Sim (LiDAR-equipped) and a physical AgileX Scout 2.0 (without LiDAR, using an Intel RealSense D435i and Jetson AGX Orin 64 GB). The evaluation demonstrates reuse of the agent and tool interfaces across the two configurations. It does not demonstrate plug-and-play transfer: sensor integration, robot motion configuration, deployment, and safety validation are platform-specific.
Key Features
Feature	Description
Reusable ROS 2 agent/tool interfaces	Shared higher-level tool architecture across the two evaluated robot configurations, with platform-specific integration.
Interchangeable VLM backends	API, local, and remote inference modes exposed through a unified interface; supported backend changes can be requested during runtime.
Constrained tool registry	The language agent invokes predefined robot tools rather than generating arbitrary motor commands.
Camera-first target perception	YOLO11n, GroundingDINO, and metric Depth Anything V2 provide detection, text grounding, and object-distance estimates.
Optional LiDAR support	The simulated platform can use LiDAR for obstacle monitoring. Camera-to-LiDAR detector-level object ranging was not validated in the revised controller.
Sensor-monitored motion control	Configured slowdown, stop triggers, freshness checks, and a software stop interface; coverage and response limitations still apply.
Target rechecking	Stopped visual rechecks, target-instance confirmation, and bounded navigation attempts.
Live dashboard and REST API	Camera display, telemetry, natural-language commands, and backend/perception settings.
System Architecture
VROSA has four interacting layers:
Operator interface: Flask dashboard and REST API, including a software stop request.
Reasoning and tools: ReAct-style agent selecting from predefined robot tools.
Perception and visual reasoning: VLM backend manager, YOLO11n, GroundingDINO, metric DA-V2, optional LiDAR, camera/depth inputs, and odometry.
ROS 2 execution and monitoring: Robot-specific topic profiles, motion primitives, sensor freshness checks, and obstacle-triggered motion constraints.
The separation supports reuse of high-level agent logic, but every new robot configuration still needs ROS 2 integration, sensing and frame verification, motion-limit configuration, and physical safety validation.
Installation
Prerequisites
Ubuntu 22.04 LTS and ROS 2 Humble, with the required ROS message packages and working sensor drivers.
Python 3.10 and a compatible PyTorch/CUDA environment for the selected inference backend.
NVIDIA Isaac Sim and a configured Nova Carter scene for simulation (the version must match the actual evaluated environment).
The appropriate camera, LiDAR (if used), and mobile-base ROS 2 integrations.
Obtain the code
During double-anonymous review, use the anonymous repository browser's Download ZIP option, extract the archive, and work in its extracted directory. Do not use the original author's identifiable GitHub URL for review-stage setup instructions.
```bash
# Run these commands from the extracted repository directory.
conda create -n vrosa python=3.10
conda activate vrosa
source /opt/ros/humble/setup.bash

# Install project dependencies in an environment compatible with your
# ROS 2, CUDA/PyTorch, and Isaac Sim versions.
# The supplied requirements.txt is a captured environment with ROS/Isaac
# packages; it has not been established as a portable, clean-install recipe.
# Review the required package versions before installing.
# pip install -r requirements.txt
```
Dependency note: The repository's `requirements.txt` currently lists Isaac Sim 4.2 packages, while the earlier README named Isaac Sim 4.5. These should be reconciled against the actual experiments before claiming a one-command reproducible setup. Never publish API secrets or private access tokens in configuration files, commands, screenshots, or logs.
Configure model backends
The launch configuration is defined by CLI arguments and environment variables in `config.py`. The planner uses the configured API-compatible language-model endpoint; the VLM backend can be API, local, or remote. Supply credentials through environment variables or another secure mechanism, not committed plaintext.
```bash
# Example variable names; supply your own authorized endpoint/model/credential.
export VROSA_API_KEY='<your-api-key>'
export VROSA_API_BASE='<your-compatible-api-base-url>'
export VROSA_API_MODEL='<your-model-id>'
```
Running in Simulation
Start Isaac Sim with a configured Nova Carter scene and its ROS 2 bridge, then from the extracted project directory run:
```bash
source /opt/ros/humble/setup.bash
python3 vrosa_live.py --robot nova_carter_sim --vlm api
```
Open the dashboard at `http://localhost:5000/view`. Example requests include “What do you see?”, “Look around”, and “Find the forklift”. Robot motion should be attempted only after confirming the expected camera, odometry, obstacle-sensing topics, and safety configuration. The exact scene and ROS 2 namespace must match the selected robot profile.
Important: The `--robot` option is required by `config.py`. Running only `python3 vrosa_live.py` without it will not start the launcher.
Running on Physical Hardware (AgileX Scout 2.0)
The physical platform uses a Jetson AGX Orin 64 GB and Intel RealSense D435i. Camera-only means no LiDAR in this evaluation; it does not mean that obstacle-safety decisions relied exclusively on inferred monocular depth. The hardware safety interlock uses physical aligned RealSense depth. Metric DA-V2 depth supports target perception, but model-predicted depth alone does not authorize physical translation in the guarded hardware configuration.
Bring-up requires validated base-driver, camera, odometry, and safety-sensor connections. The original experiment used a particular CAN interface and workspace layout; do not copy physical-interface names or motor-control settings to another robot without verifying them.
Hardware diagnostic launch (no motion)
Before connecting the motion controller, check the manufacturer's remote-control state, physical emergency stop, available free space, and the correct CAN interface. The commands below retain the original example's `can2` interface for documentation; verify the actual interface (it might be `can0` or another name). Keep the robot supervised and remain in diagnostic mode.
Terminal 1 — CAN interface (only after interface and driver verification):
```bash
ip -details link show can2  # confirm this interface exists before proceeding
sudo modprobe gs_usb        # only if this is the correct CAN adapter driver
sudo ip link set can2 up type can bitrate 500000 restart-ms 100
sudo ip link set can2 txqueuelen 1000
```
Terminal 2 — Scout base driver:
```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ros2 launch scout_base scout_base.launch.py port_name:=can2
```
Terminal 3 — RealSense camera with aligned depth:
```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true
```
Terminal 4 — VROSA, diagnostic mode (no motion):
```bash
# In the extracted repository directory, activate your prepared Python environment.
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
python3 vrosa_live.py --robot scout2 --hardware-mode diagnostic --vlm api
```
Verify active image, aligned-depth, `/odom`, and `/cmd_vel` topics and their publishers/subscribers before attempting any supervised motion tests. The original example's remote switch position must be verified against the actual robot's manual; a software command cannot establish manual-override authority by itself.
The default `diagnostic` hardware mode rejects nonzero commanded motion. Do not change to `guarded` mode until the manufacturer's safety procedures, manual control, base driver, synchronized sensor streams, odometry, emergency-stop hardware, stop thresholds, and supervised test area have been verified for that robot.
The software `/stop` interface interrupts the agent and requests zero ROS velocity. It is not a substitute for an independently functioning physical emergency stop or supervised manual override. The controller's sensor region can miss low obstacles, and a measured software stop is not proof of universal collision avoidance.
Porting to a New ROS 2 Robot
Robot-specific topic names and motion/safety settings are defined in the `ROBOT_PROFILES` dictionary in `config.py`. `ros_node.py` reads the selected profile through `ROBOT["topics"]`; topic definitions are not edited as independent `TOPIC_*` constants at the top of `ros_node.py`.
The Scout example profile includes:
```python
"topics": {
    "image": "/camera/camera/color/image_raw",
    "camera_info": "/camera/camera/color/camera_info",
    "lidar": None,
    "odom": "/odom",
    "cmd_vel": "/cmd_vel",
}
```
These are examples, not guaranteed topic names for other robots. New integrations must also verify coordinate frames, QoS, depth alignment and calibration, physical collision geometry, drive kinematics, ROS 2 launch dependencies, sensing coverage, braking limits, and the configured safety trigger. The evaluated configuration uses a 1.10 m stop trigger and 0.75 m minimum-clearance reference; neither is a universal safe threshold for another robot.
Robot Tools
The bounded tool registry is defined in `tools.py` and assembled in `agent.py`.
Tool	Purpose
`get_robot_position`	Return odometry state.
`move_forward`	Execute bounded translation through motion controls.
`rotate_robot`	Execute a bounded rotation.
`stop_robot`	Request stopping through the software control path.
`describe_current_view`	Request a VLM scene description.
`find_object`	Search the camera view for a named object.
`navigate_to_object`	Perform iterative semantic navigation and visual rechecking.
`check_path_ahead`	Check the supported forward-sensing region for obstacles.
`save_snapshot`	Save a camera frame.
`look_around`	Perform a scene scan and store target-relevant scene information.
Adding Domain-Specific Tools
New tools can be implemented as supported LangChain tools and registered in the list passed to `ROSA` in `agent.py`. They must preserve the same safety, sensor-freshness, and motion-authority constraints as the existing tools. Merely registering a new tool does not make its physical behavior safe or validated.
REST API
Endpoint	Method	Purpose
`/` or `/view`	GET	Dashboard.
`/stream`	GET	MJPEG camera stream.
`/frame.jpg`	GET	Current camera frame.
`/command`	POST	Natural-language command.
`/stop`	GET/POST	Software stop request (not a hardware emergency stop).
`/status`	GET	Robot telemetry.
`/vlm/switch`	POST	Select a VLM backend.
`/vlm/model/switch`	POST	Select an API model.
`/vlm/status`	GET	Model/backend status.
`/yolo/switch`	POST	Change supported YOLO configuration.
`/perc/config`	GET/POST	Configure available perception sources.
`/detections`	GET	Current detection summary.
`/snapshot`	GET	Save a camera frame.
Disabling all safety-relevant sensing is not a supported mode for autonomous translational motion.
Experimental Evidence and Limitations
The accompanying manuscript reports separate simulation and physical evaluations. The historical sensing-mode comparison comprised 50 attempts per sensing condition. A newer comparison involved 18 controller executions across six Isaac Sim target configurations; complete geometric measurements for all three methods were available in five configurations. At the originally specified 2.20 m geometric-proximity criterion, VROSA passed 5/5 complete-case configurations and each comparison controller passed 1/5. This small comparison did not establish statistical significance (exploratory exact paired p = 0.125).
The 24 outdoor navigation trials were reassessed conservatively as 20 full, three partial, and one failure (83.3% full completion). This is operator-assessed, instruction-level completion, not independent physical ground-truth arrival measurement. The reported confidence interval and the original assessment are explained in the manuscript.
Independent simulator-ground-truth depth experiments evaluated raw and exploratory affine-calibrated metric DA-V2 estimates. The calibrated accuracy figures are not deployed-controller accuracy; the evaluated controller used uncalibrated metric model output. Safety experiments measured selected stop-trigger configurations, sudden and moving obstacles, and supervised Scout interlocks. They do not establish complete obstacle coverage, full-trajectory clearance, or collision-free operation under all dynamic conditions. A low-obstacle region-of-interest blind spot was observed on hardware.
The original trial allocations, synchronization, and some matching records do not support broad generalization claims. Further work includes more physical-robot repetitions, unseen categories, systematic multilingual evaluation, and synchronized vision-to-actuation latency measurement.
Supporting Materials
Repository directories include implementation source code, robot-tool and model-prompt construction, tests, depth-validation utilities, and available supporting experiment records. They are intended to be interpreted alongside the exact procedures and limitations reported in the manuscript. Do not infer that every result is independently reproducible without the corresponding robot/simulator environment and dependencies.
Citation
An anonymized manuscript accompanies this repository for review. A full bibliographic citation will be provided after publication. No author-identifying reference is included here during double-anonymous review.
License
MIT License — see LICENSE.
---
Built with ROS 2 Humble · NVIDIA Isaac Sim · YOLO · GroundingDINO · Depth Anything V2
