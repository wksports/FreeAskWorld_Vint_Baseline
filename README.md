

# ViNT Baseline - ROS2 Visual Navigation

[![ROS2](https://img.shields.io/badge/ROS2-Humble-blue.svg)](https://docs.ros.org/en/humble/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green.svg)](https://www.python.org/)

ViNT-powered [ViNT](https://general-navigation-models.github.io/vint/) ROS2 navigation on the FreeAskWorld Unity simulator.

---

## 📋 Quick Start

### Prerequisites

- Ubuntu 22.04 LTS (WSL)
- ROS2 Humble Refer to [Link](https://github.com/doraemonaaaa/freeaskworld_closed_loop) to install ROS 2 dependencies. 
- Python 3.8 - 3.10
- CUDA 11.8+ (optional)
- FreeAskWord (Unity simulator)
- vint.pth [Link](https://drive.google.com/drive/folders/1a9yWR2iooXFAqjQHetz263--4_2FFggg)

### Installation

1.Create env and install deps (Download from freeaskworld_closed_loop)

```
git clone https://github.com/doraemonaaaa/freeaskworld_closed_loop.git
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2.Install the Vint_baseline

```bash
cd freeaskworld_closed_loop
git clone https://github.com/wksports/FreeAskWorld_Vint_Baseline.git
pip install -r requirements.txt
export TOPOMAP_BASE_DIR="$(pwd)/../topomaps"
export MODEL_PATH="$(pwd)/../model_weights/vint.pth"
```

3.Install the `vint_train` packages (run this inside the `vint_release/` directory):

```
git clone https://github.com/robodhruv/visualnav-transformer.git
pip install -e train/
```

---

## 📦 Usage
If you need to verify whether the model is usable, you can run: `run_inference.py`. It reads the contents of the `image` folder to perform inference.

### Recording Topological Maps

```bash
# Method : Manual recording
# - Drive robot in Unity
# - Save keyframes as 0.jpg, 1.jpg, 2.jpg, ...
python scripts/topomap_recorder.py
```

### Build workspace

```
colcon build --symlink-install
source ./install/setup.bash
```

### Running Navigation 

Edit run_vint_server.bash to change RUN_MODE

- Set RUN_MODE="single" for one episode or "full" for all 46 episodes

- Set SINGLE_EPISODE_NUMBER=15 to choose which episode to run (in single mode)

- Set FULL_RUN_EPISODE_COUNT=46 for total episodes (in full mode)

```
bash run_vint_server.bash
# wait for Unity connection, then start Unity simulator
```

---

## ⚙️ Configuration

### Navigation Parameters

Edit in `vint_baseline.py`:

```python
self.close_threshold = 3    # Distance threshold
self.radius = 2             # Search radius
self.waypoint_index = 2     # Waypoint selection (0-4)
self.max_v = 0.2           # Max linear velocity (m/s)
self.max_w = 0.4           # Max angular velocity (rad/s)
```

**Tuning tips:**
- `close_threshold` too large → switches nodes prematurely
- `close_threshold` too small → robot gets stuck
- `waypoint_index`: 0=nearest, 4=farthest; 2 is most stable

---

## 📚 System Architecture

### Navigation Pipeline

```
1. Image Capture → Maintain 6-image sliding window
2. Visual Localization → Compare with topomap nodes
3. Subgoal Selection → Choose next navigation target
4. Waypoint Prediction → ViNT model predicts trajectory
5. PD Controller → Convert waypoint to velocities
6. Command Execution → Publish to Unity
```

### File Structure

You should use  `mv` command to move the `vint_baseline.py`  file to the specified folder. 

The same applies to other files, following the directory structure below.

```
ros2/
├── src/
│   ├── simulator_messages/
│   │   └── msg/SimulatorCommand.msg
│   └── vln_connector/
│       ├── vln_connector/
│       │   ├── vln_connector.py      # Base ROS2 node
│       │   ├── vint_baseline.py      # ViNT navigation
│       │   └── events.py             # Event manager
│       └── package.xml
├── requirements.txt                   # Python dependencies
└── install/

topomaps/
├── episode_1/
├── episode_2/
└── ...

model_weights/
└── vint.pth

visualnav-transformer-main/
└── train/
    └── vint_train/                   # Required for model loading
```

---

## 📖 References

- [ViNT Official Repository](https://github.com/robodhruv/visualnav-transformer)
- [Unity ROS-TCP-Connector](https://github.com/Unity-Technologies/ROS-TCP-Connector)
- [ROS2 Humble Documentation](https://docs.ros.org/en/humble/)
