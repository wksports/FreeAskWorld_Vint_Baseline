"""
ViNT Baseline - Visual Topological Navigation

Main features:
- Maintains 6-image sliding window (5 history + 1 current)
- Uses ViNT model for waypoint prediction
- Converts waypoints to SimulatorCommand messages
"""

import threading
import cv2
import rclpy
import numpy as np
import sys
import time
import json
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms

from simulator_messages.msg import SimulatorCommand
from .vln_connector import VLNConnector

# Add ViNT training code to Python path for model class definitions
VINT_REPO_PATH = Path(__file__).parent.parent.parent.parent.parent / "visualnav-transformer-main" / "train"
if VINT_REPO_PATH.exists() and str(VINT_REPO_PATH) not in sys.path:
    sys.path.insert(0, str(VINT_REPO_PATH))


class ViNTBaseline(VLNConnector):
    """
    ViNT Navigation Controller
    
    Pipeline:
        1. ROS spin receives images
        2. InputData processes and maintains history queue
        3. ViNT Inference predicts waypoints
        4. Publish SimulatorCommand messages
    
    Inheritance:
        ViNTBaseline -> VLNConnector -> Node
        - VLNConnector: ROS2 node init, image subscription, message publishing
        - ViNTBaseline: ViNT model loading, inference logic, coordinate transforms
    """

    def __init__(self):
        """Initialize ViNT Baseline node"""
        super().__init__()  # Init ROS Node + RGBD Subscriber
        
        # ===== ViNT Initialization =====
        
        # PyTorch device (GPU if available, else CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model variables
        self.model = None              # ViNT model instance
        self.model_loaded = False      # Model load status flag
        
        # Image history queue (stores recent 6 PIL images)
        self.image_history = []
        
        # Current episode number (for auto topomap switching)
        self.current_episode = 1
        
        # ViNT context size: 5 history images + 1 current = 6 total
        self.context_size = 5
        
        # Inference thread
        self._inference_thread = None
        
        # Stop event
        self._stop_event = threading.Event()
        
        # ===== Topological Navigation Parameters =====
        import os
        
        # Topomap: keyframes of target trajectory
        self.topomap = []  # List[PIL.Image]
        self.topomap_base_dir = os.environ.get("TOPOMAP_BASE_DIR", None)
        self.current_episode = 1
        
        # Navigation state
        self.closest_node = 0  # Current closest topomap node index
        self.goal_node = -1    # Goal node index (-1 = not set)
        self.reached_goal = False
        
        # Navigation parameters (from env vars, defaults from original project)
        self.close_threshold = int(os.environ.get("VINT_CLOSE_THRESHOLD", "3"))
        self.radius = int(os.environ.get("VINT_RADIUS", "2"))
        self.waypoint_index = int(os.environ.get("VINT_WAYPOINT_INDEX", "2"))
        
        # PD controller parameters (from original robot.yaml)
        self.max_v = 0.2  # Max linear velocity
        self.max_w = 0.4  # Max angular velocity
        self.dt = 1.0 / 4.0  # Control period (4Hz)
        self.eps = 1e-8  # Small epsilon for numerical stability
        
        # Episode protection (prevent Unity delayed Stop commands from skipping episodes)
        self._episode_start_time = None
        self._episode_protection_duration = 0.5  # 0.5s protection period
        
        # Model path
        self.model_path = os.environ.get("MODEL_PATH", "model_weights/vint.pth")
        
        # Load topomap
        self._load_topomap()
        
        # Load ViNT model
        self._load_vint_model()
        
        # Log initialization
        self.get_logger().info(f"ViNT Baseline initialized - Model: {self.model_path}, Device: {self.device}")

    def simulator_command_callback(self, msg):
        """Override parent method to handle Unity responses with episode protection"""
        # Check if in episode protection period
        if self._episode_start_time is not None:
            elapsed = time.time() - self._episode_start_time
            if elapsed < self._episode_protection_duration:
                # Parse Unity command
                raw = msg.data.strip()
                try:
                    data = json.loads(raw)
                    method = data.get("method") if isinstance(data, dict) else data
                except:
                    method = raw
                
                # Ignore Stop commands during protection period
                if method and method.lower() == "stop":
                    self.get_logger().warn(f"🛡️ Ignoring Unity Stop during protection period ({elapsed:.1f}s < {self._episode_protection_duration}s)")
                    return
        
        # Log received Unity commands (for debugging)
        raw = msg.data.strip()
        self.get_logger().info(f"[Unity → ROS2] Received: {raw[:100]}")
        
        # Call parent method to handle basic logic (including Stop event)
        super().simulator_command_callback(msg)

    def _load_topomap(self, episode=None):
        """
        Load topological map for specified episode
        
        Args:
            episode: Episode number (1-46), uses current if None
        
        Directory structure:
            TOPOMAP_BASE_DIR/
            ├── episode_1/
            │   ├── 0.jpg, 1.jpg, 2.jpg, ...
            ├── episode_2/
            │   └── ...
        """
        if self.topomap_base_dir is None:
            self.get_logger().warn(
                "TOPOMAP_BASE_DIR not set. Running in exploration mode (no goal-directed navigation)."
            )
            return
        
        # Use specified or current episode
        episode_num = episode if episode is not None else self.current_episode
        
        # Build topomap directory path
        topomap_dir = Path(self.topomap_base_dir) / f"episode_{episode_num}"
        
        if not topomap_dir.exists():
            self.get_logger().error(f"Topomap directory not found: {topomap_dir}")
            return
        
        # Clear previous topomap
        self.topomap = []
        
        # Load all image files, sorted by filename
        image_files = sorted(
            topomap_dir.glob("*.jpg"),
            key=lambda x: int(x.stem)  # Sort numerically (0.jpg, 1.jpg, ...)
        )
        
        if len(image_files) == 0:
            self.get_logger().error(f"No images found in topomap directory: {topomap_dir}")
            return
        
        # Load images into memory
        for img_file in image_files:
            img = Image.open(img_file)
            self.topomap.append(img)
        
        # Set goal node to last node
        self.goal_node = len(self.topomap) - 1
        
        # Force start from beginning (fixes visual localization errors)
        self.closest_node = 0
        self.last_closest_node = 0  # Sync init for anti-stuck mechanism
        self.get_logger().info(f"🚀 Starting navigation from node 0, goal is node {self.goal_node}")
        
        self.get_logger().info(
            f"Loaded topomap for episode {episode_num} with {len(self.topomap)} nodes from {topomap_dir}"
        )

    def _load_vint_model(self):
        """
        Load ViNT model from checkpoint file
        
        Steps:
        1. Import ViNT model class
        2. Check if model file exists
        3. Load checkpoint (weights + config)
        4. Create model instance and load weights
        5. Move to device (GPU/CPU) and set to eval mode
        """
        try:
            # Import ViNT model definition
            from vint_train.models.vint.vint import ViNT
            
            # Check model file exists
            if not Path(self.model_path).exists():
                self.get_logger().error(f"Model file not found: {self.model_path}")
                return
            
            # Load model checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            # Create ViNT model instance with default config
            model = ViNT(
                context_size=5,                    # Number of history images
                len_traj_pred=5,                   # Number of predicted waypoints
                learn_angle=True,                  # Learn angle information
                obs_encoder="efficientnet-b0",     # Observation encoder
                obs_encoding_size=512,             # Encoding dimension
                late_fusion=False,                 # Late fusion flag
                mha_num_attention_heads=2,         # Multi-head attention heads
                mha_num_attention_layers=2,        # Attention layers
                mha_ff_dim_factor=4,               # Feed-forward dimension factor
            )
            
            # Load model weights (handle both multi-GPU and single-GPU checkpoints)
            loaded_model = checkpoint["model"]
            try:
                state_dict = loaded_model.module.state_dict()
                model.load_state_dict(state_dict, strict=False)
            except AttributeError:
                state_dict = loaded_model.state_dict()
                model.load_state_dict(state_dict, strict=False)
            
            # Move to device and set eval mode
            model.to(self.device)
            model.eval()  # Disable dropout and batch normalization training behavior
            
            # Save model instance and mark as loaded
            self.model = model
            self.model_loaded = True
            self.get_logger().info("ViNT model loaded successfully")
            
        except Exception as e:
            self.get_logger().error(f"Error loading ViNT model: {e}")
            self.model = None
            self.model_loaded = False

    # =====================================================
    # Main Control Logic
    # =====================================================
    def control_once_async(self):
        """Execute one control cycle (non-blocking)"""
        # Skip if previous inference still running
        if self._inference_thread is not None and self._inference_thread.is_alive():
            return
        
        # Get RGB image for ViNT inference
        if self.rgb_image is None:
            return  # Wait for image data
        
        rgb_snapshot = self.rgb_image.copy()
        
        # Start inference in separate thread (non-blocking)
        self._inference_thread = threading.Thread(
            target=self.Inference,
            kwargs={"rgb": rgb_snapshot}
        )
        self._inference_thread.start()

    
    # Input Adapter 

    def InputData(self, **kwargs):
        """
        Input data adapter: process RGB images and maintain history queue
        
        Args:
            **kwargs: Dict with 'rgb' key (OpenCV BGR image)
        
        Returns:
            list: Recent 6 PIL images
        
        ViNT is a temporal model that needs motion trajectory context:
        - 6 images = 5 history + 1 current
        - Model predicts next motion by comparing these images
        """
        rgb_img = kwargs.get("rgb")
        
        # Convert BGR to RGB (OpenCV uses BGR, PIL/DL frameworks use RGB)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image (ViNT preprocessing format)
        pil_image = Image.fromarray(rgb_img)
        
        # Add to history queue
        self.image_history.append(pil_image)
        
        # Keep queue length <= 6 (FIFO)
        if len(self.image_history) > 6:
            self.image_history.pop(0)
        
        return self.image_history

    # =====================================================
    # Inference Adapter - ViNT Inference Logic
    # =====================================================
    def Inference(self, **args):
        """
        ViNT topological navigation inference (thread-safe)
        
        Core inference function implementing complete ViNT navigation logic.
        
        Args:
            **args: Dict with 'rgb' key (ViNT only needs RGB images)
        
        Pipeline:
        1. Process input images, maintain history queue
        2. Check if goal reached
        3. With topomap: distance prediction, visual localization, subgoal selection
        4. Without topomap: exploration mode (testing only)
        5. PD controller: convert waypoint to velocity commands
        6. Publish SimulatorCommand
        """
        try:
            # Step 1: Process input data, get image history
            image_history = self.InputData(**args)
            
            # Step 2: Check image count
            if len(image_history) < 6:
                self.get_logger().warn(f"Not enough images for ViNT: {len(image_history)}/6")
                return self._create_dummy_simulator_command()
            
            # Step 3: Check model loaded
            if not self.model_loaded or self.model is None:
                self.get_logger().warn("ViNT model not loaded, using dummy navigation")
                return self._create_dummy_simulator_command()
            
            # Step 4: Check if goal reached
            if self.reached_goal:
                self.handle_stop_event("Reached navigation goal")
                return self._create_stop_simulator_command()
            
            # Step 5: Select navigation mode
            if len(self.topomap) > 0:
                # Topological navigation: use target trajectory
                waypoint = self._topological_navigation(image_history)
            else:
                # Exploration mode: no goal (testing only)
                self.get_logger().warn("No topomap loaded, using exploration mode", throttle_duration_sec=5.0)
                waypoint = self._exploration_mode(image_history)
            
            # Step 5.5: Velocity normalization (from navigate.py L226-227)
            # Normalize only position (x,y), keep direction (cos(θ), sin(θ))
            normalize = True  # Assume model uses normalization (from vint.yaml)
            if normalize:
                max_v = 0.2  # From robot.yaml
                frame_rate = 4  # From robot.yaml
                waypoint[:2] *= (max_v / frame_rate)  # Normalize position only
            
            # Step 6: Check for zero waypoint (model output all zeros, robot stuck)
            if np.allclose(waypoint[:2], [0.0, 0.0]):
                self.handle_stop_event("Model output zero waypoint (robot stuck)")
                return
            
            # Step 7: Use PD controller to compute velocities
            v, w = self.pd_controller(waypoint)
            
            # Step 8: Convert to Unity format
            # Unity Move interface: method_params = "x,y,yaw"
            # Velocity to displacement: (v,w) → (x,y,yaw)
            x = v * self.dt      # Forward displacement = linear velocity × dt
            y = 0.0              # Lateral displacement = 0 (differential drive)
            yaw = w * self.dt    # Rotation angle = angular velocity × dt
            
            sim_cmd = SimulatorCommand()
            sim_cmd.header.stamp = self.get_clock().now().to_msg()
            sim_cmd.header.frame_id = "vint_nav"
            sim_cmd.method = "Move"
            sim_cmd.method_params = f"{x:.6f},{y:.6f},{yaw:.6f}"
            
            # Step 9: Publish navigation command
            try:
                self.publish_simulator_command(sim_cmd)
            except Exception as e:
                self.get_logger().error(f"Failed to publish command: {e}")
                self.handle_stop_event(f"Publisher error: {e}")
            
            # Log inference result
            if len(waypoint) >= 4:
                angle_deg = np.degrees(np.arctan2(waypoint[3], waypoint[2]))
                self.get_logger().info(
                    f"ViNT: node={self.closest_node}/{self.goal_node}, "
                    f"waypoint=[{waypoint[0]:.3f}, {waypoint[1]:.3f}, θ={angle_deg:.1f}°], "
                    f"PD(v={v:.3f},w={w:.3f}) → Unity({x:.3f},{y:.3f},{yaw:.3f})"
                )
            else:
                self.get_logger().info(
                    f"ViNT: node={self.closest_node}/{self.goal_node}, "
                    f"waypoint=[{waypoint[0]:.3f}, {waypoint[1]:.3f}], "
                    f"PD(v={v:.3f},w={w:.3f}) → Unity({x:.3f},{y:.3f},{yaw:.3f})"
                )
            
        except Exception as e:
            self.get_logger().error(f"ViNT inference error: {e}")
            return

    def _topological_navigation(self, image_history):
        """
        Topological navigation mode: navigate using target trajectory
        
        Core ViNT navigation logic based on navigate.py.
        
        Steps:
        1. Distance prediction: predict distances to topomap nodes
        2. Visual localization: find closest node
        3. Subgoal selection: choose next target based on distance threshold
        4. Waypoint prediction: predict trajectory to subgoal
        5. Goal check: determine if final goal reached
        
        Returns:
            waypoint: np.ndarray - selected waypoint [x, y] or [x, y, cos(θ), sin(θ)]
        """
        # Ensure only last 6 images used
        if len(image_history) > 6:
            image_history = image_history[-6:]
        
        # Step 1: Prepare observation images
        obs_images = self._transform_images(image_history)
        
        # Step 2: Determine search range (from navigate.py L196-197)
        start = max(self.closest_node - self.radius, 0)
        end = min(self.closest_node + self.radius + 1, self.goal_node)
        
        # Step 3: Prepare batch data (from navigate.py L202-206)
        batch_obs_imgs = []
        batch_goal_data = []
        for i, sg_img in enumerate(self.topomap[start: end + 1]):
            transf_obs_img = self._transform_images(image_history)
            goal_data = self._transform_images([sg_img])
            batch_obs_imgs.append(transf_obs_img)
            batch_goal_data.append(goal_data)
        
        # Merge into batch
        batch_obs = torch.cat(batch_obs_imgs, dim=0).to(self.device)
        batch_goal = torch.cat(batch_goal_data, dim=0).to(self.device)
        
        # Step 4: Model inference - predict distances and waypoints
        with torch.no_grad():
            distances, waypoints = self.model(batch_obs, batch_goal)
        
        # Convert to numpy
        distances = distances.cpu().numpy().flatten()
        waypoints = waypoints.cpu().numpy()  # [batch, num_waypoints, 2or4]
        
        self.get_logger().debug(f"Waypoints shape: {waypoints.shape}, distances shape: {distances.shape}")
        
        # ViNT waypoint format: [x, y, cos(θ), sin(θ)]
        if len(waypoints.shape) == 3:
            self.get_logger().debug(f"Waypoints shape: {waypoints.shape} (format: [x, y, cos(θ), sin(θ)])")
        
        # Step 5: Visual localization
        min_dist_idx = np.argmin(distances)
        
        # Step 6: Subgoal selection and waypoint choice (from navigate.py L218-224)
        if distances[min_dist_idx] > self.close_threshold:
            # Still far from current node, keep moving toward it
            chosen_waypoint = waypoints[min_dist_idx][self.waypoint_index]
            self.closest_node = start + min_dist_idx
        else:
            # Close to current node, select next node
            next_waypoint_idx = min(min_dist_idx + 1, len(waypoints) - 1)
            chosen_waypoint = waypoints[next_waypoint_idx][self.waypoint_index]
            self.closest_node = min(start + min_dist_idx + 1, self.goal_node)
        
        # Step 7: Check if final goal reached (from navigate.py L231)
        reached_goal = self.closest_node == self.goal_node
        if reached_goal:
            self.reached_goal = True
            self.get_logger().info("🎯 Reached final goal!")
            return np.array([0.0, 0.0])  # Return zero waypoint to trigger stop
        
        return chosen_waypoint

    def _exploration_mode(self, image_history):
        """
        Exploration mode: goal-free navigation (testing only)
        
        Used when no topomap loaded. Model predicts forward waypoint 
        without explicit goal.
        
        Returns:
            waypoint: np.ndarray - predicted waypoint [x, y]
        """
        # Ensure only last 6 images used
        original_len = len(image_history)
        if len(image_history) > 6:
            image_history = image_history[-6:]
            self.get_logger().warn(f"Trimmed image_history from {original_len} to {len(image_history)}")
        
        # Use current observation as "goal" (self-navigation)
        obs_images = self._transform_images(image_history).to(self.device)
        goal_images = self._transform_images([image_history[-1]]).to(self.device)
        
        with torch.no_grad():
            _, waypoints = self.model(obs_images, goal_images)
        
        waypoints = waypoints.cpu().numpy()[0]  # [num_waypoints, 2or4]
        
        # Ensure waypoints shape correct
        if len(waypoints.shape) == 2 and waypoints.shape[1] >= 2:
            chosen_waypoint = waypoints[self.waypoint_index][:2]  # Take [x, y] only
        else:
            self.get_logger().warn(f"Unexpected waypoints shape in exploration: {waypoints.shape}")
            chosen_waypoint = waypoints[0][:2] if len(waypoints) > 0 else np.array([0.1, 0.0])
        
        return chosen_waypoint

    def pd_controller(self, waypoint):
        """
        PD controller: strictly follows original pd_controller.py
        
        Args:
            waypoint: np.ndarray - 2D or 4D vector
        
        Returns:
            v: float - linear velocity (m/s)
            w: float - angular velocity (rad/s)
        """
        assert len(waypoint) == 2 or len(waypoint) == 4, "waypoint must be a 2D or 4D vector"
        
        if len(waypoint) == 2:
            dx, dy = waypoint
        else:
            dx, dy, hx, hy = waypoint
            
        # Original logic: only use predicted heading when dx and dy near zero
        if len(waypoint) == 4 and np.abs(dx) < self.eps and np.abs(dy) < self.eps:
            v = 0
            w = self._clip_angle(np.arctan2(hy, hx)) / self.dt
        elif np.abs(dx) < self.eps:
            v = 0
            w = np.sign(dy) * np.pi / (2 * self.dt)
        else:
            v = dx / self.dt
            w = np.arctan(dy / dx) / self.dt
            
        # Clip velocity ranges
        v = np.clip(v, 0, self.max_v)
        w = np.clip(w, -self.max_w, self.max_w)
        
        return v, w
    
    def _clip_angle(self, theta):
        """Clip angle to [-pi, pi] range"""
        theta %= 2 * np.pi
        if -np.pi < theta < np.pi:
            return theta
        return theta - 2 * np.pi

    def _transform_images(self, images):
        """
        Image preprocessing: PIL Image → Tensor
        
        Args:
            images: List[PIL.Image]
        
        Returns:
            torch.Tensor - shape [1, C*T, H, W]
                C: channels (3)
                T: time steps (len(images))
                H, W: image height and width
        
        ViNT expects input format: [batch, 3*context_size, H, W]
        Example: 6 images → [1, 18, 85, 64]
        """
        if len(images) > 0:
            self.get_logger().debug(f"_transform_images: {len(images)} images, first image size: {images[0].size}")
        
        transform = transforms.Compose([
            transforms.Resize((85, 64)),  # ViNT default input size (H, W)
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Transform each image: [C, H, W]
        tensors = [transform(img) for img in images]
        
        # Concat on channel dim: [C, H, W] × T → [C*T, H, W]
        # Add batch dim: [C*T, H, W] → [1, C*T, H, W]
        stacked = torch.cat(tensors, dim=0).unsqueeze(0)
        
        self.get_logger().debug(f"_transform_images output shape: {stacked.shape}")
        
        return stacked

    def _create_dummy_simulator_command(self):
        """
        Create dummy navigation command (when ViNT not ready)
        
        Used when:
        - Image history insufficient (< 6 images)
        - Model not loaded
        - Safe default behavior
        
        Behavior: Slow forward movement (X=0.1)
        """
        sim_cmd = SimulatorCommand()
        sim_cmd.header.stamp = self.get_clock().now().to_msg()
        sim_cmd.header.frame_id = "dummy_nav"
        sim_cmd.method = "Move"
        sim_cmd.method_params = "0.1,0.0,0.0"  # [forward, lateral, yaw]
        
        return sim_cmd

    def _create_stop_simulator_command(self):
        """
        Create stop navigation command (safety mechanism)
        
        Used when:
        - Inference error
        - Emergency stop
        - Safe failure behavior
        """
        sim_cmd = SimulatorCommand()
        sim_cmd.header.stamp = self.get_clock().now().to_msg()
        sim_cmd.header.frame_id = "stop_nav"
        sim_cmd.method = "Stop"
        sim_cmd.method_params = ""
        
        return sim_cmd

    def handle_stop_event(self, reason="unknown"):
        """
        Send Stop command to Unity, let Unity response trigger actual Stop event
        
        Args:
            reason: Stop reason
        """
        self.get_logger().info(f"🛑 Stop triggered: {reason}")
        
        sim_cmd = SimulatorCommand()
        sim_cmd.header.stamp = self.get_clock().now().to_msg()
        sim_cmd.header.frame_id = "vint_nav"
        sim_cmd.method = "Stop"
        sim_cmd.method_params = ""
        
        self.publish_simulator_command(sim_cmd)
        self.get_logger().info("🛑 Stop command sent to Unity, waiting for Unity response")

    def reset_for_new_episode(self, episode_num):
        """
        Reset node state for new episode
        
        Args:
            episode_num: Episode number
        """
        self.current_episode = episode_num
        self.reached_goal = False
        self.closest_node = 0
        self.image_history = []
        self._stop_event.clear()
        self._episode_start_time = time.time()
        
        self.get_logger().info(f"🔄 ViNT Baseline reset for episode {episode_num}")

    def destroy_node(self):
        """Destroy node and cleanup resources"""
        super().destroy_node()


# =====================================================
# Main Loop
# =====================================================
def main(args=None):
    """
    Main function: multi-episode ViNT Baseline node
    
    Workflow:
    1. Initialize ROS2
    2. Loop through episodes:
       - Create/reset ViNTBaseline node
       - Run navigation
       - Reset state after reaching goal
    3. Exit after all episodes complete
    
    Args:
        args: ROS2 init arguments (usually None)
    """
    # Initialize ROS2
    rclpy.init(args=args)
    
    # Single run configuration
    import os
    total_episodes = int(os.environ.get("TOTAL_EPISODES", "46"))  # Default 46 episodes
    
    print(f"🚀 Starting single run with {total_episodes} episodes")
    
    node = None
    
    try:
        # Single run: episodes 1-46
        for episode in range(1, total_episodes + 1):
            print(f"\n===== Episode {episode}/{total_episodes} =====")
            
            # Create or reset node
            if node is None:
                node = ViNTBaseline()
                node.current_episode = episode
            else:
                node.reset_for_new_episode(episode)
                
            # Auto-load corresponding topomap
            node._load_topomap(episode)
            
            # Clear Stop event to avoid immediate exit
            node._stop_event.clear()
            
            # Set episode protection start time (prevent Unity delayed Stop)
            node._episode_start_time = time.time()
            
            node.get_logger().info(f"🛡️ Episode {episode} initialized with {node._episode_protection_duration}s protection period")
            
            # Single episode navigation loop
            while rclpy.ok():
                # Process ROS callbacks
                rclpy.spin_once(node, timeout_sec=0.05)
                
                # Execute control cycle
                node.control_once_async()
                
                # Check if current episode complete
                if node._stop_event.is_set():
                    node.get_logger().info("🔁 ViNT Baseline reseting...")
                    break
            
            # Wait after episode for Unity to fully reset state
            if episode < total_episodes:  # Wait only if not last episode
                print(f"Episode {episode} completed. Waiting 10 seconds for Unity to reset...")
                time.sleep(10.0)
                print(f"✅ Ready for next episode!")
                
                # Clear any residual Stop events
                if node is not None:
                    node._stop_event.clear()
                    node.get_logger().info("Cleared residual stop events")
        
        # All episodes complete
        print(f"\n🎉 All {total_episodes} episodes completed!")
        print("🏁 ViNT Baseline finished successfully!")
        
    except KeyboardInterrupt:
        # Catch Ctrl+C
        print("\nInterrupted by user")
    finally:
        # Cleanup resources
        if node is not None:
            # Wait for inference thread to finish
            if node._inference_thread is not None:
                node.get_logger().info("Waiting for inference thread to finish...")
                node._inference_thread.join(timeout=2.0)
            
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
