#!/usr/bin/env python3
"""
优化拓扑地图录制工具 - 支持自定义拍照设置

功能：
1. 使用和 vint_baseline.py 相同的话题订阅
2. Unity 界面 WASD 控制，终端控制录制
3. 自定义拍照间隔（如1.5秒/帧）
4. 手动控制结束录制（确认到达终点后）
5. 自动保存为 ViNT 格式

使用方法：
python3 scripts/topomap_recorder.py

参数配置：
在main()函数中修改TOPOMAP_NAME、SAVE_INTERVAL等参数
"""

import os
import sys
import time
import threading
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import pickle
import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as RosImage
from nav_msgs.msg import Odometry


class OptimizedTopomapRecorder(Node):
    """优化拓扑地图录制器 - 支持自定义拍照设置"""
    
    def __init__(self, topomap_name, save_interval=0.7, max_images=200):
        super().__init__('optimized_topomap_recorder')
        
        self.topomap_name = topomap_name
        self.save_interval = save_interval
        self.max_images = max_images
        
        # 创建保存目录
        self.project_root = Path(__file__).parent.parent
        self.topomap_dir = self.project_root / "topomaps" / "images" / topomap_name
        self.topomap_dir.mkdir(parents=True, exist_ok=True)
        
        # 录制状态
        self.recording = False
        self.image_count = 0
        self.last_save_time = 0
        self.start_time = None
        self.paused = False  # 新增：暂停功能
        self.manual_mode = True  # 新增：手动控制模式（不限制最大图片数）
        
        # 数据存储
        self.rgb_image = None
        self.base_pose = None
        self.trajectory_data = []
        
        # 连接状态
        self.unity_connected = False
        self.last_image_time = 0
        self.last_odom_time = 0
        self.connection_retry_count = 0
        self.max_retry_before_warning = 3
        self.connection_check_interval = 2.0  # 2秒检查一次连接状态
        
        # 使用和 vint_baseline.py 完全相同的话题订阅
        self.rgb_sub = self.create_subscription(
            RosImage,
            '/simulator_msg/camera/color/image_raw',  # 和 vint_baseline 相同
            self.rgb_callback,
            10
        )
        
        self.odom_sub = self.create_subscription(
            Odometry,
            '/simulator_msg/odom',  # 和 vint_baseline 相同
            self.robot_odom_callback,
            10
        )
        
        self.get_logger().info(f"Optimized Topomap Recorder initialized")
        self.get_logger().info(f"Save directory: {self.topomap_dir}")
        self.get_logger().info(f"Save interval: {self.save_interval}s per frame")
        self.get_logger().info(f"Manual mode: {'ON' if self.manual_mode else 'OFF'}")
        self.get_logger().info("")
        self.get_logger().info("📡 Subscribed to Unity topics:")
        self.get_logger().info("  - /simulator_msg/camera/color/image_raw")
        self.get_logger().info("  - /simulator_msg/odom")
        self.get_logger().info("")
        self.get_logger().info("🎮 Enhanced Controls:")
        self.get_logger().info("  - Press 's' to start recording")
        self.get_logger().info("  - Press 'p' to pause/resume recording")
        self.get_logger().info("  - Press 'q' to stop and save (when you reach endpoint)")
        self.get_logger().info("  - Press 'c' to check current status")
        self.get_logger().info("  - Use WASD in Unity to drive the robot")
        
        # 启动连接检测定时器
        self.connection_timer = self.create_timer(
            self.connection_check_interval, 
            self.check_unity_connection
        )
        
    def rgb_callback(self, msg):
        """处理 RGB 图像 - 复用 vint_baseline 的逻辑"""
        h, w = msg.height, msg.width
        rgb = np.frombuffer(msg.data, dtype=np.uint8)

        if rgb.size == h * w * 4:  # RGBA
            rgb = rgb.reshape((h, w, 4))
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)
        elif rgb.size == h * w * 3:  # RGB
            rgb = rgb.reshape((h, w, 3))
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        else:
            self.get_logger().error(f"Unexpected RGB data size: {rgb.size}")
            return

        self.rgb_image = rgb
        
        # 更新图像接收时间（用于连接检测）
        self.last_image_time = time.time()
        
        # 如果正在录制且未暂停，检查是否需要保存
        if self.recording and not self.paused:
            current_time = time.time()
            if current_time - self.last_save_time >= self.save_interval:
                self.save_current_frame()
                self.last_save_time = current_time

    def robot_odom_callback(self, msg):
        """处理里程计 - 复用 vint_baseline 的逻辑"""
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation

        # 计算 yaw（和 vint_baseline 相同的计算方式）
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

        self.base_pose = (p.x, p.y, yaw)
        
        # 更新里程计接收时间（用于连接检测）
        self.last_odom_time = time.time()
    
    def check_unity_connection(self):
        """检查Unity连接状态"""
        current_time = time.time()
        
        # 检查连接状态（增加到5秒超时，减少误判）
        image_connected = (current_time - self.last_image_time) < 5.0 if self.last_image_time > 0 else False
        odom_connected = (current_time - self.last_odom_time) < 5.0 if self.last_odom_time > 0 else False
        
        # 至少需要图像数据连接
        new_connection_status = image_connected
        
        if new_connection_status != self.unity_connected:
            self.unity_connected = new_connection_status
            if self.unity_connected:
                self.get_logger().info("✅ Unity connected! You can start recording now.")
                # 自动恢复录制（如果之前因断线而暂停）
                if self.recording and self.paused:
                    self.paused = False
                    self.last_save_time = time.time()  # 重置时间，避免立即拍照
                    self.get_logger().info("🔄 Recording auto-resumed!")
            else:
                self.get_logger().warn("❌ Unity disconnected! Please check connection.")
                if self.recording and not self.paused:
                    self.get_logger().warn("⚠️ Recording paused due to connection loss")
                    self.paused = True
                    self.get_logger().info("💡 Recording will auto-resume when Unity reconnects")
    
    def start_recording(self):
        """开始录制"""
        if self.recording:
            self.get_logger().warn("Already recording!")
            return
            
        # 检查Unity连接状态
        if not self.unity_connected:
            self.get_logger().error("❌ Cannot start recording: Unity not connected!")
            self.get_logger().info("Please make sure Unity is running and TCP Endpoint is connected.")
            return
            
        self.recording = True
        self.image_count = 0
        self.start_time = time.time()
        self.last_save_time = time.time()
        self.trajectory_data = []
        
        self.get_logger().info("🔴 Recording started!")
        if self.manual_mode:
            self.get_logger().info("📍 Manual mode: Record until you press 'q' at endpoint")
        else:
            self.get_logger().info(f"🎯 Target: {self.max_images} images")
        self.get_logger().info(f"📸 Capturing every {self.save_interval}s")
        self.get_logger().info("🎮 Drive your robot in Unity with WASD now!")
        
    def pause_recording(self):
        """暂停/恢复录制"""
        if not self.recording:
            self.get_logger().warn("Not recording!")
            return
            
        self.paused = not self.paused
        if self.paused:
            self.get_logger().info("⏸️ Recording paused")
        else:
            self.get_logger().info("▶️ Recording resumed")
            self.last_save_time = time.time()  # 重置时间，避免立即拍照
    
    def check_status(self):
        """检查当前状态"""
        # 显示连接状态
        connection_status = "✅ Connected" if self.unity_connected else "❌ Disconnected"
        self.get_logger().info(f"🔗 Unity: {connection_status}")
        
        if not self.recording:
            self.get_logger().info("📊 Status: Not recording")
        elif self.paused:
            self.get_logger().info(f"📊 Status: Paused - {self.image_count} images captured")
        else:
            elapsed = time.time() - self.start_time
            next_capture = self.save_interval - (time.time() - self.last_save_time)
            self.get_logger().info(
                f"📊 Status: Recording - {self.image_count} images, "
                f"elapsed: {elapsed:.1f}s, next capture in: {max(0, next_capture):.1f}s"
            )
        
    def stop_recording(self):
        """停止录制并保存"""
        if not self.recording:
            self.get_logger().warn("Not recording!")
            return
            
        self.recording = False
        
        # 保存轨迹数据
        self.save_trajectory_data()
        
        duration = time.time() - self.start_time
        self.get_logger().info("⏹️ Recording stopped!")
        self.get_logger().info(f"Saved {self.image_count} images in {duration:.1f}s")
        self.get_logger().info(f"Average interval: {duration/max(1, self.image_count):.2f}s")
        self.get_logger().info(f"Topomap saved to: {self.topomap_dir}")
        
        # 显示使用说明
        self.print_usage_instructions()
        
        # 自动准备下一个episode
        self.auto_prepare_next_episode()
        
        # 提示用户可以继续录制
        self.get_logger().info("")
        self.get_logger().info("🔄 Ready for next episode!")
        self.get_logger().info("Press 's' to start recording next episode, or 'q' to quit completely")
    
    def prepare_next_episode(self):
        """准备录制下一个episode"""
        if self.recording:
            self.get_logger().warn("Cannot prepare next episode while recording!")
            return
        
        # 找到下一个episode编号
        next_episode = find_next_episode_number()
        new_topomap_name = f"episode_{next_episode}"
        
        # 更新目录
        self.topomap_dir = self.project_root / "topomaps" / "images" / new_topomap_name
        self.topomap_dir.mkdir(parents=True, exist_ok=True)
        
        # 重置状态
        self.image_count = 0
        self.trajectory_data = []
        self.paused = False
        
        self.get_logger().info(f"🆕 Prepared for {new_topomap_name}")
        self.get_logger().info(f"📁 Directory: {self.topomap_dir}")
        self.get_logger().info("Press 's' to start recording when ready!")
    
    def reset_connection_status(self):
        """重置连接状态（用于断联重连后的恢复）"""
        self.get_logger().info("🔄 Resetting connection status...")
        
        # 重置连接状态
        self.unity_connected = False
        self.last_image_time = 0
        self.last_odom_time = 0
        
        # 强制检查当前状态
        current_time = time.time()
        if self.rgb_image is not None:
            self.last_image_time = current_time
        if self.latest_pose is not None:
            self.last_odom_time = current_time
            
        # 立即检查连接
        self.check_unity_connection()
        
        self.get_logger().info("✅ Connection status reset complete!")
    
    def auto_prepare_next_episode(self):
        """自动准备下一个episode（静默模式）"""
        if self.recording:
            return
        
        # 找到下一个episode编号
        next_episode = find_next_episode_number()
        new_topomap_name = f"episode_{next_episode}"
        
        # 更新目录
        self.topomap_dir = self.project_root / "topomaps" / "images" / new_topomap_name
        self.topomap_dir.mkdir(parents=True, exist_ok=True)
        
        # 重置状态
        self.image_count = 0
        self.trajectory_data = []
        self.paused = False
        
    def save_current_frame(self):
        """保存当前帧"""
        if self.rgb_image is None:
            self.get_logger().warn("No image available, skipping frame")
            return
            
        # 在手动模式下不限制图片数量
        if not self.manual_mode and self.image_count >= self.max_images:
            self.get_logger().warn(f"Reached maximum images ({self.max_images}), stopping...")
            self.stop_recording()
            return
        
        try:
            # 保存图像（转换为 RGB 格式）
            rgb_img = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_img)
            
            img_path = self.topomap_dir / f"{self.image_count}.jpg"
            pil_img.save(img_path, quality=90)
            
            # 记录轨迹数据
            if self.base_pose:
                self.trajectory_data.append({
                    'image_id': self.image_count,
                    'position': [self.base_pose[0], self.base_pose[1]],
                    'yaw': self.base_pose[2],
                    'timestamp': time.time()
                })
            else:
                # 如果没有里程计数据，使用默认值
                self.trajectory_data.append({
                    'image_id': self.image_count,
                    'position': [0.0, 0.0],
                    'yaw': 0.0,
                    'timestamp': time.time()
                })
            
            self.image_count += 1
            
            # 显示进度
            if self.image_count % 5 == 0 or self.image_count <= 10:
                if self.manual_mode:
                    self.get_logger().info(f"📸 Saved image {self.image_count} (manual mode)")
                else:
                    self.get_logger().info(f"📸 Saved image {self.image_count}/{self.max_images}")
                
        except Exception as e:
            self.get_logger().error(f"Error saving frame: {e}")
    
    def save_trajectory_data(self):
        """保存轨迹数据为 ViNT 格式"""
        try:
            # 转换为 ViNT 期望的格式
            positions = np.array([data['position'] for data in self.trajectory_data])
            yaws = np.array([data['yaw'] for data in self.trajectory_data])
            
            traj_data = {
                'position': positions,  # [N, 2] - xy coordinates
                'yaw': yaws,           # [N,] - yaw angles
                'timestamps': [data['timestamp'] for data in self.trajectory_data]
            }
            
            # 保存到拓扑地图目录
            traj_path = self.topomap_dir / 'traj_data.pkl'
            with open(traj_path, 'wb') as f:
                pickle.dump(traj_data, f)
                
            self.get_logger().info(f"Trajectory data saved to: {traj_path}")
            
        except Exception as e:
            self.get_logger().error(f"Error saving trajectory data: {e}")
    
    def print_usage_instructions(self):
        """打印使用说明"""
        self.get_logger().info("")
        self.get_logger().info("=" * 50)
        self.get_logger().info("🎯 Topomap created successfully!")
        self.get_logger().info("")
        self.get_logger().info("To use with ViNT:")
        self.get_logger().info(f"export TOPOMAP_DIR=\"{self.topomap_dir}\"")
        self.get_logger().info("bash run_vint_server.bash")
        self.get_logger().info("")
        self.get_logger().info("Files created:")
        self.get_logger().info(f"  - Images: {self.topomap_dir}/*.jpg")
        self.get_logger().info(f"  - Trajectory: {self.topomap_dir}/traj_data.pkl")
        self.get_logger().info("=" * 50)


def keyboard_input_thread(recorder):
    """键盘输入线程 - 改进的断联重连处理"""
    try:
        print("\n🎮 Enhanced Keyboard Controls:")
        print("  s - Start recording (auto-prepares next episode after q)")
        print("  p - Pause/Resume recording")
        print("  c - Check current status")
        print("  n - Manually prepare next episode (optional)")
        print("  r - Reset connection status (if commands not working)")
        print("  q - Stop recording and auto-prepare next episode")
        print("  h - Show help")
        print("")
        
        while rclpy.ok():
            try:
                key = input().strip().lower()
                
                # 添加连接检查（先强制检查一次连接状态）
                if key in ['s']:
                    recorder.check_unity_connection()  # 强制检查连接
                    if not recorder.unity_connected:
                        print("⚠️  Unity not connected! Please wait for connection or press 'r' to reset.")
                        continue
                
                if key == 's':
                    recorder.start_recording()
                elif key == 'p':
                    recorder.pause_recording()
                elif key == 'c':
                    recorder.check_unity_connection()  # 强制检查连接
                    recorder.check_status()
                elif key == 'n':
                    recorder.prepare_next_episode()
                elif key == 'r':
                    print("🔄 Resetting connection status...")
                    recorder.reset_connection_status()
                elif key == 'q':
                    if recorder.recording:
                        recorder.stop_recording()
                    break
                elif key == 'h':
                    print("\n🎮 Enhanced Controls:")
                    print("  s - Start recording (auto-prepares next episode after q)")
                    print("  p - Pause/Resume recording")
                    print("  c - Check current status")
                    print("  n - Manually prepare next episode (optional)")
                    print("  r - Reset connection status (if commands not working)")
                    print("  q - Stop recording and auto-prepare next episode")
                    print("  h - Show this help")
                    print("  Use WASD in Unity to drive the robot")
                    print(f"  Current interval: {recorder.save_interval}s per frame")
                elif key:
                    print(f"Unknown command: '{key}'. Press 'h' for help.")
                    
            except Exception as e:
                print(f"Input error: {e}. Continuing...")
                time.sleep(0.1)
                
    except (EOFError, KeyboardInterrupt):
        pass


def find_next_episode_number():
    """
    自动找到下一个可用的episode编号
    
    返回：
        int: 下一个可用的episode编号（从1开始）
    """
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent
    topomaps_dir = project_root / "topomaps" / "images"
    
    episode_num = 1
    while True:
        episode_dir = topomaps_dir / f"episode_{episode_num}"
        if not episode_dir.exists():
            return episode_num
        episode_num += 1

def main():
    # 硬编码的配置参数（不再使用命令行参数）
    SAVE_INTERVAL = 0.7  # 保存间隔（秒）- 0.7秒一帧
    MAX_IMAGES = 200  # 最大图片数量
    MANUAL_MODE = True  # 手动模式（True=手动结束，False=达到最大数量自动结束）
    
    # 自动找到下一个episode编号
    episode_num = find_next_episode_number()
    topomap_name = f"episode_{episode_num}"
    
    print(f"🎬 Starting Topomap Recorder")
    print(f"Episode: {topomap_name}")
    print(f"Interval: {SAVE_INTERVAL}s per frame")
    print(f"Mode: {'Manual (stop when you press q)' if MANUAL_MODE else f'Auto (stop at {MAX_IMAGES} images)'}")
    print("")
    
    # 初始化 ROS2
    rclpy.init()
    
    # 创建录制器
    recorder = OptimizedTopomapRecorder(
        topomap_name=topomap_name,
        save_interval=SAVE_INTERVAL,
        max_images=MAX_IMAGES
    )
    
    # 设置模式
    recorder.manual_mode = MANUAL_MODE
    
    # 启动键盘输入线程
    input_thread = threading.Thread(target=keyboard_input_thread, args=(recorder,))
    input_thread.daemon = True
    input_thread.start()
    
    try:
        print("\n" + "=" * 50)
        print(f"🎬 Optimized Topomap Recorder - {topomap_name}")
        print("=" * 50)
        print("🔗 Waiting for Unity connection...")
        print("Make sure Unity is running and TCP Endpoint is connected")
        print("")
        print(f"📸 Capture interval: {SAVE_INTERVAL}s per frame")
        print(f"🎯 Mode: {'Manual (stop when you reach endpoint)' if recorder.manual_mode else f'Auto (stop at {MAX_IMAGES} images)'}")
        print("")
        print("⚠️  You can only start recording after Unity is connected!")
        print("Press 'c' to check connection status")
        print("Press 's' to start recording (only when connected)")
        print("Use WASD in Unity to drive the robot")
        print("")
        
        rclpy.spin(recorder)
        
    except KeyboardInterrupt:
        pass
    finally:
        recorder.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
