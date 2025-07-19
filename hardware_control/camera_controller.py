#!/usr/bin/env python3
"""
相机控制器模块
负责拍照功能
"""

from picamera2 import Picamera2
import os
from datetime import datetime
import time
import io
from threading import Lock
import cv2
import numpy as np

class CameraController:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.picam2 = None
        self.photos_dir = "photos"
        self.streaming = False
        self.frame_lock = Lock()
        self.latest_frame = None
        self._ensure_photos_dir()
    
    def _ensure_photos_dir(self):
        """确保照片目录存在"""
        if not os.path.exists(self.photos_dir):
            os.makedirs(self.photos_dir)
            print(f"创建照片目录: {self.photos_dir}")
    
    def initialize_camera(self):
        """初始化摄像头"""
        # 如果相机已经初始化，先释放
        if self.picam2:
            self.release_camera()
            
        try:
            print("正在初始化相机...")
            self.picam2 = Picamera2()
            
            # 配置相机
            config = self.picam2.create_still_configuration(
                main={"size": (1920, 1080)},  # 高分辨率拍照
                lores={"size": (640, 480)},   # 低分辨率预览
                display="lores"
            )
            self.picam2.configure(config)
            
            # 启动相机
            self.picam2.start()
            time.sleep(2)  # 等待相机稳定
            
            print(f"PiCamera2 初始化成功")
            return True
        except Exception as e:
            print(f"初始化相机失败: {e}")
            return False
    
    def take_photo(self, photo_name=None):
        """拍照并保存"""
        if not self.picam2:
            if not self.initialize_camera():
                return None
        
        try:
            # 生成文件名
            if not photo_name:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                photo_name = f"photo_{timestamp}.jpg"
            
            photo_path = os.path.join(self.photos_dir, photo_name)
            
            # 拍照并保存
            self.picam2.capture_file(photo_path)
            print(f"照片已保存: {photo_path}")
            return photo_path
                
        except Exception as e:
            print(f"拍照时出错: {e}")
            return None
    
    def take_rotation_photo(self, rotation_number):
        """为特定旋转编号拍照"""
        photo_name = f"rotation_{rotation_number:03d}.jpg"
        return self.take_photo(photo_name)
    
    def take_command_photo(self, command):
        """为特定命令拍照"""
        photo_name = f"cmd_{command}.jpg"
        return self.take_photo(photo_name)

    def start_streaming(self):
        """开始视频流"""
        # 停止任何现有的流
        self.stop_streaming()
        
        if not self.picam2:
            if not self.initialize_camera():
                print("无法初始化相机用于视频流")
                return False
        
        try:
            self.streaming = True
            print("视频流已启动")
            return True
        except Exception as e:
            print(f"启动视频流失败: {e}")
            self.streaming = False
            return False
    
    def stop_streaming(self):
        """停止视频流"""
        self.streaming = False
        print("视频流已停止")
    
    def get_frame(self):
        """获取当前帧（JPEG格式）"""
        if not self.picam2 or not self.streaming:
            return None
        
        try:
            # 捕获当前帧
            frame = self.picam2.capture_array()
            
            # 转换为BGR格式（OpenCV标准）
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
            
            # 编码为JPEG
            ret, buffer = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ret:
                return buffer.tobytes()
            else:
                return None
                
        except Exception as e:
            print(f"获取帧失败: {e}")
            return None
    
    def generate_frames(self):
        """生成视频流帧（用于Flask streaming）"""
        while self.streaming:
            frame = self.get_frame()
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.1)  # 控制帧率
    
    def release_camera(self):
        """释放摄像头资源"""
        self.stop_streaming()  # 先停止视频流
        if self.picam2:
            try:
                print("正在释放相机资源...")
                self.picam2.stop()
                self.picam2.close()
                print("相机资源已释放")
            except Exception as e:
                print(f"释放相机资源时出错: {e}")
            finally:
                self.picam2 = None
                time.sleep(1)  # 给系统一点时间释放资源
    
    def __del__(self):
        """析构函数"""
        self.release_camera()
    
    def is_camera_ready(self):
        """检查相机是否准备就绪"""
        return self.picam2 is not None

    def get_camera_status(self):
        """获取相机状态信息"""
        return {
            'initialized': self.picam2 is not None,
            'streaming': self.streaming,
            'photos_dir': self.photos_dir
        }
