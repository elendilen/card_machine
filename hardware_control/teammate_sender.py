#!/usr/bin/env python3
"""
队友使用脚本 - 发送角度数据到树莓派
增加照片发送功能和结束信号功能
"""

import requests
import json
import base64
import os
from datetime import datetime

# 配置树莓派IP地址 - 请修改为实际IP
RASPBERRY_PI_IP = "192.168.183.170"  # 修改为你的树莓派IP
API_URL = f"http://{RASPBERRY_PI_IP}:5000/api/receive_angles"

class TeammateSender:
    def __init__(self, teammate_url=None):
        self.teammate_url = teammate_url
        self.timeout = 30
    
    def set_teammate_url(self, url):
        """设置队友的URL"""
        self.teammate_url = url
        print(f"队友URL设置为: {self.teammate_url}")
    
    def encode_image_to_base64(self, image_path):
        """将图片编码为base64字符串"""
        try:
            with open(image_path, 'rb') as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                return encoded_string
        except Exception as e:
            print(f"编码图片失败: {e}")
            return None
    
    def send_photo(self, photo_path, rotation_number=None, additional_data=None):
        """发送照片给队友"""
        if not os.path.exists(photo_path):
            print(f"照片文件不存在: {photo_path}")
            return False
        
        try:
            # 编码图片
            image_base64 = self.encode_image_to_base64(photo_path)
            if not image_base64:
                return False
            
            # 准备发送的数据
            data = {
                'image': image_base64,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'filename': os.path.basename(photo_path),
                'rotation_number': rotation_number,
                'sender': 'advance_model_camera'
            }
            
            # 添加额外数据
            if additional_data:
                data.update(additional_data)
            
            # 发送POST请求
            response = requests.post(
                f"{self.teammate_url}/api/receive_photo",
                json=data,
                timeout=self.timeout,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success', False):
                    print(f"照片发送成功: {photo_path} -> {self.teammate_url}")
                    return True
                else:
                    print(f"队友处理照片失败: {result.get('error', '未知错误')}")
                    return False
            else:
                print(f"发送照片失败，状态码: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"网络请求失败: {e}")
            return False
        except Exception as e:
            print(f"发送照片时出错: {e}")
            return False
    
    def send_gesture(self, additional_data=None):
        """发信息队友进行手势识别"""
        try:
            data = {
                'gesture': 'gesture recognize',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'sender': 'advance_model_camera'
            }
            
            # 添加额外数据
            if additional_data:
                data.update(additional_data)
            
            response = requests.post(
                f"{self.teammate_url}/api/recognize_gesture_from_file",
                json=data,
                timeout=self.timeout,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                result = response.json()
                #print(f"[DEBUG] 队友完整响应: {result}")  # 添加调试信息
                if result.get('success', False):
                    print("手势数据发送成功")
                    return True
                else:
                    error_msg = result.get('error', result.get('message', '未知错误'))
                    print(f"队友处理手势数据失败: {error_msg}")
                    return False
            else:
                print(f"发送手势数据失败，状态码: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"网络请求失败: {e}")
            return False
        except Exception as e:
            print(f"发送手势数据时出错: {e}")
            return False
    def send_rotation_status(self, rotation_number, status, photo_path=None):
        """发送旋转状态信息"""
        try:
            data = {
                'rotation_number': rotation_number,
                'status': status,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'sender': 'advance_model_camera'
            }
            
            if photo_path and os.path.exists(photo_path):
                # 如果有照片，一起发送
                return self.send_photo(photo_path, rotation_number, {
                    'rotation_status': status
                })
            else:
                # 只发送状态信息
                response = requests.post(
                    f"{self.teammate_url}/api/receive_status",
                    json=data,
                    timeout=self.timeout,
                    headers={'Content-Type': 'application/json'}
                )
                
                if response.status_code == 200:
                    print(f"状态发送成功: 旋转{rotation_number} - {status}")
                    return True
                else:
                    print(f"发送状态失败，状态码: {response.status_code}")
                    return False
                    
        except Exception as e:
            print(f"发送状态时出错: {e}")
            return False