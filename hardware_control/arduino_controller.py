#!/usr/bin/env python3
"""
Arduino控制器模块
负责与Arduino板子的串口通信
"""

import serial
import time

class ArduinoController:
    def __init__(self):
        self.ser = None
        self.port = None
        self.baudrate = 9600
        self.timeout = 1
        self._connect()
    
    
    def _connect(self):
        """连接到Arduino"""
        self.port = "/dev/ttyACM0"  # 根据实际情况修改端口  
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            print(f"成功连接到Arduino: {self.port} (波特率: {self.baudrate})")
            time.sleep(2)  # 等待Arduino初始化
        except serial.SerialException as e:
            print(f"连接Arduino失败: {e}")
            self.ser = None
            
    def is_connected(self):
        """检查是否已连接"""
        return self.ser is not None and self.ser.is_open
    
    def send_rotate(self):
        """发送旋转命令到Arduino"""
        if not self.is_connected():
            print("Arduino未连接,无法发送旋转命令")
            return False
        
        try:
            command = "r"  # 假设'r'是旋转命令
            self.ser.write(command.encode('utf-8'))
            print("发送旋转命令到Arduino")
            return True
        except Exception as e:
            print(f"发送旋转命令时出错: {e}")
            return False

    def return_start(self):
        """发送开始信号到Arduino"""
        if not self.is_connected():
            print("Arduino未连接,无法发送开始信号")
            return False
        
        try:
            command = "q"  # 假设'q'是开始命令
            self.ser.write(command.encode('utf-8'))
            print("发送开始信号到Arduino")
            return True
        except Exception as e:
            print(f"发送开始信号时出错: {e}")
            return False
            
    def recieve_end(self, timeout=30):
        """接收Arduino发送的结束信号，持续等待直到收到END信号"""
        if not self.is_connected():
            print("Arduino未连接,无法接收结束信号")
            return False
        
        start_time = time.time()
        print("等待Arduino发送END信号...")
        
        while True:
            try:
                # 检查是否超时
                if time.time() - start_time > timeout:
                    print(f"等待END信号超时({timeout}秒)")
                    return False
                
                # 设置较短的超时时间来检查数据
                if self.ser.in_waiting > 0:
                    response = self.ser.readline().decode('utf-8').strip()
                    print(response)
                    if response == "END":
                        print("接收到Arduino的结束信号")
                        return True
                    elif response:  # 如果收到其他非空信号
                        print(f"接收到信号: {response}")
                
                # 短暂休眠避免CPU过度占用
                #time.sleep(0.1)
                
            except Exception as e:
                print(f"接收数据时出错: {e}")
                return False

    def send_angles(self, angles):
        """发送角度数据到Arduino"""       
        try:
            print(f"发送角度到Arduino: {angles}")
            
            for i, angle in enumerate(angles):
                # 发送角度值
                command = f"s {angle}"
                self.ser.write(command.encode('utf-8'))
                print(f"  发送角度{i+1}: {angle}°")
                #time.sleep(0.5)  # 短暂延时确保数据发送完成
            
            print("所有角度发送完成")
            return True
            
        except Exception as e:
            print(f"发送数据到Arduino时出错: {e}")
            return False
    
    def send_single_angle(self, angle):
        """发送单个角度到Arduino并等待完成"""
        if not self.is_connected():
            print("Arduino未连接,无法发送角度")
            return False
        
        try:
            command = f"s {angle}"
            self.ser.write(command.encode('utf-8'))
            print(f"发送单个角度: {angle}°")
            return True
        except Exception as e:
            print(f"发送单个角度时出错: {e}")
            return False
    
    def send_single_angle_without_card(self, angle):
        """发送单个角度到Arduino并等待完成"""
        if not self.is_connected():
            print("Arduino未连接,无法发送角度")
            return False

        try:
            command = f"R {angle}"
            self.ser.write(command.encode('utf-8'))
            print(f"发送单个角度(无卡片): {angle}°")
            return True
        except Exception as e:
            print(f"发送单个角度(无卡片)时出错: {e}")
            return False
            
    def close(self):
        """关闭串口连接"""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("Arduino连接已关闭")
    
    def __del__(self):
        """析构函数，确保串口正确关闭"""
        self.close()
