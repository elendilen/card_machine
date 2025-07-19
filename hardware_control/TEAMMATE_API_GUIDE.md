# 队友API调用指南

## 概述
本文档说明如何向旋转拍照系统发送指令。系统运行在 `http://你的IP:5000`，提供以下API接口供队友调用。

## API接口

### 1. 发送角度数据
**接口地址**: `POST http://你的IP:5000/api/receive_angles`

**用途**: 发送四个角度数据到系统（只需要发送一次）

**请求格式**:
```json
{
    "angles": [45, 90, 135, 180]
}
```

**参数说明**:
- `angles`: 数组，包含4个角度值
- 每个角度值必须在0-180度之间
- 角度值可以是整数或小数

**响应示例**:
```json
{
    "success": true,
    "message": "角度数据已接收"
}
```

**Python调用示例**:
```python
import requests

url = "http://你的IP:5000/api/receive_angles"
data = {
    "angles": [45, 90, 135, 180]
}

response = requests.post(url, json=data)
print(response.json())
```

### 2. 发送执行命令
**接口地址**: `POST http://你的IP:5000/api/receive_command`

**用途**: 触发系统执行四个角度的旋转和拍照（每次需要拍照时调用）

**请求格式**:
```json
{
    "command": "拍照指令描述"
}
```

**参数说明**:
- `command`: 字符串，描述本次拍照的目的或指令内容
- 可以是任意文本，如"检查状态"、"记录数据"等

**响应示例**:
```json
{
    "success": true,
    "message": "角度旋转任务完成"
}
```

**Python调用示例**:
```python
import requests

url = "http://你的IP:5000/api/receive_command"
data = {
    "command": "检查当前状态"
}

response = requests.post(url, json=data)
print(response.json())
```

### 3. 查看系统状态
**接口地址**: `GET http://你的IP:5000/api/get_status`

**用途**: 查看当前系统状态和角度数据

**响应示例**:
```json
{
    "angles": [45, 90, 135, 180],
    "timestamp": "2025-07-15 14:30:25",
    "status": "等待后续命令"
}
```

**Python调用示例**:
```python
import requests

url = "http://你的IP:5000/api/get_status"
response = requests.get(url)
print(response.json())
```

## 使用流程

### 步骤1: 等待初始旋转完成
系统会自动进行90次完整旋转拍照，无需队友操作。

### 步骤2: 发送角度数据（仅一次）
在系统完成初始旋转后，发送四个角度数据：
```python
import requests

# 发送角度数据
url = "http://你的IP:5000/api/receive_angles"
angles_data = {
    "angles": [45, 90, 135, 180]  # 替换为实际角度
}
response = requests.post(url, json=angles_data)
print("角度数据发送结果:", response.json())
```

### 步骤3: 发送执行命令（按需多次）
每次需要系统拍照时，发送执行命令：
```python
import requests

# 发送执行命令
url = "http://你的IP:5000/api/receive_command"
command_data = {
    "command": "第一次检查"  # 可以是任意描述
}
response = requests.post(url, json=command_data)
print("命令执行结果:", response.json())
```

## 完整示例脚本

```python
import requests
import time

# 系统IP地址
BASE_URL = "http://你的IP:5000"

def send_angles(angles):
    """发送角度数据"""
    url = f"{BASE_URL}/api/receive_angles"
    data = {"angles": angles}
    
    try:
        response = requests.post(url, json=data)
        result = response.json()
        print(f"发送角度数据: {result}")
        return result.get('success', False)
    except Exception as e:
        print(f"发送角度数据失败: {e}")
        return False

def send_command(command):
    """发送执行命令"""
    url = f"{BASE_URL}/api/receive_command"
    data = {"command": command}
    
    try:
        response = requests.post(url, json=data)
        result = response.json()
        print(f"发送命令结果: {result}")
        return result.get('success', False)
    except Exception as e:
        print(f"发送命令失败: {e}")
        return False

def get_status():
    """获取系统状态"""
    url = f"{BASE_URL}/api/get_status"
    
    try:
        response = requests.get(url)
        result = response.json()
        print(f"系统状态: {result}")
        return result
    except Exception as e:
        print(f"获取状态失败: {e}")
        return None

# 使用示例
if __name__ == "__main__":
    # 1. 等待系统完成初始旋转（手动确认）
    input("请确认系统已完成初始90次旋转，按回车继续...")
    
    # 2. 发送角度数据（只需要一次）
    angles = [45, 90, 135, 180]  # 替换为实际需要的角度
    if send_angles(angles):
        print("角度数据发送成功！")
        
        # 3. 等待一段时间，然后发送第一个命令
        time.sleep(2)
        
        # 4. 发送执行命令
        commands = [
            "第一次检查",
            "第二次检查", 
            "第三次检查"
        ]
        
        for i, cmd in enumerate(commands, 1):
            print(f"\n--- 执行第{i}次命令 ---")
            if send_command(cmd):
                print(f"命令 '{cmd}' 执行成功！")
                # 可以根据需要添加延时
                if i < len(commands):
                    time.sleep(5)  # 等待5秒再发送下一个命令
            else:
                print(f"命令 '{cmd}' 执行失败！")
                break
    else:
        print("角度数据发送失败！")
```

## 注意事项

1. **IP地址**: 请将 `你的IP` 替换为实际的系统IP地址
2. **角度数据**: 只需要发送一次，系统会保存这些角度用于后续所有拍照
3. **执行时机**: 确保系统完成初始90次旋转后再发送角度数据
4. **命令间隔**: 建议在命令之间添加适当延时，避免系统繁忙
5. **错误处理**: 请检查每次API调用的返回结果，确保操作成功

## 故障排除

### 常见错误及解决方法

1. **"没有存储的角度数据"**
   - 先调用 `/api/receive_angles` 发送角度数据

2. **"连接拒绝"**
   - 检查IP地址是否正确
   - 确认系统是否正在运行

3. **"角度无效"**
   - 确保角度值在0-180度之间
   - 确保发送4个角度值

4. **"旋转超时"**
   - 系统硬件可能有问题，请联系操作员
