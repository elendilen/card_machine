import base64
import os
import requests
import cv2
import numpy as np
from datetime import datetime
from fastapi import FastAPI, Request,UploadFile,File,Query
from fastapi.responses import JSONResponse
from insightface.app import FaceAnalysis
from fastapi.middleware.cors import CORSMiddleware
import torch
from trainCNN import GestureCNN  # 你的 CNN 类
from data import normalize_keypoints
import mediapipe as mp
import threading
import pickle
from collections import defaultdict
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 或者指定 ["http://127.0.0.1:8888"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PHOTOS_DIR = "./facephotos"
os.makedirs(PHOTOS_DIR, exist_ok=True)

# 假设这些全局变量和函数已定义
face_app = FaceAnalysis()
face_app.prepare(ctx_id=0)  # InsightFace 初始化
threshold = 0.4

def cosine_similarity(emb1, emb2):
    emb1 = emb1 / np.linalg.norm(emb1)
    emb2 = emb2 / np.linalg.norm(emb2)
    return float(np.dot(emb1, emb2))

# 加载数据库
with open("embeddings_db.pkl", "rb") as f:
    embedding_db = pickle.load(f)

def recognize_face_from_image(frame):
    faces = face_app.get(frame)
    if not faces:
        return {"player": "NoFace", "similarity": 0.0, "message": "未检测到人脸"}

    best_id, best_sim = "Unknown", 0.0
    for face in faces:
        emb = face.embedding
        for pid, emb_list in embedding_db.items():
            for saved_emb in emb_list:
                sim = cosine_similarity(emb, saved_emb)
                if sim > threshold and sim > best_sim:
                    best_sim = sim
                    best_id = pid

    return {
        "player": best_id,
        "similarity": round(best_sim, 3),
        "message": "识别成功" if best_id != "Unknown" else "未匹配到注册玩家"
    }

# 新增：自动将识别出的玩家绑定为当前角色（通过8888端口）
def auto_update_role_to_game_server(player_name):
    name_to_id_map = {
        "SHEN": 1,
        "TIAN": 2,
        "WAN": 3,
        "WANG": 4
    }

    if player_name not in name_to_id_map:
        print(f"[WARN] 未知玩家名：{player_name}")
        return

    player_id = name_to_id_map[player_name]
    game_server_url = "http://127.0.0.1:8888/update_role"
    try:
        resp = requests.post(game_server_url, json={"player_id": player_id}, timeout=3)
        print(f"[INFO] 成功更新玩家 {player_id}（{player_name}）的角色身份，响应：{resp.json()}")
    except Exception as e:
        print(f"[ERROR] 无法连接游戏服务进行身份绑定：{e}")


@app.post("/api/receive_photo")
async def receive_and_recognize(request: Request):
    try:
        data = await request.json()
        if not data or 'image' not in data:
            return JSONResponse(content={"success": False, "error": "缺少图片数据"}, status_code=400)

        # 解析 Base64 图片
        image_data = base64.b64decode(data['image'])
        filename = data.get('filename', f'photo_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg')
        photo_path = os.path.join(PHOTOS_DIR, filename)

        # 保存图片
        with open(photo_path, 'wb') as f:
            f.write(image_data)

        # 识别人脸
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return JSONResponse(content={"success": False, "error": "无法解析图片"}, status_code=400)

        frame = cv2.rotate(frame, cv2.ROTATE_180)
        recognition_result = recognize_face_from_image(frame)
        print(f"[INFO] 图片 {filename} 识别结果: {recognition_result}")

        # 发送身份
        if recognition_result["player"] not in ["Unknown", "NoFace"]:
            auto_update_role_to_game_server(recognition_result["player"])

        return JSONResponse(content={
            "success": True,
            "filename": filename,
            "recognition": recognition_result
        })

    except Exception as e:
        print(f"[ERROR] 接收照片失败: {e}")
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)

# 硬件的 API 基础地址
HARDWARE_BASE_URL = "http://192.168.183.170:5000"  # 替换成硬件实际 IP

def send_command_to_hardware(command: str):
    """
    向硬件发送执行命令
    """
    url = f"{HARDWARE_BASE_URL}/api/receive_command"
    data = {"command": command}

    try:
        response = requests.post(url, json=data, timeout=5)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, {"error": f"硬件返回状态码: {response.status_code}", "details": response.text}
    except Exception as e:
        return False, {"error": f"发送命令失败: {str(e)}"}



@app.post("/send_command")
async def send_command(command: str):
    """
    FastAPI 接口：向硬件发送命令
    调用示例: POST /send_command?command=rotate_90_and_capture
    """
    success, result = send_command_to_hardware(command)
    return JSONResponse(content={"success": success, "message": "命令已发送" if success else "命令发送失败"})

#gesture代码

GESTURE_PHOTOS_DIR = "./gesturephotos"
os.makedirs(GESTURE_PHOTOS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GestureCNN().to(device)
model.load_state_dict(torch.load("best_model2.pt", map_location=device))
model.eval()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
label_map = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5", 5: "6", 6: "OK", 7: "X"}

# 全局变量
current_frame = None
frame_lock = threading.Lock()
last_prediction = "None"
stable_prediction = "None"
predict_count = 0
GAME_SERVER_URL = "http://127.0.0.1:8888/submit_gesture"

CMD_FILES = ["cmd_1.jpg", "cmd_2.jpg", "cmd_3.jpg", "cmd_4.jpg"]


@app.post("/api/recognize_gesture_from_file")
async def recognize_gesture_from_file():
    global stable_prediction
    results = []

    for filename in CMD_FILES:
        photo_path = os.path.join(PHOTOS_DIR, filename)
        print(f"🎯 处理图像: {filename}")
        if not os.path.exists(photo_path):
            print(f"🚫 图像文件不存在: {filename}")
            results.append({"filename": filename, "success": False, "error": f"文件不存在: {filename}"})
            continue

        try:
            frame = cv2.imread(photo_path)
            if frame is None:
                results.append({"filename": filename, "success": False, "error": "无法加载图像"})
                continue

            frame = cv2.rotate(frame, cv2.ROTATE_180)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(frame_rgb)

            if not result.multi_hand_landmarks:
                try:
                    response = requests.post(GAME_SERVER_URL, json={"gesture": "None"})
                    print(f"📤 没有检测到手势，发送 'None'，响应: {response.status_code}")
                except Exception as e:
                    print(f"❌ 无法通知游戏服务无手势: {e}")
                results.append({"filename": filename, "success": True, "gesture": None, "message": "未检测到手势"})
                continue

            # 下面代码是你原封不动复制过来的
            gesture_found = False
            for lm in result.multi_hand_landmarks:
                keypoints = [(p.x, p.y) for p in lm.landmark]
                if len(keypoints) == 21:
                    kp_norm = normalize_keypoints(np.array(keypoints).flatten())
                    x = torch.tensor(kp_norm, dtype=torch.float32).reshape(1, 1, 6, 7).to(device)
                    with torch.no_grad():
                        pred = model(x)
                        probs = torch.softmax(pred, dim=1).cpu().numpy()[0]
                        pred_idx = int(np.argmax(probs))
                        pred_label = label_map[pred_idx]
                        confidence = float(probs[pred_idx])

                    stable_prediction = pred_label

                    try:
                        response = requests.post(GAME_SERVER_URL, json={"gesture": pred_label})
                        game_response = response.json()
                        print(f"✅ 识别结果: {pred_label}, 置信度: {confidence:.3f}")
                        print(f"📤 向游戏服务发送: {pred_label}，响应: {game_response}")
                        submit_vote_by_filename(filename, pred_label)

                    except Exception as e:
                        game_response = {"error": f"无法连接游戏服务: {e}"}

                    results.append({
                        "filename": filename,
                        "success": True,
                        "gesture": pred_label,
                        "confidence": confidence,
                        "game_update": game_response
                    })
                    gesture_found = True
                    break  # 这里保持你原逻辑中对多手的处理方式

            if not gesture_found:
                try:
                    response = requests.post(GAME_SERVER_URL, json={"gesture": "None"})
                    print(f"📤 没有识别出有效手势，发送 'None'，响应: {response.status_code}")
                except Exception as e:
                    print(f"❌ 无法通知游戏服务手势失败: {e}")

                results.append({"filename": filename, "success": True, "gesture": None, "message": "未识别有效手势"})

        except Exception as e:
            results.append({"filename": filename, "success": False, "error": str(e)})

    has_success = any(r.get("success", False) and r.get("gesture") for r in results)
    print("📝 最终识别结果汇总：", results)

    # 🔚 判断是否满足“白天投票”条件后再结算投票
    state = get_game_state()
    if state.get("phase") == "day":
        try:
            finalize_result = finalize_vote()
            print("✅ 自动触发投票结算，响应：", finalize_result)
        except Exception as e:
            print("❌ 投票结算失败：", e)
    else:
        print(f"⚠️ 当前阶段为 {state.get('phase')} / {state.get('current_role')}，跳过投票结算")

    return JSONResponse(content={
        "success": has_success,
        "results": results,
        "message": "手势识别完成" if has_success else "未识别到有效手势"
    })


@app.get("/get_last_prediction")
def get_last_prediction():
    return {"gesture": stable_prediction}

import requests

name_to_id = {"SHEN": 1, "TIAN": 2, "WAN": 3, "WANG": 4}
id_to_name = {v: k for k, v in name_to_id.items()}


def get_game_state():
    try:
        resp = requests.get("http://127.0.0.1:8888/get_game_state")
        return resp.json()
    except:
        return {}

# 全局投票计数器
vote_counter = defaultdict(int)

def record_vote(gesture):
    state = get_game_state()
    if state.get("phase") != "day":
        print("[跳过] 当前不是白天，忽略手势投票")
        return

    if gesture and gesture.isdigit():
        vote_counter[int(gesture)] += 1
        print(f"[记录投票] 玩家 {gesture} 获得一票")
    else:
        print("[跳过] 非数字手势或 None")


def finalize_vote():
    if not vote_counter:
        return {"message": "没有有效投票，平票处理", "eliminated": None}

    # 找出最高票
    max_votes = max(vote_counter.values())
    candidates = [pid for pid, count in vote_counter.items() if count == max_votes]

    if len(candidates) == 1:
        eliminated = candidates[0]
        message = f"玩家 {eliminated} 被处决"
    else:
        eliminated = None
        message = f"平票，无人被处决，候选人: {candidates}"

    # 通知游戏服务
    try:
        res = requests.post("http://127.0.0.1:8888/submit_vote", json={
            "eliminated": eliminated
        })
        game_response = res.json()
        print("成功发送！！！")
    except Exception as e:
        game_response = {"error": f"无法连接游戏服务器: {e}"}

    # 清空票数
    vote_counter.clear()

    return {
        "message": message,
        "eliminated": eliminated,
        "game_response": game_response
    }


def submit_vote_by_filename(filename, gesture):
    if gesture is None:
        print(f"[投票跳过] {filename} 未检测到手势，视为弃票")
        return

    print(f"[✓] 文件 {filename} 识别为投给 {gesture}")
    record_vote(gesture)
