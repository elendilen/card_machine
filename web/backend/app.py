
from fastapi import FastAPI, File, UploadFile,Response,WebSocket,Request
from fastapi.responses import JSONResponse,FileResponse,StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import torchaudio
import whisper
import tempfile
import os
import cv2
import threading,time
import subprocess
import uuid
from gesture import GestureRecognizer
import tensorflow as tf
import librosa
import numpy as np
import torch
import pickle
from insightface.app import FaceAnalysis
import requests
import base64
from datetime import datetime
from flask import request, jsonify



# 添加 ffmpeg所在路径到环境变量
ffmpeg_path = r"C:\ffmpeg-7.1.1-essentials_build\ffmpeg-7.1.1-essentials_build\bin"
os.environ["PATH"] += os.pathsep + ffmpeg_path


app = FastAPI()

app.mount("/static", StaticFiles(directory="../front/static"), name="static")

torchaudio.set_audio_backend("soundfile")
@app.get("/")
async def index():
    return FileResponse("../front/static/index2.html")

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化模型
whisper_model = whisper.load_model("medium")
gesture_recognizer = GestureRecognizer()
cap = cv2.VideoCapture(0)

MODEL_PATH="speaker_cnn4.h5"
model = tf.keras.models.load_model(MODEL_PATH)

label_map = {0: "SHEN", 1: "TIAN", 2: "WAN", 3: "WANG"}

SR = 16000
N_MELS = 64
MAX_LEN = 320

# ===== 提取 Mel 特征函数 =====
def extract_mel(y, sr=16000, n_mels=64, max_len=320):
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    # Padding
    if log_mel.shape[1] > max_len:
        log_mel = log_mel[:, :max_len]
    else:
        pad_width = max_len - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode='constant')
    return log_mel

camera_active = False
lock = threading.Lock()

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        with lock:
            if not camera_active:
                break
        ret, frame = cap.read()

        if not ret:
            break
        frame = cv2.flip(frame, 0)  # 上下颠倒
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
    cap.release()

@app.post("/api/start_stream")
def start_stream():
    global camera_active
    with lock:
        camera_active = True
    return {"status": "started"}

@app.get("/api/video_feed")
def video_feed():
    return Response(cap = cv2.VideoCapture(0), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/api/stop_stream")
def stop_stream():
    global camera_active
    with lock:
        camera_active = False
    return {"status": "stopped"}



@app.websocket("/ws/gesture")
async def gesture_ws(websocket: WebSocket):
    await websocket.accept()
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # 旋转 180 度（如果摄像头倒装）
        frame = cv2.rotate(frame, cv2.ROTATE_180)

        # 预测手势
        prediction = gesture_recognizer.predict(frame)

        if prediction:
            await websocket.send_json({"prediction": prediction})


dialogue_history = []
"""
@app.post("/api/recognize/")
async def recognize_audio(file: UploadFile = File(...)):
    try:
        # 临时文件处理
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp_in:
            tmp_in.write(await file.read())
            tmp_in_path = tmp_in.name

        tmp_out_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}.wav")

        # ffmpeg 转码
        subprocess.run([
            "ffmpeg", "-y", "-i", tmp_in_path, "-ar", str(SR), "-ac", "1", tmp_out_path
        ], check=True)

        # Whisper 语音转文字
        result = whisper_model.transcribe(tmp_out_path, language="en", task="transcribe")
        text = result.get("text", "").strip()

        # CNN 声纹识别
        y, _ = librosa.load(tmp_out_path, sr=SR, mono=True, dtype=np.float32)
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))  # 归一化
        y, _ = librosa.effects.trim(y)
        feature = extract_mel(y, sr=SR, n_mels=N_MELS, max_len=MAX_LEN)
        feature = feature[np.newaxis, ..., np.newaxis]  # (1, 64, 320, 1)

        pred = model.predict(feature)
        speaker_id = np.argmax(pred)
        confidence = float(pred[0][speaker_id])
        speaker_name = label_map.get(speaker_id, "Unknown")

        dialogue_history.append({
            "player": speaker_name,
            "text": text
        })
        # 清理临时文件
        os.remove(tmp_in_path)
        os.remove(tmp_out_path)

        # 返回前端需要的结构
        return {
            "player": speaker_name,
            "text": text,  #保留文本
            "confidence": confidence
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
"""

from ecapa_identifier import ECAPAIdentifier  # 放在 app.py 顶部导入
identifier = ECAPAIdentifier()               # 放在全局初始化区域

@app.post("/api/recognize/")
async def recognize_audio(file: UploadFile = File(...)):
    try:
        # 临时文件处理
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp_in:
            tmp_in.write(await file.read())
            tmp_in_path = tmp_in.name

        tmp_out_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}.wav")

        # ffmpeg 转码为 wav
        subprocess.run([
            "ffmpeg", "-y", "-i", tmp_in_path, "-ar", str(SR), "-ac", "1", tmp_out_path
        ], check=True)

        # Whisper 语音转文字（识别内容）
        result = whisper_model.transcribe(tmp_out_path, language="en", task="transcribe")
        text = result.get("text", "").strip()

        # ✅ 替换为 ECAPA 声纹识别
        speaker_name, confidence = identifier.identify(tmp_out_path)

        # 记录到对话历史中
        dialogue_history.append({
            "player": speaker_name,
            "text": text
        })

        # 清理临时文件
        os.remove(tmp_in_path)
        os.remove(tmp_out_path)

        return {
            "player": speaker_name,
            "text": text,
            "confidence": float(confidence)
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
 
    
from gameManager import GameManager
from pydantic import BaseModel

game=GameManager()

# 数据模型
class InitRequest(BaseModel):
    num_players: int

class GestureInput(BaseModel):
    gesture: str

class RoleUpdateInput(BaseModel):
    player_id: int
    
# 初始化游戏
@app.post("/init_game")
def init_game(data: InitRequest):
    game.init_players(data.num_players)
    return {"message": f"初始化 {data.num_players} 名玩家成功", "state": game.state}

# 更新角色（通过人脸识别）
@app.post("/update_role")
def update_role(data: RoleUpdateInput):
    success = game.update_role(data.player_id)
    if success:
        return {"message": f"玩家 {data.player_id} 被标记为 {game.state['current_role']}"}
    return {"message": "当前阶段无法分配角色"}

# 获取当前状态
@app.get("/get_game_state")
def get_game_state():
    return game.state

#提交手势
class GestureInput(BaseModel):
    gesture: str

@app.post("/submit_gesture")
async def submit_gesture(data: GestureInput):
    gesture = data.gesture
    if gesture == "None":
        print("🕳️ 手势为空，跳过该玩家操作")
        return {"message": "无手势输入，跳过操作", "skip": True}
    print(f"收到手势: {gesture}")
    
    game.handle_action(gesture)  # handle_action 里已经 next_role 了
    
    print("当前玩家状态：")
    for pid, info in game.state["players"].items():
        print(f"玩家 {pid} - 角色: {info['role']}, 存活: {info['alive']}")

    return {
        "message": f"手势 {gesture} 已处理",
        "current_role": game.state["current_role"],
        "players": game.players
    }



from pydantic import BaseModel
from typing import Optional

class VoteResultInput(BaseModel):
    eliminated: Optional[int]


@app.post("/submit_vote")
def submit_vote(data: VoteResultInput):
    print("🟡 收到来自 5000 的投票结果请求")
    print("📩 请求体内容：", data)

    eliminated = data.eliminated

    if eliminated is not None:
        print(f"🔧 尝试放逐玩家 {eliminated}")
        success, msg = game.eliminate_player(eliminated)
        # 更新当前提示信息，方便前端刷新
        game.state["current_prompt"] = f"投票结果：{eliminated}号被放逐"

        print(f"玩家 {eliminated} 被成功放逐") if success else print("❌ 放逐失败")
    else:
        # ✅ 平票情况：不放逐任何人
        game.state["eliminated_today"] = None
        game.state["current_prompt"] = "平票，无人被放逐"
        success = True
        msg = "平票，无人被处决"
        print("⚖️ 平票，无人被放逐")

    print("📊 玩家生存状态更新：")
    for pid, info in game.state["players"].items():
        print(f"玩家 {pid} - 存活: {info['alive']}")

    # ✅ 统一推进逻辑（不管有没有放逐成功）
    game.check_game_end()
    
    game.advance_phase()
    print("当前阶段：", game.state["phase"])


    return {"success": success, "message": msg, "eliminated": eliminated}

    
def build_prompt(dialogue: list[dict]) -> str:
    prompt = "【狼人杀发言记录】\n"
    for idx, d in enumerate(dialogue, 1):
        prompt += f"玩家{idx}（{d['player']}）：{d['text']}\n"
    prompt += "\n请根据上面的发言，分析每位玩家的可能身份，并说明理由。最后需要将回答转换成英文。"
    return prompt

@app.post("/api/deduce/")
async def llm_deduction():
    api_key = "sk-866c8c007f1a4440a2158962c6f28771"  # ✅ 正式部署建议改为环境变量
    url = "https://api.deepseek.com/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    prompt = build_prompt(dialogue_history)

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "system",
                "content": "你是一个狼人杀裁判，根据发言内容推理身份，并且需要给出相应的理由，请将回答转换成英文。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": False
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return {
            "prompt": prompt,
            "analysis": data["choices"][0]["message"]["content"]
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/advance_phase")
def advance_phase():
    game.advance_phase()
    return {
        "message": "阶段已推进",
        "current_phase": game.state["phase"],
        "current_role": game.state["current_role"],
        "round": game.state["round"]
    }




