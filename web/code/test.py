import torchaudio
from backend.speaker_identifier import SpeakerIdentifier
from backend.registration import PlayerRegistrar
import whisper
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write

SAMPLE_FILE = "samples/player2.wav"
TEMP_FILE = "test_temp.wav"
SAMPLERATE = 16000
DURATION = 3  # 录音时长

def record_audio(filename=TEMP_FILE, duration=DURATION):
    print("🎙️ 现在请说话进行实时识别测试...")
    audio = sd.rec(int(SAMPLERATE * duration), samplerate=SAMPLERATE, channels=1, dtype='float32')
    sd.wait()
    write(filename, SAMPLERATE, (audio * 32767).astype(np.int16))
    print("✅ 录音完成")

def main():
    # 初始化
    whisper_model = whisper.load_model("medium")
    speaker_identifier = SpeakerIdentifier()
    registrar = PlayerRegistrar()

    # 注册 player1
    waveform, sr = torchaudio.load(SAMPLE_FILE)
    emb = speaker_identifier.extract_embedding(waveform, sr)
    registrar.register("player1", emb)
    print("✅ 注册 player1 完成")

    # 实时录音
    record_audio()

    # 提取识别文本
    result = whisper_model.transcribe(TEMP_FILE, language="en", task="transcribe")
    text = result["text"].strip()
    print(f"📝 Whisper识别内容：{text}")

    # 提取当前声纹 & 比对
    waveform, sr = torchaudio.load(TEMP_FILE)
    emb = speaker_identifier.extract_embedding(waveform, sr)
    player_id = registrar.match(emb)

    # 输出格式
    if text:
        print(f"\n✅ 识别结果：player{player_id}: ({text})")
    else:
        print("⚠️ 未识别出语音内容")

if __name__ == "__main__":
    main()
