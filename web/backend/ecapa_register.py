# ecapa_register.py
import os
import sqlite3
import pickle
import numpy as np
from speechbrain.pretrained import EncoderClassifier
import torchaudio
from sklearn.metrics.pairwise import cosine_similarity
import torch

class ECAPARegistrar:
    def __init__(self, db_path="voiceprints.db", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.db_path = db_path
        self.device = device
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": self.device}
        )
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS voiceprints (
                        player TEXT PRIMARY KEY,
                        embedding BLOB
                    )""")
        conn.commit()
        conn.close()

    def register_player(self, player_name, folder_path):
        embeddings = []

        for file in os.listdir(folder_path):
            if file.startswith(player_name) and file.endswith(".wav"):
                wav_path = os.path.join(folder_path, file)
                waveform, sr = torchaudio.load(wav_path)
                if sr != 16000:
                    waveform = torchaudio.functional.resample(waveform, sr, 16000)

                with torch.no_grad():
                    emb = self.model.encode_batch(waveform.to(self.device)).squeeze(0).squeeze(0)
                    vec = emb.cpu().numpy()
                    if vec.shape != (192,):
                        print(f"[⚠️] {file} 嵌入维度异常: {vec.shape}, 已跳过")
                        continue
                    embeddings.append(vec)

        if not embeddings:
            print(f"❌ 未找到 {player_name} 的音频文件")
            return

        avg_embedding = np.mean(embeddings, axis=0)
        self._save_to_db(player_name, avg_embedding)
        print(f"[✓] 已注册玩家 {player_name}，嵌入维度：{avg_embedding.shape}")

    def _save_to_db(self, player, embedding):
        blob = pickle.dumps(embedding)
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("INSERT OR REPLACE INTO voiceprints (player, embedding) VALUES (?, ?)", (player, blob))
        conn.commit()
        conn.close()
