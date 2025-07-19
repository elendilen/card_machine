# ecapa_identifier.py
import pickle
import sqlite3
import torch
import torchaudio
import numpy as np
from speechbrain.pretrained import EncoderClassifier
from sklearn.metrics.pairwise import cosine_similarity

class ECAPAIdentifier:
    def __init__(self, db_path="voiceprints.db", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.db_path = db_path
        self.device = device
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": self.device}
        )

    def identify(self, audio_path):
        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)

        with torch.no_grad():
            emb = self.model.encode_batch(waveform.to(self.device)).squeeze(0).squeeze(0)
            emb = emb.cpu().numpy()

        return self._find_best_match(emb)

    def _find_best_match(self, query_embedding):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT player, embedding FROM voiceprints")
        rows = c.fetchall()
        conn.close()

        best_match = "Unknown"
        best_score = -1

        for player, blob in rows:
            db_emb = pickle.loads(blob)
            score = cosine_similarity([query_embedding], [db_emb])[0][0]
            if score > best_score:
                best_score = score
                best_match = player

        return best_match, best_score
