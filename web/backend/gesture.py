import cv2
import torch
import numpy as np
import mediapipe as mp
from gestureCode.trainCNN import GestureCNN
from gestureCode.data import normalize_keypoints

class GestureRecognizer:
    def __init__(self, model_path="gestureCode/checkpoints/best_model2.pt", device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GestureCNN().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.label_map_inv = {0: '1', 1: '2', 2: '3', 3: '4', 4: '5', 5: '6', 6: 'OK', 7: 'X'}
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False,
                                         max_num_hands=1,
                                         min_detection_confidence=0.5,
                                         min_tracking_confidence=0.5)

    def predict(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(frame_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                keypoints = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
                if len(keypoints) != 21:
                    continue

                kp_norm = normalize_keypoints(np.array(keypoints).flatten())
                x = torch.tensor(kp_norm, dtype=torch.float32).reshape(1, 1, 6, 7).to(self.device)

                with torch.no_grad():
                    output = self.model(x)
                    pred = torch.argmax(output, dim=1).item()
                    return self.label_map_inv[pred]

        return None  # 没检测到手势