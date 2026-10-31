# -*- coding: utf-8 -*-
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import pickle
import pandas as pd
import numpy as np
import os
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ========== โหลดโมเดล AI ==========
right_model, left_model = None, None

if os.path.exists('right_hand_model.pkl'):
    with open('right_hand_model.pkl', 'rb') as f:
        right_model = pickle.load(f)
    print("[OK] โหลดโมเดลมือขวา (ทิศทาง)")

if os.path.exists('left_hand_model.pkl'):
    with open('left_hand_model.pkl', 'rb') as f:
        left_model = pickle.load(f)
    print("[OK] โหลดโมเดลมือซ้าย (หัวดูด)")

if not right_model and not left_model:
    print("[ERROR] ไม่พบโมเดล! รัน train_model.py ก่อน")
    sys.exit(1)

# ========== MediaPipe ==========
MODEL_PATH = 'hand_landmarker.task'
if not os.path.exists(MODEL_PATH):
    print(f"[ERROR] ไม่พบ {MODEL_PATH}")
    sys.exit(1)

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    running_mode=vision.RunningMode.IMAGE,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
landmarker = vision.HandLandmarker.create_from_options(options)

# ========== สี ==========
DIR_COLORS = {
    "FORWARD": (0, 255, 0), "BACKWARD": (0, 165, 255),
    "LEFT": (255, 255, 0), "RIGHT": (255, 0, 255),
    "UP": (0, 255, 255), "DOWN": (255, 100, 0),
}
ACT_COLORS = {"GRAB": (0, 200, 255), "RELEASE": (255, 200, 0)}

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17),
]


def draw_landmarks(frame, landmarks, color):
    h, w, _ = frame.shape
    for s, e in HAND_CONNECTIONS:
        cv2.line(frame,
                 (int(landmarks[s].x * w), int(landmarks[s].y * h)),
                 (int(landmarks[e].x * w), int(landmarks[e].y * h)),
                 (255, 255, 255), 2)
    for lm in landmarks:
        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, color, -1)


def predict_hand(model_ai, landmarks):
    """ให้ AI ทำนายท่าทางจาก landmark"""
    row = []
    for lm in landmarks:
        row.extend([lm.x, lm.y, lm.z])
    X_input = pd.DataFrame([row])
    command = model_ai.predict(X_input)[0]
    prob = np.max(model_ai.predict_proba(X_input)[0])
    return command, prob


# ========== Main Loop ==========
cap = cv2.VideoCapture(0)

print("\n" + "=" * 50)
print("  AI Dual-Hand Controller")
print("  มือขวา = ทิศทาง | มือซ้าย = หัวดูด")
print("  กด ESC เพื่อออก")
print("=" * 50)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

    right_cmd, right_prob = "---", 0.0
    left_cmd, left_prob = "---", 0.0
    dir_color = (128, 128, 128)
    act_color = (128, 128, 128)

    if result.hand_landmarks:
        for i, hand in enumerate(result.hand_landmarks):
            label = result.handedness[i][0].category_name  # "Left" or "Right"

            # กล้อง mirror: MediaPipe "Left" = มือขวาจริง
            if label == "Left" and right_model:
                draw_landmarks(frame, hand, (100, 255, 100))
                right_cmd, right_prob = predict_hand(right_model, hand)
                dir_color = DIR_COLORS.get(right_cmd, (128, 128, 128))

            elif label == "Right" and left_model:
                draw_landmarks(frame, hand, (255, 100, 100))
                left_cmd, left_prob = predict_hand(left_model, hand)
                act_color = ACT_COLORS.get(left_cmd, (128, 128, 128))

    # ===== UI: แถบบน (มือขวา - ทิศทาง) =====
    cv2.rectangle(frame, (0, 0), (350, 45), dir_color, -1)
    cv2.putText(frame, f"DIR: {right_cmd} ({right_prob*100:.0f}%)", (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # ===== UI: แถบล่าง (มือซ้าย - หัวดูด) =====
    cv2.rectangle(frame, (0, 50), (350, 95), act_color, -1)
    cv2.putText(frame, f"ACT: {left_cmd} ({left_prob*100:.0f}%)", (10, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("AI Dual-Hand Controller", frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

landmarker.close()
cap.release()
cv2.destroyAllWindows()
