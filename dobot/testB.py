# -*- coding: utf-8 -*-
from collections import deque

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import pickle
import pandas as pd
import numpy as np
import time
import os
import sys
import io
from pydobot import Dobot
from pydobot.enums import PTPMode

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ==========================================
#  AI Gesture + Dobot Magician Controller
#  มือขวา = ควบคุมทิศทางแขนกล
#  มือซ้าย = ควบคุมหัวดูด (GRAB / RELEASE)
# ==========================================

# ========== ตั้งค่า Dobot ==========
COM_PORT = 'COM3'
MOVE_STEP = 1          # องศาที่ขยับต่อคำสั่ง
COMMAND_DELAY = 0.15   # เว้นระยะระหว่างการส่งคำสั่ง (วินาที)
SMOOTHING_FRAMES = 3   # ใช้ผลทำนาย 3 เฟรมล่าสุด
SMOOTHING_VOTES = 2    # ต้องตรงกันอย่างน้อย 2 เฟรม

# ขอบเขตองศาข้อต่อ (ป้องกันชน)
J1_MIN, J1_MAX = -135, 135   # หมุนฐาน ซ้าย-ขวา
J2_MIN, J2_MAX = -5, 85      # ไหล่ ก้ม-เงย
J3_MIN, J3_MAX = -25, 60     # ข้อศอก

# ========== โหลดโมเดล AI ==========
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

right_model, left_model = None, None
right_pkl = os.path.join(MODEL_DIR, 'right_hand_model.pkl')
left_pkl = os.path.join(MODEL_DIR, 'left_hand_model.pkl')

if os.path.exists(right_pkl):
    with open(right_pkl, 'rb') as f:
        right_model = pickle.load(f)
    print("[OK] โหลดโมเดลมือขวา (ทิศทาง)")

if os.path.exists(left_pkl):
    with open(left_pkl, 'rb') as f:
        left_model = pickle.load(f)
    print("[OK] โหลดโมเดลมือซ้าย (หัวดูด)")

if not right_model and not left_model:
    print("[ERROR] ไม่พบโมเดล! รัน train_model.py ก่อน")
    sys.exit(1)

# ========== MediaPipe ==========
MP_MODEL = os.path.join(MODEL_DIR, 'hand_landmarker.task')
if not os.path.exists(MP_MODEL):
    print(f"[ERROR] ไม่พบ {MP_MODEL}")
    sys.exit(1)

base_options = python.BaseOptions(model_asset_path=MP_MODEL)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    running_mode=vision.RunningMode.IMAGE,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
landmarker = vision.HandLandmarker.create_from_options(options)

# ========== สี UI ==========
DIR_COLORS = {
    "FRONT": (0, 255, 0), "BACK": (0, 165, 255),
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
    row = []
    for lm in landmarks:
        row.extend([lm.x, lm.y, lm.z])
    X_input = pd.DataFrame([row])
    command = model_ai.predict(X_input)[0]
    prob = np.max(model_ai.predict_proba(X_input)[0])
    return command, prob


def clamp(value, min_val, max_val):
    return max(min_val, min(max_val, value))


def smooth_prediction(history, command, probability):
    """ยืนยันคำสั่งเมื่อผลทำนายตรงกันอย่างน้อย 2 ใน 3 เฟรม"""
    history.append(command if probability >= MIN_PROB else None)
    votes = [item for item in history if item is not None]

    if not votes:
        return "---"

    best_command = max(set(votes), key=votes.count)
    return best_command if votes.count(best_command) >= SMOOTHING_VOTES else "---"


# ==========================================
#  เชื่อมต่อ Dobot Magician
# ==========================================
device = None
try:
    print(f"\n[INFO] กำลังเชื่อมต่อ Dobot ที่พอร์ต {COM_PORT}...")
    device = Dobot(port=COM_PORT, verbose=False)
    device.speed(velocity=40, acceleration=40)
    print("[OK] เชื่อมต่อ Dobot สำเร็จ!")
except Exception as e:
    print(f"[WARN] เชื่อมต่อ Dobot ไม่ได้: {e}")
    print("[INFO] จะทำงานในโหมดจำลอง (Simulation Mode)")

# ฟังก์ชันควบคุม Dobot
def move_joints(joints):
    if device:
        try:
            device._set_ptp_cmd(
                joints[0], joints[1], joints[2], joints[3],
                mode=PTPMode.MOVJ_ANGLE,
                wait=False
            )
        except AttributeError:
            pass # Ignore serial timeout/response errors

def set_suction(enabled):
    if device:
        device.suck(enabled)

# ตำแหน่งปัจจุบันของข้อต่อ
current_joints = [0, 10, 10, 0]  # [J1, J2, J3, J4] เริ่มที่ Home
suction_on = False

# ไป Home ก่อน
print("[INFO] ไปตำแหน่ง Home...")
move_joints(current_joints)

# ========== Main Loop ==========
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] เปิดกล้องไม่ได้!")
    sys.exit(1)

print("\n" + "=" * 55)
print("  AI Gesture + Dobot Controller")
print("  มือขวา = ทิศทาง | มือซ้าย = หัวดูด")
print("  กด ESC เพื่อออก")
print("=" * 55)

MIN_PROB = 0.6         # ความมั่นใจขั้นต่ำถึงจะสั่งงาน
last_cmd_time = 0      # เวลาคำสั่งล่าสุด
right_history = deque(maxlen=SMOOTHING_FRAMES)
left_history = deque(maxlen=SMOOTHING_FRAMES)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

    right_cmd, right_prob = "---", 0.0
    left_cmd, left_prob = "---", 0.0

    if result.hand_landmarks:
        for i, hand in enumerate(result.hand_landmarks):
            label = result.handedness[i][0].category_name

            # กล้อง mirror: MediaPipe "Left" = มือขวาจริง
            if label == "Left" and right_model:
                draw_landmarks(frame, hand, (100, 255, 100))
                right_cmd, right_prob = predict_hand(right_model, hand)
                if right_cmd == "FORWARD":
                    right_cmd = "FRONT"
                elif right_cmd == "BACKWARD":
                    right_cmd = "BACK"

            elif label == "Right" and left_model:
                draw_landmarks(frame, hand, (255, 100, 100))
                left_cmd, left_prob = predict_hand(left_model, hand)

    right_cmd = smooth_prediction(right_history, right_cmd, right_prob)
    left_cmd = smooth_prediction(left_history, left_cmd, left_prob)
    if right_cmd == "---":
        right_prob = 0.0
    if left_cmd == "---":
        left_prob = 0.0
    dir_color = DIR_COLORS.get(right_cmd, (128, 128, 128))
    act_color = ACT_COLORS.get(left_cmd, (128, 128, 128))

    now = time.time()

    # ===== สั่ง Dobot ตามมือขวา (ทิศทาง) =====
    if right_prob >= MIN_PROB and (now - last_cmd_time) > COMMAND_DELAY:
        moved = False

        if right_cmd == "FRONT":
            current_joints[1] = clamp(current_joints[1] + MOVE_STEP, J2_MIN, J2_MAX)
            moved = True

        elif right_cmd == "BACK":
            current_joints[1] = clamp(current_joints[1] - MOVE_STEP, J2_MIN, J2_MAX)
            moved = True

        elif right_cmd == "LEFT":
            current_joints[0] = clamp(current_joints[0] + MOVE_STEP, J1_MIN, J1_MAX)
            moved = True

        elif right_cmd == "RIGHT":
            current_joints[0] = clamp(current_joints[0] - MOVE_STEP, J1_MIN, J1_MAX)
            moved = True

        elif right_cmd == "UP":
            current_joints[2] = clamp(current_joints[2] - MOVE_STEP, J3_MIN, J3_MAX)
            moved = True

        elif right_cmd == "DOWN":
            current_joints[2] = clamp(current_joints[2] + MOVE_STEP, J3_MIN, J3_MAX)
            moved = True

        if moved:
            move_joints(current_joints)
            last_cmd_time = now

    # ===== สั่ง Dobot ตามมือซ้าย (หัวดูด) =====
    if left_prob >= MIN_PROB:
        if left_cmd == "GRAB" and not suction_on:
            set_suction(True)
            suction_on = True
            print("[DOBOT] เปิดหัวดูด")

        elif left_cmd == "RELEASE" and suction_on:
            set_suction(False)
            suction_on = False
            print("[DOBOT] ปิดหัวดูด")

    # ===== UI =====
    # แถบบน: มือขวา (ทิศทาง)
    cv2.rectangle(frame, (0, 0), (400, 40), dir_color, -1)
    cv2.putText(frame, f"DIR: {right_cmd} ({right_prob*100:.0f}%)", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # แถบกลาง: มือซ้าย (หัวดูด)
    cv2.rectangle(frame, (0, 45), (400, 85), act_color, -1)
    suction_label = "ON" if suction_on else "OFF"
    cv2.putText(frame, f"ACT: {left_cmd} ({left_prob*100:.0f}%) | Suction: {suction_label}",
                (10, 73), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # แถบล่าง: ตำแหน่งข้อต่อปัจจุบัน
    cv2.rectangle(frame, (0, 90), (400, 120), (50, 50, 50), -1)
    cv2.putText(frame,
                f"J1:{current_joints[0]} J2:{current_joints[1]} J3:{current_joints[2]} J4:{current_joints[3]}",
                (10, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("AI Gesture + Dobot Controller", frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

# ===== Cleanup =====
print("\n[INFO] กำลังปิดระบบ...")
landmarker.close()
cap.release()
cv2.destroyAllWindows()

if device:
    set_suction(False)
    move_joints([0, 10, 10, 0])  # กลับ Home
    device.close()
    print("[OK] ปิด Dobot เรียบร้อย")

print("[OK] ปิดระบบทั้งหมดแล้ว")