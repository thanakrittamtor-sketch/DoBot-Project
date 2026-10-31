"""Simple webcam-controlled RC car game.

Run from any directory with: python game/game.py
The trained model is loaded from the project root next to this folder.
"""

from collections import deque
from pathlib import Path
import pickle
import sys
import time

import cv2
import mediapipe as mp
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_FILE = PROJECT_DIR / "gesture_model.pkl"
WINDOW_NAME = "Gesture RC Car"
CAMERA_WIDTH = 380
WINDOW_WIDTH = 1120
WINDOW_HEIGHT = 640
GAME_X = CAMERA_WIDTH
CONFIDENCE_THRESHOLD = 0.60
SMOOTHING_FRAMES = 5

COMMAND_COLORS = {
    "FORWARD": (70, 210, 70),
    "BACKWARD": (40, 150, 245),
    "LEFT": (235, 190, 40),
    "RIGHT": (220, 90, 220),
    "STOP": (70, 70, 240),
}


def load_model():
    if not MODEL_FILE.is_file():
        raise FileNotFoundError(
            f"ไม่พบโมเดล {MODEL_FILE}\n"
            "กรุณารัน train_model.py เพื่อสร้าง gesture_model.pkl ก่อน"
        )

    with MODEL_FILE.open("rb") as model_file:
        model = pickle.load(model_file)

    if getattr(model, "n_features_in_", None) != 63:
        raise ValueError("โมเดลต้องรับข้อมูลพิกัดมือ 63 ค่า กรุณาฝึกโมเดลใหม่")

    expected_commands = {"FORWARD", "BACKWARD", "LEFT", "RIGHT", "STOP"}
    if not expected_commands.issubset(set(getattr(model, "classes_", []))):
        raise ValueError("โมเดลต้องมีคำสั่ง FORWARD, BACKWARD, LEFT, RIGHT และ STOP")
    return model


def draw_text(image, text, position, scale=0.65, color=(240, 240, 240), thickness=2):
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_car(image, center, heading):
    car_length = 48
    car_width = 27
    forward = np.array([np.sin(heading), -np.cos(heading)])
    right = np.array([np.cos(heading), np.sin(heading)])
    center_point = np.array(center, dtype=float)
    corners = [
        center_point + forward * car_length / 2 + right * car_width / 2,
        center_point + forward * car_length / 2 - right * car_width / 2,
        center_point - forward * car_length / 2 - right * car_width / 2,
        center_point - forward * car_length / 2 + right * car_width / 2,
    ]
    polygon = np.array(corners, dtype=np.int32)
    cv2.fillPoly(image, [polygon], (55, 185, 255))
    cv2.polylines(image, [polygon], True, (250, 250, 250), 2, cv2.LINE_AA)
    nose = center_point + forward * (car_length / 2 - 5)
    cv2.circle(image, tuple(nose.astype(int)), 4, (40, 40, 255), -1)


def main():
    model = load_model()
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError("เปิดกล้องไม่ได้ กรุณาตรวจสอบการเชื่อมต่อกล้อง")

    holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    drawing = mp.solutions.drawing_utils
    hand_connections = mp.solutions.hands.HAND_CONNECTIONS
    recent_commands = deque(maxlen=SMOOTHING_FRAMES)

    car_x = GAME_X + 370.0
    car_y = 350.0
    heading = 0.0
    speed = 0.0
    max_speed = 245.0
    max_reverse_speed = 115.0
    last_frame_time = time.monotonic()

    print("Gesture RC Car เริ่มทำงานแล้ว")
    print("FORWARD / BACKWARD = เดินหน้า / ถอยหลัง")
    print("LEFT / RIGHT = เลี้ยวซ้าย / ขวา | STOP = หยุด | ESC = ออก")

    try:
        while True:
            success, camera_frame = camera.read()
            if not success:
                raise RuntimeError("อ่านภาพจากกล้องไม่ได้ กรุณาตรวจสอบการเชื่อมต่อกล้อง")

            camera_frame = cv2.flip(camera_frame, 1)
            results = holistic.process(cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB))
            hand_landmarks = results.right_hand_landmarks or results.left_hand_landmarks
            command = "NO HAND"
            confidence = 0.0

            if hand_landmarks:
                drawing.draw_landmarks(camera_frame, hand_landmarks, hand_connections)
                features = np.array(
                    [[coordinate for point in hand_landmarks.landmark
                      for coordinate in (point.x, point.y, point.z)]],
                    dtype=np.float32,
                )
                probabilities = model.predict_proba(features)[0]
                best_index = int(np.argmax(probabilities))
                confidence = float(probabilities[best_index])
                if confidence >= CONFIDENCE_THRESHOLD:
                    recent_commands.append(str(model.classes_[best_index]))
                    command = max(set(recent_commands), key=recent_commands.count)
                else:
                    recent_commands.clear()
                    command = "UNCERTAIN"
            else:
                recent_commands.clear()

            if command in ("NO HAND", "UNCERTAIN"):
                drive_command = "STOP"
            else:
                drive_command = command

            now = time.monotonic()
            delta_time = min(now - last_frame_time, 0.05)
            last_frame_time = now

            if drive_command == "FORWARD":
                speed = min(speed + 250.0 * delta_time, max_speed)
            elif drive_command == "BACKWARD":
                speed = max(speed - 250.0 * delta_time, -max_reverse_speed)
            else:
                speed = max(0.0, speed - 340.0 * delta_time) if speed > 0 else min(
                    0.0, speed + 340.0 * delta_time
                )

            if drive_command == "LEFT":
                heading -= 2.3 * delta_time * max(0.3, abs(speed) / max_speed)
            elif drive_command == "RIGHT":
                heading += 2.3 * delta_time * max(0.3, abs(speed) / max_speed)

            car_x += float(np.sin(heading)) * speed * delta_time
            car_y -= float(np.cos(heading)) * speed * delta_time

            left_bound, right_bound = GAME_X + 52, WINDOW_WIDTH - 52
            top_bound, bottom_bound = 112, WINDOW_HEIGHT - 48
            if not (left_bound <= car_x <= right_bound and top_bound <= car_y <= bottom_bound):
                car_x = min(max(car_x, left_bound), right_bound)
                car_y = min(max(car_y, top_bound), bottom_bound)
                speed = 0.0

            camera_view = cv2.resize(camera_frame, (CAMERA_WIDTH, 285))
            screen = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
            screen[:] = (30, 35, 42)
            screen[:285, :CAMERA_WIDTH] = camera_view
            cv2.rectangle(screen, (0, 0), (WINDOW_WIDTH, 54), (25, 30, 38), -1)
            draw_text(screen, "GESTURE RC CAR", (18, 36), 0.8)
            draw_text(screen, "ESC: quit", (WINDOW_WIDTH - 150, 35), 0.55, (190, 190, 190))
            draw_text(screen, "CAMERA", (18, 320), 0.65)
            status_color = COMMAND_COLORS.get(drive_command, (150, 150, 150))
            draw_text(screen, f"Gesture: {command}", (18, 360), 0.62, status_color)
            draw_text(screen, f"Confidence: {confidence * 100:.0f}%", (18, 395), 0.55)
            draw_text(screen, "Show a trained hand gesture to drive", (18, 445), 0.47, (190, 190, 190))
            draw_text(screen, "No hand / unclear gesture = brake", (18, 475), 0.47, (190, 190, 190))

            cv2.rectangle(screen, (GAME_X + 24, 82), (WINDOW_WIDTH - 24, WINDOW_HEIGHT - 24), (47, 105, 58), -1)
            track_center = (GAME_X + 370, 340)
            cv2.ellipse(screen, track_center, (270, 190), 0, 0, 360, (85, 88, 90), -1)
            cv2.ellipse(screen, track_center, (178, 98), 0, 0, 360, (47, 105, 58), -1)
            cv2.ellipse(screen, track_center, (225, 144), 0, 0, 360, (215, 215, 205), 2)
            cv2.ellipse(screen, track_center, (225, 144), 0, 0, 360, (245, 245, 245), 1, cv2.LINE_AA)
            draw_text(screen, f"COMMAND: {drive_command}", (GAME_X + 28, 115), 0.7, status_color)
            draw_text(screen, f"SPEED: {abs(speed):.0f}", (GAME_X + 28, 151), 0.58)
            draw_car(screen, (car_x, car_y), heading)

            cv2.imshow(WINDOW_NAME, screen)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
    finally:
        holistic.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"เกิดข้อผิดพลาด: {error}", file=sys.stderr)
        sys.exit(1)
