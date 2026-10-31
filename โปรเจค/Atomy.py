# -*- coding: utf-8 -*-
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import argparse
import cv2
import csv
import os
import sys
import io
import tempfile

# ตั้งค่า stdout ให้รองรับ UTF-8 (แก้ปัญหา cp874 บน Windows)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ==========================================
#  สคริปต์บันทึกข้อมูลท่าทางร่างกาย (Anatomy)
#  มือขวา = ควบคุมทิศทาง
#  มือซ้าย = ควบคุมหัวดูด (GRAB / RELEASE)
# ==========================================

# ========== ตั้งค่าท่าทาง ==========
RIGHT_GESTURES = ["FORWARD", "BACKWARD", "LEFT", "RIGHT", "UP", "DOWN"]  # มือขวา: ทิศทาง
LEFT_GESTURES = ["GRAB", "RELEASE"]  # มือซ้าย: หัวดูด

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
RIGHT_CSV = os.path.join(DATA_DIR, "right_hand_dataset.csv")
LEFT_CSV = os.path.join(DATA_DIR, "left_hand_dataset.csv")
SAMPLES_PER_GESTURE = 200

# Path ไปยังไฟล์โมเดล MediaPipe
MODEL_PATH = os.path.join(DATA_DIR, "hand_landmarker.task")

# Header: 21 landmark x 3 (x, y, z) = 63 คอลัมน์ + class
NUM_LANDMARKS = 21
HEADER = []
for i in range(NUM_LANDMARKS):
    HEADER.extend([f"x{i}", f"y{i}", f"z{i}"])
HEADER.append("class")

# เส้นเชื่อม landmark มือ
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]


def draw_landmarks(frame, landmarks, color=(0, 255, 0)):
    """วาด landmark มือบน frame"""
    h, w, _ = frame.shape
    for s, e in HAND_CONNECTIONS:
        cv2.line(frame,
                 (int(landmarks[s].x * w), int(landmarks[s].y * h)),
                 (int(landmarks[e].x * w), int(landmarks[e].y * h)),
                 (255, 255, 255), 2)
    for lm in landmarks:
        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, color, -1)


def create_csv(filepath):
    """สร้างไฟล์ CSV ถ้ายังไม่มี"""
    if not os.path.exists(filepath):
        with open(filepath, 'w', newline='') as f:
            csv.writer(f).writerow(HEADER)
        print(f"[OK] สร้าง {filepath}")
    else:
        print(f"[INFO] พบ {filepath} -- เพิ่มต่อท้าย")


def replace_gesture_rows(filepath, gesture, new_rows):
    """แทนตัวอย่างของท่าหนึ่ง โดยเก็บข้อมูลท่าอื่นไว้"""
    with open(filepath, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    if not rows:
        raise ValueError(f"ไฟล์ข้อมูลว่าง ไม่มีหัวตาราง: {filepath}")

    header = rows[0]
    existing_rows = rows[1:]
    kept_rows = [row for row in existing_rows if not row or row[-1] != gesture]
    removed_count = len(existing_rows) - len(kept_rows)

    with tempfile.TemporaryDirectory(dir=os.path.dirname(filepath)) as temp_dir:
        temp_path = os.path.join(temp_dir, os.path.basename(filepath))
        with open(temp_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(kept_rows)
            writer.writerows(new_rows)
        os.replace(temp_path, filepath)

    return removed_count


def record_hand(landmarker, cap, gestures, csv_file, hand_name, hand_label, hand_color):
    """บันทึกท่าทางของมือข้างหนึ่ง"""
    create_csv(csv_file)

    print(f"\n{'='*60}")
    print(f"  บันทึกท่าทาง: {hand_name}")
    print(f"  ท่าทาง: {', '.join(gestures)}")
    print(f"{'='*60}")

    for gesture in gestures:
        print(f"\n[TARGET] {hand_name} -- เตรียมทำท่า: [{gesture}]")
        print("   กด 'S' เพิ่มข้อมูล | 'R' แทนข้อมูลท่าเดิม | 'ESC' ข้าม | 'Q' ออก")

        # === รอเลือกวิธีบันทึก ===
        waiting, skipped, replace_existing = True, False, False
        while waiting and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

            # วาด landmark ทุกมือที่เจอ
            if result.hand_landmarks:
                for i, hand in enumerate(result.hand_landmarks):
                    h_label = result.handedness[i][0].category_name  # "Left" or "Right"
                    c = (255, 100, 100) if h_label == "Left" else (100, 255, 100)
                    draw_landmarks(frame, hand, c)

            # UI
            cv2.rectangle(frame, (0, 0), (640, 80), (50, 50, 50), -1)
            cv2.putText(frame, f"{hand_name}: {gesture}", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, hand_color, 2)
            cv2.putText(frame, "S=Append | R=Replace | ESC=Skip | Q=Quit", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.imshow("Gesture Recorder", frame)

            key = cv2.waitKey(5) & 0xFF
            if key in (ord("s"), ord("S"), ord("r"), ord("R")):
                replace_existing = key in (ord("r"), ord("R"))
                waiting = False
            elif key == 27:
                print(f"   [SKIP] ข้ามท่า {gesture}")
                waiting, skipped = False, True
            elif key == ord("q") or key == ord("Q"):
                return False  # ออกจากการบันทึก

        if skipped:
            continue

        # === บันทึก ===
        print(f"   [REC] กำลังบันทึก {gesture} (ใช้ {hand_name}) ...")
        count = 0
        new_rows = []

        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            while count < SAMPLES_PER_GESTURE and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

                target_hand = None
                if result.hand_landmarks:
                    for i, hand in enumerate(result.hand_landmarks):
                        h_label = result.handedness[i][0].category_name
                        c = (255, 100, 100) if h_label == "Left" else (100, 255, 100)
                        draw_landmarks(frame, hand, c)
                        # MediaPipe กลับซ้าย-ขวาเพราะ mirror
                        if h_label == hand_label:
                            target_hand = hand

                if target_hand:
                    row = []
                    for lm in target_hand:
                        row.extend([lm.x, lm.y, lm.z])
                    row.append(gesture)
                    if replace_existing:
                        new_rows.append(row)
                    else:
                        writer.writerow(row)
                    count += 1

                # Progress UI
                pct = int((count / SAMPLES_PER_GESTURE) * 100)
                bar_w = int((count / SAMPLES_PER_GESTURE) * 300)
                cv2.rectangle(frame, (0, 0), (640, 80), (50, 50, 50), -1)
                cv2.putText(frame, f"REC {hand_name}: {gesture} [{count}/{SAMPLES_PER_GESTURE}]",
                            (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.rectangle(frame, (10, 50), (310, 70), (100, 100, 100), -1)
                cv2.rectangle(frame, (10, 50), (10 + bar_w, 70), hand_color, -1)
                cv2.putText(frame, f"{pct}%", (320, 67),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.imshow("Gesture Recorder", frame)

                if cv2.waitKey(5) & 0xFF == 27:
                    print(f"   [WARN] หยุดก่อน (ได้ {count} เฟรม)")
                    break

        if replace_existing:
            if count == SAMPLES_PER_GESTURE:
                removed_count = replace_gesture_rows(csv_file, gesture, new_rows)
                print(
                    f"   [OK] แทนข้อมูลท่า {gesture} แล้ว "
                    f"(ลบ {removed_count} แถวเดิม, เพิ่ม {count} แถวใหม่)"
                )
            else:
                print(
                    f"   [WARN] บันทึกไม่ครบ {SAMPLES_PER_GESTURE} แถว "
                    "จึงเก็บข้อมูลเดิมไว้"
                )

        print(f"   [DONE] {gesture} เสร็จ! ({count} เฟรม)")

    return True


def clear_datasets(dataset):
    """ล้างข้อมูลที่เลือกและสร้างหัวตารางใหม่"""
    datasets = {
        "right": [RIGHT_CSV],
        "left": [LEFT_CSV],
        "all": [RIGHT_CSV, LEFT_CSV],
    }
    files = datasets[dataset]

    print("[WARN] การล้างข้อมูลจะลบตัวอย่างเก่าในไฟล์:")
    for filepath in files:
        print(f"  - {filepath}")
    if input("พิมพ์ CLEAR เพื่อยืนยัน: ") != "CLEAR":
        print("[CANCEL] ยกเลิกการล้างข้อมูล")
        return

    for filepath in files:
        with open(filepath, "w", newline="") as f:
            csv.writer(f).writerow(HEADER)
        print(f"[OK] ล้างข้อมูลใน {filepath} แล้ว")


def main():
    parser = argparse.ArgumentParser(description="บันทึกข้อมูลท่าทางมือ")
    parser.add_argument(
        "--clear-data",
        choices=("right", "left", "all"),
        help="ล้างข้อมูลเก่าของมือที่เลือก (right, left หรือ all)",
    )
    args = parser.parse_args()
    if args.clear_data:
        clear_datasets(args.clear_data)
        return

    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] ไม่พบไฟล์โมเดล: {MODEL_PATH}")
        return

    # สร้าง HandLandmarker
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        running_mode=vision.RunningMode.IMAGE,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] เปิดกล้องไม่ได้!")
        return

    print("\n" + "=" * 60)
    print("  ระบบบันทึกท่าทาง (Dual Hand Recorder)")
    print("  มือขวา = ทิศทาง | มือซ้าย = หัวดูด")
    print("=" * 60)

    # บันทึกมือขวา (ทิศทาง)
    # หมายเหตุ: เพราะกล้อง mirror ภาพ, MediaPipe จะเห็น "Left" เป็นมือขวาจริง
    ok = record_hand(landmarker, cap, RIGHT_GESTURES, RIGHT_CSV,
                     "มือขวา (ทิศทาง)", "Left", (100, 255, 100))

    if ok:
        # บันทึกมือซ้าย (หัวดูด)
        record_hand(landmarker, cap, LEFT_GESTURES, LEFT_CSV,
                    "มือซ้าย (หัวดูด)", "Right", (255, 100, 100))

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print("  บันทึกเสร็จทั้งหมด!")
    print(f"  มือขวา: {RIGHT_CSV}")
    print(f"  มือซ้าย: {LEFT_CSV}")
    print(f"  ขั้นตอนถัดไป: รัน train_model.py เพื่อเทรน AI")
    print("=" * 60)


if __name__ == "__main__":
    main()
