# -*- coding: utf-8 -*-
from collections import Counter, deque
import os
import sys
import threading
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from pydobot import Dobot

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(DATA_DIR)
MODEL_PATHS = (
    os.path.join(DATA_DIR, "hand_landmarker.task"),
    os.path.join(ROOT_DIR, "hand_landmarker.task"),
)
CALIBRATION_PATH = os.path.join(DATA_DIR, "hand_xy_calibration.npz")

CAMERA_INDEX = 0
COM_PORT = "COM3"
CAMERA_LABEL_FOR_RIGHT_HAND = "Left"
CAMERA_LABEL_FOR_LEFT_HAND = "Right"

ROBOT_SPEED = 25
MAX_STEP_MM = 5.0
MIN_MOVE_MM = 1.5
MOVE_INTERVAL_SEC = 0.15
POSE_HISTORY_FRAMES = 5
POSE_VOTES_REQUIRED = 4

WINDOW_NAME = "Dobot Hand Position Control"
CALIBRATION_POINT_NAMES = (
    "มุมซ้ายบน",
    "มุมขวาบน",
    "มุมขวาล่าง",
    "มุมซ้ายล่าง",
)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


def find_model_path():
    for path in MODEL_PATHS:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "ไม่พบ hand_landmarker.task ในโฟลเดอร์โปรเจกต์หรือโฟลเดอร์หลัก"
    )


def create_landmarker():
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=find_model_path()),
        num_hands=2,
        running_mode=vision.RunningMode.IMAGE,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    return vision.HandLandmarker.create_from_options(options)


def open_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"เปิดกล้อง index {CAMERA_INDEX} ไม่สำเร็จ")
    return cap


def get_hand_by_label(result, label):
    if not result.hand_landmarks:
        return None
    for index, hand in enumerate(result.hand_landmarks):
        if result.handedness[index][0].category_name == label:
            return hand
    return None


def hand_center(hand, frame_shape):
    height, width = frame_shape[:2]
    palm_points = (hand[0], hand[5], hand[9], hand[13], hand[17])
    center_x = sum(point.x for point in palm_points) / len(palm_points)
    center_y = sum(point.y for point in palm_points) / len(palm_points)
    return center_x * width, center_y * height


def draw_hand(frame, hand, color):
    height, width = frame.shape[:2]
    for start, end in HAND_CONNECTIONS:
        cv2.line(
            frame,
            (int(hand[start].x * width), int(hand[start].y * height)),
            (int(hand[end].x * width), int(hand[end].y * height)),
            (255, 255, 255),
            2,
        )
    for landmark in hand:
        cv2.circle(
            frame,
            (int(landmark.x * width), int(landmark.y * height)),
            4,
            color,
            -1,
        )


def classify_hand_pose(hand):
    wrist = hand[0]
    finger_pairs = ((8, 6), (12, 10), (16, 14), (20, 18))
    extended_count = 0
    folded_count = 0

    for tip_index, joint_index in finger_pairs:
        tip = hand[tip_index]
        joint = hand[joint_index]
        tip_distance = np.hypot(tip.x - wrist.x, tip.y - wrist.y)
        joint_distance = np.hypot(joint.x - wrist.x, joint.y - wrist.y)
        if tip_distance > joint_distance * 1.15:
            extended_count += 1
        else:
            folded_count += 1

    if extended_count >= 3:
        return "OPEN"
    if folded_count >= 3:
        return "FIST"
    return None


def stable_hand_pose(history, pose):
    history.append(pose)
    votes = Counter(item for item in history if item is not None)
    if not votes:
        return None
    best_pose, vote_count = votes.most_common(1)[0]
    return best_pose if vote_count >= POSE_VOTES_REQUIRED else None


def is_convex_quad(points):
    points = np.asarray(points, dtype=np.float32).reshape((-1, 1, 2))
    return (
        points.shape[0] == 4
        and abs(cv2.contourArea(points)) > 1.0
        and cv2.isContourConvex(points)
    )


def build_calibration(image_points, robot_points, image_size):
    image_points = np.asarray(image_points, dtype=np.float32)
    robot_points = np.asarray(robot_points, dtype=np.float32)
    if image_points.shape != (4, 2) or robot_points.shape != (4, 2):
        raise ValueError("ต้องใช้จุดคาลิเบรต 4 คู่")
    if not np.isfinite(image_points).all() or not np.isfinite(robot_points).all():
        raise ValueError("พิกัดคาลิเบรตต้องเป็นตัวเลขที่ถูกต้อง")
    if not is_convex_quad(image_points) or not is_convex_quad(robot_points):
        raise ValueError("จุดทั้ง 4 ต้องเรียงเป็นมุมรอบพื้นที่ทำงานและไม่ตัดกัน")

    homography, _ = cv2.findHomography(image_points, robot_points, method=0)
    if homography is None or not np.isfinite(homography).all():
        raise ValueError("คำนวณพิกัดคาลิเบรตไม่สำเร็จ")

    mapped_points = cv2.perspectiveTransform(
        image_points.reshape((1, 4, 2)), homography
    )[0]
    if np.max(np.linalg.norm(mapped_points - robot_points, axis=1)) > 1.0:
        raise ValueError("ค่าคาลิเบรตคลาดเคลื่อนเกิน 1 มม.")

    return {
        "homography": homography,
        "image_points": image_points,
        "robot_points": robot_points,
        "image_size": np.asarray(image_size, dtype=np.int32),
    }


def save_calibration(calibration):
    np.savez_compressed(
        CALIBRATION_PATH,
        homography=calibration["homography"],
        image_points=calibration["image_points"],
        robot_points=calibration["robot_points"],
        image_size=calibration["image_size"],
    )
    print(f"[OK] บันทึกคาลิเบรตแล้ว: {CALIBRATION_PATH}")


def load_calibration(image_size):
    if not os.path.isfile(CALIBRATION_PATH):
        return None

    with np.load(CALIBRATION_PATH, allow_pickle=False) as data:
        saved_size = np.asarray(data["image_size"], dtype=np.int32)
        if not np.array_equal(saved_size, np.asarray(image_size, dtype=np.int32)):
            print("[WARN] ความละเอียดกล้องเปลี่ยน ต้องคาลิเบรตใหม่")
            return None
        calibration = build_calibration(
            data["image_points"], data["robot_points"], image_size
        )
        saved_homography = np.asarray(data["homography"], dtype=np.float64)
        if saved_homography.shape != (3, 3) or not np.isfinite(saved_homography).all():
            raise ValueError("ไฟล์คาลิเบรตมี homography ที่ไม่ถูกต้อง")
        mapped_points = cv2.perspectiveTransform(
            calibration["image_points"].reshape((1, 4, 2)),
            saved_homography,
        )[0]
        if np.max(
            np.linalg.norm(mapped_points - calibration["robot_points"], axis=1)
        ) > 1.0:
            raise ValueError("homography ในไฟล์คาลิเบรตไม่ตรงกับจุดอ้างอิง")
        calibration["homography"] = saved_homography
        return calibration


def map_pixel_to_robot(calibration, pixel):
    image_polygon = calibration["image_points"].reshape((-1, 1, 2))
    if cv2.pointPolygonTest(image_polygon, pixel, False) < 0:
        return None

    point = np.asarray([[pixel]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(point, calibration["homography"])[0, 0]
    robot_point = (float(mapped[0]), float(mapped[1]))
    robot_polygon = calibration["robot_points"].reshape((-1, 1, 2))
    if cv2.pointPolygonTest(robot_polygon, robot_point, False) < 0:
        return None
    return robot_point


def calibrate_workspace(cap, landmarker):
    image_points = []
    robot_points = []
    print("\n[CALIBRATION] คาลิเบรตพื้นที่ทำงาน X/Y จำนวน 4 จุด")
    print("เรียงจุดตามลำดับ มุมซ้ายบน > ขวาบน > ขวาล่าง > ซ้ายล่าง")
    print("ให้มืออยู่ตรงจุดอ้างอิงของพื้นที่ แล้วกด C เพื่อบันทึกตำแหน่งภาพ")
    print("พิมพ์พิกัด X,Y จริงของ Dobot เป็นมิลลิเมตรในหน้าต่าง Terminal")
    print("ห้ามขยับกล้องหลังคาลิเบรตเสร็จ | กด Q เพื่อยกเลิก\n")

    while len(image_points) < 4:
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError("ไม่สามารถอ่านภาพจากกล้องระหว่างคาลิเบรต")

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = landmarker.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        )
        hand = get_hand_by_label(result, CAMERA_LABEL_FOR_RIGHT_HAND)
        center = None
        if hand is not None:
            draw_hand(frame, hand, (60, 240, 60))
            center = hand_center(hand, frame.shape)
            cv2.circle(frame, tuple(map(int, center)), 8, (0, 0, 255), -1)

        next_point = len(image_points)
        cv2.putText(
            frame,
            f"Calibration {next_point + 1}/4: {CALIBRATION_POINT_NAMES[next_point]}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            "C=Capture hand | Q=Cancel",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q"), 27):
            return None
        if key not in (ord("c"), ord("C")):
            continue
        if center is None:
            print("[WARN] ยังไม่พบมือขวาในภาพ")
            continue

        try:
            x_mm = float(input(f"จุด {next_point + 1} - พิกัด X ของ Dobot (มม.): "))
            y_mm = float(input(f"จุด {next_point + 1} - พิกัด Y ของ Dobot (มม.): "))
        except ValueError:
            print("[WARN] พิกัดต้องเป็นตัวเลข กรุณาบันทึกจุดนี้ใหม่")
            continue
        if not np.isfinite([x_mm, y_mm]).all():
            print("[WARN] พิกัดต้องเป็นตัวเลขที่ถูกต้อง กรุณาบันทึกจุดนี้ใหม่")
            continue

        image_points.append(center)
        robot_points.append((x_mm, y_mm))
        print(
            f"[POINT] {next_point + 1}/4: pixel=({center[0]:.1f},{center[1]:.1f}) "
            f"robot=({x_mm:.1f},{y_mm:.1f}) มม."
        )

    frame_height, frame_width = frame.shape[:2]
    calibration = build_calibration(
        image_points,
        robot_points,
        (frame_width, frame_height),
    )
    save_calibration(calibration)
    return calibration


class RobotCommandWorker:
    def __init__(self, device, initial_pose):
        self.device = device
        self.condition = threading.Condition()
        self.tasks = deque()
        self.busy = False
        self.stopping = False
        self.error = None
        self.current_pose = tuple(initial_pose)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def is_idle(self):
        with self.condition:
            return not self.busy and not self.tasks and self.error is None

    def get_current_pose(self):
        with self.condition:
            return self.current_pose

    def submit_move(self, pose):
        with self.condition:
            if self.busy or self.tasks or self.stopping or self.error is not None:
                return False
            self.tasks.append(("move", tuple(pose)))
            self.condition.notify()
            return True

    def submit_suction(self, enabled):
        with self.condition:
            if self.stopping or self.error is not None:
                return False
            if self.tasks and self.tasks[-1] == ("suction", enabled):
                return True
            self.tasks.append(("suction", enabled))
            self.condition.notify()
            return True

    def _run(self):
        while True:
            with self.condition:
                while not self.tasks and not self.stopping:
                    self.condition.wait()
                if self.stopping and not self.tasks:
                    return
                command, value = self.tasks.popleft()
                self.busy = True

            try:
                if command == "move":
                    self.device.move_to(*value, wait=True)
                else:
                    self.device.suck(value)
            except Exception as error:
                with self.condition:
                    self.error = error
                    self.tasks.clear()
                    self.busy = False
                    self.stopping = True
                    self.condition.notify_all()
                return

            with self.condition:
                if command == "move":
                    self.current_pose = value
                self.busy = False
                self.condition.notify_all()

    def close(self):
        with self.condition:
            self.stopping = True
            self.tasks.clear()
            self.condition.notify_all()
        self.thread.join()


def step_toward(current_xy, target_xy, max_step):
    delta = np.asarray(target_xy, dtype=np.float64) - np.asarray(
        current_xy, dtype=np.float64
    )
    distance = float(np.linalg.norm(delta))
    if distance <= MIN_MOVE_MM:
        return None
    if distance > max_step:
        delta *= max_step / distance
    next_xy = np.asarray(current_xy, dtype=np.float64) + delta
    return float(next_xy[0]), float(next_xy[1])


def pose_is_inside_workspace(calibration, pose):
    robot_polygon = calibration["robot_points"].reshape((-1, 1, 2))
    return cv2.pointPolygonTest(
        robot_polygon, (float(pose[0]), float(pose[1])), False
    ) >= 0


def process_gripper_pose(
    worker,
    hand,
    history,
    previous_pose,
    suction_state,
    enabled,
):
    if not enabled:
        history.clear()
        return previous_pose, suction_state

    current = classify_hand_pose(hand) if hand is not None else None
    stable = stable_hand_pose(history, current)
    if stable is None:
        return previous_pose, suction_state

    if previous_pose is None:
        if stable == "OPEN":
            print("[GRIPPER] มือซ้ายแบมือแล้ว พร้อมรับคำสั่งกำ/แบ")
        return stable, suction_state

    if stable == previous_pose:
        return previous_pose, suction_state

    enabled_suction = stable == "FIST"
    if worker.submit_suction(enabled_suction):
        state_text = "หยิบ (เปิดหัวดูด)" if enabled_suction else "ปล่อย (ปิดหัวดูด)"
        print(f"[GRIPPER] {state_text}")
        return stable, enabled_suction
    return previous_pose, suction_state


def annotate_status(
    frame,
    right_hand,
    left_hand,
    target_xy,
    enabled,
    worker,
    suction_state,
    pointer_in_workspace,
):
    if right_hand is not None:
        draw_hand(frame, right_hand, (60, 240, 60))
    if left_hand is not None:
        draw_hand(frame, left_hand, (255, 120, 80))

    enabled_text = "ON - กด Space เพื่อหยุดส่งคำสั่ง" if enabled else "OFF"
    cv2.putText(
        frame,
        f"CONTROL: {enabled_text}",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (0, 0, 255) if enabled else (0, 220, 255),
        2,
    )

    if target_xy is not None:
        cv2.putText(
            frame,
            f"Robot target XY: ({target_xy[0]:.1f}, {target_xy[1]:.1f}) mm",
            (10, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (60, 240, 60) if pointer_in_workspace else (0, 0, 255),
            2,
        )
    else:
        cv2.putText(
            frame,
            "ไม่พบมือขวาในพื้นที่คาลิเบรต",
            (10, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 0, 255),
            2,
        )

    current_pose = worker.get_current_pose()
    suction_text = (
        "ไม่ทราบสถานะ"
        if suction_state is None
        else "ON / ดูด" if suction_state else "OFF / ปล่อย"
    )
    status_text = (
        f"Robot XY: ({current_pose[0]:.1f}, {current_pose[1]:.1f}) mm | "
        f"หัวดูด: {suction_text}"
    )
    cv2.putText(
        frame,
        status_text,
        (10, 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        "Space=Enable/Disable | C=Calibrate | Q=Quit",
        (10, frame.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )


def main():
    cap = None
    landmarker = None
    device = None
    worker = None
    try:
        cap = open_camera()
        landmarker = create_landmarker()
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        ok, first_frame = cap.read()
        if not ok or first_frame is None:
            raise RuntimeError("ไม่สามารถอ่านภาพจากกล้องได้")
        frame_height, frame_width = first_frame.shape[:2]
        image_size = (frame_width, frame_height)

        try:
            calibration = load_calibration(image_size)
        except (OSError, ValueError, KeyError) as error:
            print(f"[WARN] อ่านไฟล์คาลิเบรตไม่ได้: {error}")
            calibration = None
        if calibration is None:
            calibration = calibrate_workspace(cap, landmarker)
            if calibration is None:
                print("[INFO] ยกเลิกการคาลิเบรต")
                return

        print(f"[INFO] กำลังเชื่อมต่อ Dobot ที่พอร์ต {COM_PORT}...")
        device = Dobot(port=COM_PORT, verbose=False)
        device.speed(velocity=ROBOT_SPEED, acceleration=ROBOT_SPEED)
        pose = tuple(float(value) for value in device.pose()[:4])
        print(
            f"[OK] เชื่อมต่อ Dobot | เริ่มต้น X={pose[0]:.1f}, "
            f"Y={pose[1]:.1f}, Z={pose[2]:.1f}, R={pose[3]:.1f}"
        )
        print("[SAFE] แกน Z และ R จะคงค่าตำแหน่งเริ่มต้นตลอดการควบคุม")
        print("[SAFE] วางแขนกลให้อยู่ภายในพื้นที่คาลิเบรตก่อนกด Space")
        print("[INFO] มือขวาควบคุมตำแหน่ง X/Y | มือซ้ายแบมือเพื่อเตรียม แล้วกำมือ=หยิบ แบมือ=ปล่อย")
        print("[INFO] กด C เพื่อคาลิเบรตใหม่ | กด Space เพื่อเปิด/ปิดการควบคุม | กด Q เพื่อออก")
        worker = RobotCommandWorker(device, pose)

        enabled = False
        suction_state = None
        previous_gripper_pose = None
        gripper_history = deque(maxlen=POSE_HISTORY_FRAMES)
        last_move_time = 0.0

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError("สัญญาณภาพจากกล้องขาดหาย")

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = landmarker.detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            )
            right_hand = get_hand_by_label(
                result, CAMERA_LABEL_FOR_RIGHT_HAND
            )
            left_hand = get_hand_by_label(
                result, CAMERA_LABEL_FOR_LEFT_HAND
            )

            target_xy = None
            pointer_in_workspace = False
            if right_hand is not None:
                pixel = hand_center(right_hand, frame.shape)
                target_xy = map_pixel_to_robot(calibration, pixel)
                pointer_in_workspace = target_xy is not None
                cv2.circle(
                    frame,
                    tuple(map(int, pixel)),
                    8,
                    (0, 255, 255) if pointer_in_workspace else (0, 0, 255),
                    -1,
                )

            previous_gripper_pose, suction_state = process_gripper_pose(
                worker,
                left_hand,
                gripper_history,
                previous_gripper_pose,
                suction_state,
                enabled,
            )

            now = time.monotonic()
            if (
                enabled
                and target_xy is not None
                and worker.is_idle()
                and now - last_move_time >= MOVE_INTERVAL_SEC
            ):
                current_pose = worker.get_current_pose()
                next_xy = step_toward(
                    current_pose[:2],
                    target_xy,
                    MAX_STEP_MM,
                )
                if next_xy is not None:
                    next_pose = (
                        next_xy[0],
                        next_xy[1],
                        current_pose[2],
                        current_pose[3],
                    )
                    if worker.submit_move(next_pose):
                        last_move_time = now

            if worker.error is not None:
                raise RuntimeError(
                    f"คำสั่ง Dobot ล้มเหลว: {worker.error}"
                ) from worker.error

            annotate_status(
                frame,
                right_hand,
                left_hand,
                target_xy,
                enabled,
                worker,
                suction_state,
                pointer_in_workspace,
            )
            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break
            if key == ord(" "):
                if not enabled:
                    current_pose = worker.get_current_pose()
                    if not pose_is_inside_workspace(calibration, current_pose):
                        print(
                            "[WARN] ตำแหน่งปลายแขนอยู่นอกพื้นที่คาลิเบรต "
                            "จึงยังไม่เปิดการควบคุม"
                        )
                        continue
                    previous_gripper_pose = None
                enabled = not enabled
                print(
                    "[CONTROL] เปิดการควบคุม"
                    if enabled
                    else "[CONTROL] ปิดการควบคุม"
                )
            elif key in (ord("c"), ord("C")):
                enabled = False
                print("[INFO] รอให้การเคลื่อนที่ปัจจุบันเสร็จก่อนคาลิเบรตใหม่")
                while not worker.is_idle():
                    if worker.error is not None:
                        raise RuntimeError(
                            f"คำสั่ง Dobot ล้มเหลว: {worker.error}"
                        ) from worker.error
                    time.sleep(0.05)
                new_calibration = calibrate_workspace(cap, landmarker)
                if new_calibration is not None:
                    calibration = new_calibration
                    print("[OK] ใช้ค่าคาลิเบรตใหม่แล้ว")

    finally:
        if worker is not None:
            worker.close()
        if device is not None:
            device.close()
        if cap is not None:
            cap.release()
        if landmarker is not None:
            landmarker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
