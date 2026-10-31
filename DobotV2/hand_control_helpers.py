# -*- coding: utf-8 -*-
"""ส่วนช่วยที่ testC.py import ไปใช้

ดูแลกล้อง/MediaPipe/ท่ามือ การแปลงพิกัด FOLLOW/JOG
และการเชื่อม Dobot จริงเพื่อส่งคำสั่งเคลื่อนที่กับหัวดูด
"""

from collections import Counter, deque
import os
import struct
import threading
import time

import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from pydobot import Dobot
from pydobot.enums import PTPMode

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
MODEL_CANDIDATES = (
    os.path.join(SCRIPT_DIR, "hand_landmarker.task"),
    os.path.join(PROJECT_DIR, "hand_landmarker.task"),
)

CAMERA_INDEX = 0
COM_PORT = "COM3"
HOME_JOINTS = (0.0, 10.0, 10.0, 0.0)
RIGHT_HAND_LABEL = "Left"
LEFT_HAND_LABEL = "Right"

ROBOT_SPEED_MM_S = 50
MAX_STEP_MM = 5.0
FOLLOW_DEADBAND_MM = 1.5
MOVE_INTERVAL_SEC = 0.15
CONTROL_FRAME_WIDTH_MM = 200.0
CONTROL_FRAME_HEIGHT_MM = 200.0
CONTROL_FRAME_RATIO = 0.65
EDGE_JOG_RATIO = 0.1
MAX_Z_STEP_MM = 5.0
POSE_HISTORY_FRAMES = 5
POSE_VOTES_REQUIRED = 4
WINDOW_NAME = "Dobot Live Hand Control"

# จุด landmark ที่เชื่อมเป็นเส้นโครงกระดูกมือในหน้าต่างภาพ
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

# 2) กล้องและ MediaPipe: เตรียมโมเดล รับภาพ และเลือกมือที่ตรวจพบ
def find_model_path():
    """ค้นหาไฟล์โมเดล hand_landmarker.task ในตำแหน่งที่โปรแกรมรองรับ"""
    for path in MODEL_CANDIDATES:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "ไม่พบ hand_landmarker.task ในโฟลเดอร์ dobot หรือโฟลเดอร์โปรเจกต์"
    )


def open_camera():
    """เปิดกล้องตาม CAMERA_INDEX และลอง backend สำรองหากจำเป็น"""
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"เปิดกล้อง index {CAMERA_INDEX} ไม่สำเร็จ")
    return cap


def create_landmarker():
    """สร้างตัวตรวจจับ MediaPipe ที่ตรวจมือได้พร้อมกันสูงสุดสองข้าง"""
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=find_model_path()),
        num_hands=2,
        running_mode=vision.RunningMode.IMAGE,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    return vision.HandLandmarker.create_from_options(options)


def find_hand(result, handedness_label):
    """คืน landmark ของมือที่ตรงกับชื่อ handedness หรือ None หากไม่พบ"""
    if not result.hand_landmarks:
        return None
    for index, landmarks in enumerate(result.hand_landmarks):
        if result.handedness[index][0].category_name == handedness_label:
            return landmarks
    return None


def get_palm_center(hand, frame_shape):
    """คำนวณจุดกึ่งกลางฝ่ามือเป็นพิกเซลสำหรับสั่งเคลื่อน XY"""
    height, width = frame_shape[:2]
    palm_points = (hand[0], hand[5], hand[9], hand[13], hand[17])
    center_x = sum(point.x for point in palm_points) / len(palm_points)
    center_y = sum(point.y for point in palm_points) / len(palm_points)
    return center_x * width, center_y * height


# 3) ท่าทางมือ: วาดจุดตรวจจับ และจำแนกกำมือ/แบมือ/V
def draw_hand(frame, landmarks, color):
    """วาด landmark และเส้นข้อต่อมือทับบนภาพกล้อง"""
    height, width = frame.shape[:2]
    for start, end in HAND_CONNECTIONS:
        cv2.line(
            frame,
            (int(landmarks[start].x * width), int(landmarks[start].y * height)),
            (int(landmarks[end].x * width), int(landmarks[end].y * height)),
            (255, 255, 255),
            2,
        )
    for point in landmarks:
        cv2.circle(
            frame,
            (int(point.x * width), int(point.y * height)),
            4,
            color,
            -1,
        )


def classify_hand_pose(hand):
    """จำแนก OPEN/FIST สำหรับมือซ้ายที่สั่งหยิบหรือปล่อยชิ้นงาน"""
    wrist = hand[0]
    finger_pairs = ((8, 6), (12, 10), (16, 14), (20, 18))
    extended = 0
    folded = 0

    for tip_index, joint_index in finger_pairs:
        tip = hand[tip_index]
        joint = hand[joint_index]
        tip_distance = np.hypot(tip.x - wrist.x, tip.y - wrist.y)
        joint_distance = np.hypot(joint.x - wrist.x, joint.y - wrist.y)
        if tip_distance > joint_distance * 1.15:
            extended += 1
        else:
            folded += 1

    if extended >= 3:
        return "OPEN"
    if folded >= 3:
        return "FIST"
    return None


def classify_vertical_gesture(hand):
    """จำแนก UP จากท่า V และ DOWN จากกำมือ สำหรับควบคุมแกน Z"""
    wrist = hand[0]
    finger_pairs = ((8, 6), (12, 10), (16, 14), (20, 18))
    extended = []
    for tip_index, joint_index in finger_pairs:
        tip = hand[tip_index]
        joint = hand[joint_index]
        tip_distance = np.hypot(tip.x - wrist.x, tip.y - wrist.y)
        joint_distance = np.hypot(joint.x - wrist.x, joint.y - wrist.y)
        extended.append(tip_distance > joint_distance * 1.15)

    index_extended, middle_extended, ring_extended, pinky_extended = extended
    if index_extended and middle_extended and not ring_extended and not pinky_extended:
        return "UP"
    if not any(extended):
        return "DOWN"
    return None


def stable_pose(history, pose):
    """กรองผลท่ามือด้วยเสียงข้างมากจากหลายเฟรมเพื่อลดการสั่งผิด"""
    history.append(pose)
    votes = Counter(item for item in history if item is not None)
    if not votes:
        return None
    best_pose, count = votes.most_common(1)[0]
    return best_pose if count >= POSE_VOTES_REQUIRED else None


# 4) กรอบและพิกัด: แปลงตำแหน่งมือเป็นพิกัด Dobot จริง
def get_control_frame(frame_shape):
    """สร้างกรอบควบคุมให้อยู่กึ่งกลางภาพ โดยอิงสัดส่วนภาพจากกล้อง"""
    height, width = frame_shape[:2]
    box_width = max(2, int(width * CONTROL_FRAME_RATIO))
    box_height = max(2, int(height * CONTROL_FRAME_RATIO))
    left = (width - box_width) // 2
    top = (height - box_height) // 2
    return left, top, left + box_width, top + box_height


def pixel_to_robot(pixel, control_frame, origin_pose):
    """แปลงพิกเซลในกรอบเป็นพิกัด XY มม.; นอกกรอบจะคืน None"""
    left, top, right, bottom = control_frame
    pixel_x, pixel_y = pixel
    if not (left <= pixel_x <= right and top <= pixel_y <= bottom):
        return None

    image_x = (pixel_x - left) / (right - left)
    image_y = (pixel_y - top) / (bottom - top)
    robot_x = origin_pose[0] + (0.5 - image_y) * CONTROL_FRAME_HEIGHT_MM
    robot_y = origin_pose[1] + (image_x - 0.5) * CONTROL_FRAME_WIDTH_MM
    return float(robot_x), float(robot_y)


def draw_control_frame(frame, control_frame, blocked_directions=None):
    """วาดกรอบควบคุมและแจ้งเมื่อถึงขอบเขตการเคลื่อนที่ที่กำหนด"""
    left, top, right, bottom = control_frame
    center_x = (left + right) // 2
    center_y = (top + bottom) // 2
    blocked_directions = blocked_directions or set()
    limit_reached = bool(blocked_directions)
    border_color = (0, 0, 255) if limit_reached else (0, 255, 0)
    cv2.rectangle(frame, (left, top), (right, bottom), border_color, 3)
    cv2.line(
        frame,
        (center_x - 12, center_y),
        (center_x + 12, center_y),
        border_color,
        2,
    )
    cv2.line(
        frame,
        (center_x, center_y - 12),
        (center_x, center_y + 12),
        border_color,
        2,
    )
    if limit_reached:
        blocked_text = ", ".join(sorted(blocked_directions))
        cv2.putText(
            frame,
            f"ROBOT CANNOT MOVE ({blocked_text}) - move hand away",
            (left, min(frame.shape[0] - 40, bottom + 28)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            border_color,
            2,
        )
    cv2.putText(
        frame,
        f"{CONTROL_FRAME_WIDTH_MM:.0f} x {CONTROL_FRAME_HEIGHT_MM:.0f} mm control frame",
        (left, max(22, top - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )


def get_control_center_pose(initial_pose):
    """คัดลอกตำแหน่ง XYZ/R เริ่มต้นเพื่อใช้เป็นจุดอ้างอิงกลางกรอบ"""
    return tuple(initial_pose[:4])


# 5) FOLLOW/JOG: คำนวณทิศและระยะก้าวของการควบคุม XY
def get_edge_jog_directions(pixel, control_frame):
    """คืนทิศ JOG ตามขอบที่ฝ่ามือแตะ รวมการขยับทแยงที่มุม"""
    left, top, right, bottom = control_frame
    pixel_x, pixel_y = pixel
    edge_x = (right - left) * EDGE_JOG_RATIO
    edge_y = (bottom - top) * EDGE_JOG_RATIO
    directions = set()

    if pixel_x <= left + edge_x:
        directions.add("LEFT")
    elif pixel_x >= right - edge_x:
        directions.add("RIGHT")
    if pixel_y <= top + edge_y:
        directions.add("UP")
    elif pixel_y >= bottom - edge_y:
        directions.add("DOWN")
    return directions


def get_follow_step(current_pose, target_xy, blocked_directions):
    """คำนวณก้าวถัดไปให้หุ่นตามพิกัดมือ โดยจำกัดระยะต่อก้าว"""
    delta_x = target_xy[0] - current_pose[0]
    delta_y = target_xy[1] - current_pose[1]
    if abs(delta_x) <= FOLLOW_DEADBAND_MM:
        delta_x = 0.0
    elif ("UP" if delta_x > 0 else "DOWN") in blocked_directions:
        delta_x = 0.0
    if abs(delta_y) <= FOLLOW_DEADBAND_MM:
        delta_y = 0.0
    elif ("RIGHT" if delta_y > 0 else "LEFT") in blocked_directions:
        delta_y = 0.0

    distance = float(np.hypot(delta_x, delta_y))
    if distance == 0:
        return 0.0, 0.0
    step_scale = min(MAX_STEP_MM / distance, 1.0)
    return delta_x * step_scale, delta_y * step_scale


def get_follow_directions(delta_x, delta_y):
    """แปลงความคลาดเคลื่อน XY เป็นชื่อทิศเพื่อแสดงและตรวจขีดจำกัด"""
    directions = set()
    if abs(delta_x) > FOLLOW_DEADBAND_MM:
        directions.add("UP" if delta_x > 0 else "DOWN")
    if abs(delta_y) > FOLLOW_DEADBAND_MM:
        directions.add("RIGHT" if delta_y > 0 else "LEFT")
    return directions


def get_jog_step(directions, blocked_directions):
    """แปลงทิศ JOG ที่ใช้งานได้เป็นเวกเตอร์การเคลื่อนที่ XY"""
    jog_x = float(
        ("UP" in directions and "UP" not in blocked_directions)
        - ("DOWN" in directions and "DOWN" not in blocked_directions)
    )
    jog_y = float(
        ("RIGHT" in directions and "RIGHT" not in blocked_directions)
        - ("LEFT" in directions and "LEFT" not in blocked_directions)
    )
    return jog_x, jog_y


# 6) Dobot จริง: เชื่อมพอร์ต COM และตรวจว่าคำสั่งในคิวเสร็จภายในเวลาที่กำหนด
def wait_for_queued_command(device, response, timeout_sec):
    """รอจบคำสั่ง PTP ด้วย queue index และ timeout ป้องกันรอไม่สิ้นสุด"""
    expected_index = struct.unpack_from("L", response.params, 0)[0]
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if device._get_queued_cmd_current_index() >= expected_index:
            return
        time.sleep(0.05)
    raise TimeoutError(
        f"Dobot ไม่ทำคำสั่ง PTP หมายเลข {expected_index} เสร็จใน "
        f"{timeout_sec:.0f} วินาที"
    )


def connect_dobot():
    """เชื่อม Dobot จริงผ่าน COM3 และคืนตำแหน่งเริ่มต้น XYZ/R"""
    device = None
    try:
        print(f"[INFO] กำลังเชื่อมต่อ Dobot จริงที่พอร์ต {COM_PORT}...")
        device = Dobot(port=COM_PORT, verbose=False)
        device.speed(
            velocity=ROBOT_SPEED_MM_S,
            acceleration=ROBOT_SPEED_MM_S,
        )
        pose = device.pose()
        initial_pose = tuple(float(value) for value in pose[:4])
        print(
            f"[OK] เชื่อมต่อ Dobot สำเร็จ | XYZ/R="
            f"{tuple(round(value, 1) for value in initial_pose)} มม./องศา"
        )
        return device, initial_pose
    except Exception as error:
        close_error = None
        if device is not None:
            try:
                device.close()
            except Exception as cleanup_error:
                close_error = cleanup_error
        detail = f" ({close_error})" if close_error is not None else ""
        raise RuntimeError(
            f"เชื่อมต่อ/เตรียม Dobot ที่ {COM_PORT} ไม่สำเร็จ: {error}{detail}"
        ) from error


def move_dobot_home(device):
    """สั่ง Dobot กลับข้อต่อ Home ตามค่าที่ใช้ใน testB พร้อม timeout"""
    response = device._set_ptp_cmd(
        *HOME_JOINTS,
        mode=PTPMode.MOVJ_ANGLE,
        wait=False,
    )
    wait_for_queued_command(device, response, timeout_sec=30.0)


class RobotCommandWorker:
    """รันคำสั่ง PTP และหัวดูดแยกจากลูปกล้อง เพื่อไม่ให้ภาพค้าง"""

    def __init__(self, device, initial_pose):
        """สร้างคิวคำสั่งและเริ่ม worker สำหรับ Dobot จริง"""
        self.device = device
        self.condition = threading.Condition()
        self.commands = deque()
        self.busy = False
        self.stopping = False
        self.error = None
        self.current_pose = tuple(initial_pose[:4])
        self.suction_on = False
        self.gripped_object_name = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def is_idle(self):
        """ตรวจว่าคิวว่างและ worker พร้อมรับคำสั่งใหม่หรือไม่"""
        with self.condition:
            return not self.busy and not self.commands and self.error is None

    def get_current_pose(self):
        """คืนสำเนาตำแหน่งล่าสุดของหุ่นยนต์ที่ worker บันทึก"""
        with self.condition:
            return tuple(self.current_pose)

    def get_gripper_state(self):
        """คืนสถานะหัวดูดและชื่อชิ้นงานที่กำลังถูกจับ"""
        with self.condition:
            return self.suction_on, self.gripped_object_name

    def submit_move(self, pose):
        """เข้าคิว PTP แบบพิกัด XYZ เมื่อ worker ว่าง"""
        with self.condition:
            if self.busy or self.commands or self.stopping or self.error:
                return False
            self.commands.append(("move", tuple(float(value) for value in pose[:4])))
            self.condition.notify()
            return True

    def submit_suction(self, enabled):
        """เข้าคิวเปิดหรือปิดหัวดูดจริงของ Dobot"""
        with self.condition:
            if self.stopping or self.error:
                return False
            if self.commands and self.commands[-1] == ("suction", enabled):
                return True
            self.commands.append(("suction", enabled))
            self.condition.notify()
            return True

    def _run(self):
        """อ่านคำสั่งจากคิวและบันทึกข้อผิดพลาดจาก Dobot"""
        while True:
            with self.condition:
                while not self.commands and not self.stopping:
                    self.condition.wait()
                if self.stopping and not self.commands:
                    return
                command, value = self.commands.popleft()
                self.busy = True

            try:
                if command == "move":
                    response = self.device._set_ptp_cmd(
                        *value,
                        mode=PTPMode.MOVJ_XYZ,
                        wait=False,
                    )
                    wait_for_queued_command(self.device, response, timeout_sec=8.0)
                    actual_pose = tuple(float(item) for item in self.device.pose()[:4])
                else:
                    self.device.suck(value)
                    attached_name = "Dobot suction cup" if value else None
                    print(
                        "[DOBOT] เปิดหัวดูดจริง" if value
                        else "[DOBOT] ปิดหัวดูดจริง"
                    )
            except Exception as error:
                with self.condition:
                    self.error = error
                    self.commands.clear()
                    self.busy = False
                    self.stopping = True
                    self.condition.notify_all()
                return

            with self.condition:
                if command == "move":
                    self.current_pose = actual_pose
                else:
                    self.suction_on = value
                    self.gripped_object_name = attached_name
                self.busy = False
                self.condition.notify_all()

    def close(self):
        """หยุด worker และรอให้เธรดจบก่อนปิดโปรแกรม"""
        with self.condition:
            self.stopping = True
            self.commands.clear()
            self.condition.notify_all()
        self.thread.join()


# 7) การตรวจขอบเขตและหัวดูด: ป้องกันออกนอกกรอบ XY และแปลท่ามือซ้าย
def is_inside_control_frame(pose, origin_pose):
    """ตรวจว่า pose ปัจจุบันอยู่ในขอบเขตกรอบควบคุมหรือไม่"""
    offset_x = pose[0] - origin_pose[0]
    offset_y = pose[1] - origin_pose[1]
    return (
        abs(offset_x) <= CONTROL_FRAME_WIDTH_MM / 2
        and abs(offset_y) <= CONTROL_FRAME_HEIGHT_MM / 2
    )


def update_suction(worker, hand, history, previous_pose, suction_state, enabled):
    """ส่งคำสั่งหยิบเมื่อมือซ้ายกำ และปล่อยเมื่อมือซ้ายแบอย่างมั่นคง"""
    if not enabled:
        history.clear()
        return previous_pose, suction_state

    observed_pose = classify_hand_pose(hand) if hand is not None else None
    stable = stable_pose(history, observed_pose)
    if stable is None:
        return previous_pose, suction_state
    if previous_pose is None:
        return stable, suction_state
    if stable == previous_pose:
        return previous_pose, suction_state

    suction_enabled = stable == "FIST"
    if worker.submit_suction(suction_enabled):
        message = "กำมือ: เปิดหัวดูดเพื่อหยิบ" if suction_enabled else "แบมือ: ปิดหัวดูดเพื่อปล่อย"
        print(f"[GRIPPER] {message}")
        return stable, suction_enabled
    return previous_pose, suction_state
