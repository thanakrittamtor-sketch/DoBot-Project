# -*- coding: utf-8 -*-
"""ไฟล์หลักสำหรับควบคุม Dobot ใน RoboDK ด้วยภาพสดจากกล้อง

ลำดับหลัก: ตั้งค่า Station -> เปิดกล้อง -> วนตรวจมือ/ส่งคำสั่ง -> กลับ Home
ฟังก์ชันตรวจจับมือและคำสั่ง RoboDK อยู่ใน hand_control_helpers.py

ปุ่ม: Space เปิด/ปิดการควบคุม, T สลับ FOLLOW/JOG, Q หรือ Esc ออกจากโปรแกรม
"""

from collections import deque
import sys
import time

import cv2
import mediapipe as mp
import numpy as np
from robodk import robomath

from hand_control_robodk_helpers import (
    CONTROL_FRAME_HEIGHT_MM,
    CONTROL_FRAME_WIDTH_MM,
    LEFT_HAND_LABEL,
    MAX_STEP_MM,
    MAX_Z_STEP_MM,
    MOVE_INTERVAL_SEC,
    POSE_HISTORY_FRAMES,
    RIGHT_HAND_LABEL,
    ROBOT_SPEED_MM_S,
    RobotCommandWorker,
    WINDOW_NAME,
    classify_vertical_gesture,
    connect_robodk,
    create_landmarker,
    draw_control_frame,
    draw_hand,
    find_hand,
    get_control_center_pose,
    get_control_frame,
    get_edge_jog_directions,
    get_follow_directions,
    get_follow_step,
    get_jog_step,
    get_palm_center,
    is_inside_control_frame,
    is_robot_pose_reachable,
    open_camera,
    pixel_to_robot,
    stable_pose,
    update_suction,
)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

# 8) main(): เริ่มการเชื่อมต่อและควบคุมหุ่นยนต์จากภาพสดทีละเฟรม
def main():
    """เริ่มกล้องและ RoboDK วนตรวจมือ/ส่งคำสั่งจนกว่าจะออกจากโปรแกรม"""
    cap = None
    landmarker = None
    robot = None
    worker = None
    home_joints = None
    try:
        _, robot, tool, _ = connect_robodk()
        robot.setSpeed(ROBOT_SPEED_MM_S)
        home_joints = robot.JointsHome()
        initial_pose = robot.Pose()
        print(
            f"[OK] เชื่อมต่อ RoboDK Robot '{robot.Name()}' | "
            f"TCP X={initial_pose[0, 3]:.1f}, Y={initial_pose[1, 3]:.1f}, "
            f"Z={initial_pose[2, 3]:.1f} มม."
        )
        print(
            "[SAFE] ควบคุม X/Y และ Z; ทดสอบ IK ก่อนสั่งทุกจุด "
            "เพื่อข้ามจุดเอื้อมไม่ถึง"
        )

        cap = open_camera()
        landmarker = create_landmarker()
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError("อ่านภาพจากกล้องไม่สำเร็จ")
        control_frame = get_control_frame(frame.shape)
        origin_pose = get_control_center_pose(initial_pose)
        print(
            f"[FRAME] ขนาด {CONTROL_FRAME_WIDTH_MM:.0f} x "
            f"{CONTROL_FRAME_HEIGHT_MM:.0f} มม. | "
            f"กลางกรอบ = X{origin_pose[0, 3]:.1f}, "
            f"Y{origin_pose[1, 3]:.1f} มม."
        )
        print(
            "[FRAME] ไม่ต้องคาลิเบรต; กลางภาพคือพิกัดอ้างอิง (0,0) "
            f"กรอบ X/Y ด้านละ {CONTROL_FRAME_WIDTH_MM / 2:.0f} มม."
        )
        print(
            "[INFO] มือขวาตามตำแหน่งฝ่ามือเพื่อควบคุม X/Y (กด T สลับ jog ขอบ) "
            "| ชู V=Z ขึ้น, กำมือ=Z ลง "
            "| มือซ้ายกำ=หยิบ, แบ=ปล่อย"
        )
        print(f"[INFO] ปิดโปรแกรมจะกลับ Home joints {home_joints.list()}")
        print("[INFO] Space=เปิด/ปิดการควบคุม | T=สลับตามมือ/jog | Q=ออก")
        worker = RobotCommandWorker(robot, tool, initial_pose)

        enabled = False
        movement_mode = "FOLLOW"
        suction_state = None
        previous_gripper_pose = None
        gripper_history = deque(maxlen=POSE_HISTORY_FRAMES)
        vertical_history = deque(maxlen=POSE_HISTORY_FRAMES)
        last_move_time = 0.0
        blocked_directions = set()

        # 8.1 วนลูปหลัก: รับภาพ ตรวจมือ คำนวณการเคลื่อน และแสดงสถานะ
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError("สัญญาณภาพจากกล้องขาดหาย")

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = landmarker.detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            )
            right_hand = find_hand(result, RIGHT_HAND_LABEL)
            left_hand = find_hand(result, LEFT_HAND_LABEL)
            right_pixel = None
            jog_directions = set()
            if right_hand is not None:
                right_pixel = get_palm_center(right_hand, frame.shape)
                jog_directions = get_edge_jog_directions(
                    right_pixel, control_frame
                )
            target_xy = (
                pixel_to_robot(right_pixel, control_frame, origin_pose)
                if right_pixel is not None
                else None
            )

            vertical_gesture = None
            if enabled:
                observed_vertical_gesture = (
                    classify_vertical_gesture(right_hand)
                    if right_hand is not None
                    else None
                )
                vertical_gesture = stable_pose(
                    vertical_history, observed_vertical_gesture
                )
            else:
                vertical_history.clear()

            vertical_direction = (
                "Z_UP" if vertical_gesture == "UP"
                else "Z_DOWN" if vertical_gesture == "DOWN"
                else None
            )
            current_pose = worker.get_current_pose()
            if movement_mode == "FOLLOW" and target_xy is not None:
                follow_dx = target_xy[0] - current_pose[0, 3]
                follow_dy = target_xy[1] - current_pose[1, 3]
                xy_directions = get_follow_directions(follow_dx, follow_dy)
            elif movement_mode == "JOG":
                follow_dx = 0.0
                follow_dy = 0.0
                xy_directions = set(jog_directions)
            else:
                follow_dx = 0.0
                follow_dy = 0.0
                xy_directions = set()

            active_directions = set(xy_directions)
            if vertical_direction is not None:
                active_directions.add(vertical_direction)
            blocked_directions.intersection_update(active_directions)
            draw_control_frame(
                frame, control_frame, blocked_directions
            )
            if right_hand is not None:
                draw_hand(frame, right_hand, (60, 240, 60))
                marker_color = (0, 255, 255) if target_xy is not None else (0, 0, 255)
                cv2.circle(
                    frame, tuple(map(int, right_pixel)), 8, marker_color, -1
                )
            if left_hand is not None:
                draw_hand(frame, left_hand, (255, 120, 80))

            previous_gripper_pose, suction_state = update_suction(
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
                and active_directions
                and worker.is_idle()
                and now - last_move_time >= MOVE_INTERVAL_SEC
            ):
                current_pose = worker.get_current_pose()
                next_pose = robomath.Mat(current_pose)
                has_move = False
                jog_x = 0.0
                jog_y = 0.0
                if movement_mode == "FOLLOW" and target_xy is not None:
                    jog_x, jog_y = get_follow_step(
                        current_pose, target_xy, blocked_directions
                    )
                    next_pose[0, 3] += jog_x
                    next_pose[1, 3] += jog_y
                    has_move = jog_x != 0.0 or jog_y != 0.0
                else:
                    jog_x, jog_y = get_jog_step(
                        jog_directions, blocked_directions
                    )

                    planar_length = float(np.hypot(jog_x, jog_y))
                    if planar_length > 0:
                        next_pose[0, 3] += jog_x * MAX_STEP_MM / planar_length
                        next_pose[1, 3] += jog_y * MAX_STEP_MM / planar_length
                        has_move = True

                if (
                    vertical_gesture == "UP"
                    and "Z_UP" not in blocked_directions
                ):
                    next_pose[2, 3] += MAX_Z_STEP_MM
                    has_move = True
                elif (
                    vertical_gesture == "DOWN"
                    and "Z_DOWN" not in blocked_directions
                ):
                    next_pose[2, 3] -= MAX_Z_STEP_MM
                    has_move = True

                if has_move:
                    if is_robot_pose_reachable(robot, next_pose):
                        if worker.submit_move(next_pose):
                            last_move_time = now
                    else:
                        attempted_directions = set()
                        if jog_x < 0:
                            attempted_directions.add("DOWN")
                        elif jog_x > 0:
                            attempted_directions.add("UP")
                        if jog_y < 0:
                            attempted_directions.add("LEFT")
                        elif jog_y > 0:
                            attempted_directions.add("RIGHT")
                        if (
                            vertical_gesture == "UP"
                            and "Z_UP" not in blocked_directions
                        ):
                            attempted_directions.add("Z_UP")
                        elif (
                            vertical_gesture == "DOWN"
                            and "Z_DOWN" not in blocked_directions
                        ):
                            attempted_directions.add("Z_DOWN")
                        newly_blocked = attempted_directions - blocked_directions
                        if newly_blocked:
                            blocked_directions.update(newly_blocked)
                            print(
                                "[SAFE] ปลายแขนไปต่อไม่ได้ "
                                f"({', '.join(sorted(newly_blocked))}); "
                                "กรอบจะแดงจนกว่าจะเปลี่ยนทิศทางมือ/ออกจากขอบ"
                            )

            rejected_target = worker.pop_rejected_move()
            if rejected_target is not None:
                current_pose = worker.get_current_pose()
                rejected_dx = rejected_target[0, 3] - current_pose[0, 3]
                rejected_dy = rejected_target[1, 3] - current_pose[1, 3]
                rejected_dz = rejected_target[2, 3] - current_pose[2, 3]
                rejected_directions = set()
                if abs(rejected_dx) > 0.1:
                    rejected_directions.add("UP" if rejected_dx > 0 else "DOWN")
                if abs(rejected_dy) > 0.1:
                    rejected_directions.add("RIGHT" if rejected_dy > 0 else "LEFT")
                if abs(rejected_dz) > 0.1:
                    rejected_directions.add("Z_UP" if rejected_dz > 0 else "Z_DOWN")
                newly_blocked = rejected_directions - blocked_directions
                blocked_directions.update(rejected_directions)
                print(
                    "[SAFE] RoboDK ปฏิเสธเส้นทาง MoveL ไปยัง "
                    f"X={rejected_target[0, 3]:.1f}, "
                    f"Y={rejected_target[1, 3]:.1f}, "
                    f"Z={rejected_target[2, 3]:.1f} มม. "
                    f"จึงบล็อกทิศ {', '.join(sorted(newly_blocked)) or 'เดิม'} "
                    "ชั่วคราว; ขยับมือออกจากทิศนั้นเพื่อปลดล็อก"
                )

            if worker.error is not None:
                raise RuntimeError(
                    f"ส่งคำสั่ง RoboDK ไม่สำเร็จ: {worker.error}"
                ) from worker.error

            mode = "ON" if enabled else "OFF"
            mode_color = (0, 0, 255) if enabled else (0, 220, 255)
            cv2.putText(
                frame,
                f"CONTROL: {mode} (Space)",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                mode_color,
                2,
            )
            if target_xy is None:
                target_text = "มือขวาอยู่นอกกรอบ/ไม่พบมือ"
                target_color = (0, 0, 255)
            else:
                target_text = f"Hand XY: ({target_xy[0]:.1f}, {target_xy[1]:.1f}) mm"
                target_color = (60, 240, 60)
            cv2.putText(
                frame,
                target_text,
                (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                target_color,
                2,
            )

            current_pose = worker.get_current_pose()
            suction_on, gripped_object_name = worker.get_gripper_state()
            suction_text = "ON" if suction_on else "OFF"
            if gripped_object_name:
                suction_text += f" ({gripped_object_name})"
            cv2.putText(
                frame,
                f"Robot XY: ({current_pose[0, 3]:.1f}, {current_pose[1, 3]:.1f}) "
                f"| Gripper: {suction_text}",
                (10, 88),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                2,
            )
            jog_text = "+".join(sorted(jog_directions)) or "HOLD"
            vertical_text = vertical_gesture or "HOLD"
            cv2.putText(
                frame,
                f"XY: {movement_mode} (T toggle) | Edge: {jog_text} | Z: {vertical_text}",
                (10, 118),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                "Space=Enable/Disable | T=Follow/Jog | Q=Quit",
                (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )
            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("t"), ord("T")):
                movement_mode = "JOG" if movement_mode == "FOLLOW" else "FOLLOW"
                blocked_directions.clear()
                print(
                    "[CONTROL] โหมดตามมือ" if movement_mode == "FOLLOW"
                    else "[CONTROL] โหมด jog ที่ขอบกรอบ"
                )
            if key == ord(" "):
                if not enabled:
                    current_pose = worker.get_current_pose()
                    if not is_inside_control_frame(current_pose, origin_pose):
                        print(
                            "[WARN] ตำแหน่งปลายแขนปัจจุบันอยู่นอกกรอบควบคุม "
                            "จึงยังเปิดควบคุมไม่ได้"
                        )
                        continue
                    suction_state, _ = worker.get_gripper_state()
                    previous_gripper_pose = (
                        "FIST" if suction_state else "OPEN"
                    )
                enabled = not enabled
                print("[CONTROL] เปิดการควบคุม" if enabled else "[CONTROL] ปิดการควบคุม")

    finally:
        # 9) ปิดกล้อง/หน้าต่าง และคืนหุ่นยนต์ไปยัง Home ในทุกกรณีที่ออก
        try:
            if worker is not None:
                worker.close()
        finally:
            try:
                if robot is not None and home_joints is not None:
                    print("[HOME] กำลังกลับ Home ตามที่ตั้งไว้ใน RoboDK...")
                    robot.MoveJ(home_joints, blocking=True)
                    print("[HOME] กลับ Home สำเร็จ")
            finally:
                if cap is not None:
                    cap.release()
                if landmarker is not None:
                    landmarker.close()
                cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
