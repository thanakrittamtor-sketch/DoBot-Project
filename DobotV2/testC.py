# -*- coding: utf-8 -*-
"""ควบคุม Dobot Magician จริงจากภาพสดของกล้อง

ลำดับหลัก: เชื่อมต่อ COM3 -> เปิดกล้อง -> ควบคุมมือ -> กลับ Home เมื่อออก
ฟังก์ชันตรวจจับมือและคำสั่ง Dobot อยู่ใน hand_control_helpers.py

ปุ่ม: Space เปิด/ปิดการควบคุม, T สลับ FOLLOW/JOG, Q หรือ Esc ออกจากโปรแกรม
"""

from collections import deque
import sys
import time

import cv2
import mediapipe as mp
import numpy as np

from hand_control_helpers import (
    CONTROL_FRAME_HEIGHT_MM,
    CONTROL_FRAME_WIDTH_MM,
    HOME_JOINTS,
    LEFT_HAND_LABEL,
    MAX_STEP_MM,
    MOVE_INTERVAL_SEC,
    POSE_HISTORY_FRAMES,
    RIGHT_HAND_LABEL,
    RobotCommandWorker,
    WINDOW_NAME,
    connect_dobot,
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
    move_dobot_home,
    open_camera,
    pixel_to_robot,
    update_suction,
)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

# 8) main(): เชื่อมต่อ Dobot จริงและควบคุมจากภาพสดทีละเฟรม
def main():
    """เริ่มกล้องและ Dobot จริง วนตรวจมือ/ส่งคำสั่งจนกว่าจะออก"""
    cap = None
    landmarker = None
    device = None
    worker = None
    cleanup_errors = []

    def cleanup_step(label, action):
        try:
            action()
        except Exception as error:
            cleanup_errors.append(f"{label}: {error}")

    try:
        device, initial_pose = connect_dobot()
        print(
            f"[OK] Dobot จริงเริ่มต้นที่ X={initial_pose[0]:.1f}, "
            f"Y={initial_pose[1]:.1f}, Z={initial_pose[2]:.1f} มม."
        )
        print(
            "[SAFE] จำกัด X/Y ในกรอบ ±100 มม. จากตำแหน่งเริ่มต้น; "
            "ปิดควบคุม Z ไว้"
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
            f"กลางกรอบ = X{origin_pose[0]:.1f}, "
            f"Y{origin_pose[1]:.1f} มม."
        )
        print(
            "[FRAME] ไม่ต้องคาลิเบรต; กลางภาพคือพิกัดอ้างอิง (0,0) "
            f"กรอบ X/Y ด้านละ {CONTROL_FRAME_WIDTH_MM / 2:.0f} มม."
        )
        print(
            "[INFO] มือขวาตามตำแหน่งฝ่ามือเพื่อควบคุม X/Y (กด T สลับ jog ขอบ) "
            "| ปิดควบคุม Z ไว้ "
            "| มือซ้ายกำ=หยิบ, แบ=ปล่อย"
        )
        print(f"[INFO] ปิดโปรแกรมจะกลับ Home joints {list(HOME_JOINTS)}")
        print("[INFO] Space=เปิด/ปิดการควบคุม | T=สลับตามมือ/jog | Q=ออก")
        worker = RobotCommandWorker(device, initial_pose)

        enabled = False
        movement_mode = "FOLLOW"
        suction_state = None
        previous_gripper_pose = None
        gripper_history = deque(maxlen=POSE_HISTORY_FRAMES)
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

            current_pose = worker.get_current_pose()
            if movement_mode == "FOLLOW" and target_xy is not None:
                follow_dx = target_xy[0] - current_pose[0]
                follow_dy = target_xy[1] - current_pose[1]
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
                next_pose = list(current_pose)
                has_move = False
                jog_x = 0.0
                jog_y = 0.0
                if movement_mode == "FOLLOW" and target_xy is not None:
                    jog_x, jog_y = get_follow_step(
                        current_pose, target_xy, blocked_directions
                    )
                    next_pose[0] += jog_x
                    next_pose[1] += jog_y
                    has_move = jog_x != 0.0 or jog_y != 0.0
                else:
                    jog_x, jog_y = get_jog_step(
                        jog_directions, blocked_directions
                    )

                    planar_length = float(np.hypot(jog_x, jog_y))
                    if planar_length > 0:
                        next_pose[0] += jog_x * MAX_STEP_MM / planar_length
                        next_pose[1] += jog_y * MAX_STEP_MM / planar_length
                        has_move = True

                if has_move:
                    if is_inside_control_frame(next_pose, origin_pose):
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
                        newly_blocked = attempted_directions - blocked_directions
                        if newly_blocked:
                            blocked_directions.update(newly_blocked)
                            print(
                                "[SAFE] ถึงขอบเขต XY ที่กำหนด "
                                f"({', '.join(sorted(newly_blocked))}); "
                                "ขยับมือกลับเข้าด้านในเพื่อยกเลิกการเตือน"
                            )

            if worker.error is not None:
                raise RuntimeError(
                    f"ส่งคำสั่ง Dobot ไม่สำเร็จ: {worker.error}"
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
                f"Robot XY: ({current_pose[0]:.1f}, {current_pose[1]:.1f}) "
                f"| Gripper: {suction_text}",
                (10, 88),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                2,
            )
            jog_text = "+".join(sorted(jog_directions)) or "HOLD"
            cv2.putText(
                frame,
                f"XY: {movement_mode} (T toggle) | Edge: {jog_text} | Z: DISABLED",
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
        # 9) หยุด worker, ปิดหัวดูด, กลับ Home และปิดพอร์ต/กล้อง
        if worker is not None:
            cleanup_step("หยุด worker", worker.close)
        if device is not None:
            cleanup_step("ปิดหัวดูด", lambda: device.suck(False))
            print("[HOME] กำลังกลับ Home ของ Dobot...")
            cleanup_step("กลับ Home", lambda: move_dobot_home(device))
            cleanup_step("ปิดพอร์ต Dobot", device.close)
        if cap is not None:
            cleanup_step("ปิดกล้อง", cap.release)
        if landmarker is not None:
            cleanup_step("ปิด MediaPipe", landmarker.close)
        cleanup_step("ปิดหน้าต่าง", cv2.destroyAllWindows)
        if cleanup_errors:
            print("[CLEANUP ERROR] " + " | ".join(cleanup_errors))


if __name__ == "__main__":
    main()
