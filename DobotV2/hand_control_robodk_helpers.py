# -*- coding: utf-8 -*-
"""RoboDK simulation helpers used only by testC ROBO.py."""

from collections import deque
import threading

from robodk import robomath
from robodk.robolink import (
    ITEM_TYPE_ROBOT,
    ITEM_TYPE_TOOL,
    Robolink,
    TargetReachError,
)

from hand_control_helpers import (
    CONTROL_FRAME_HEIGHT_MM,
    CONTROL_FRAME_WIDTH_MM,
    FOLLOW_DEADBAND_MM,
    LEFT_HAND_LABEL,
    MAX_STEP_MM,
    MAX_Z_STEP_MM,
    MOVE_INTERVAL_SEC,
    POSE_HISTORY_FRAMES,
    RIGHT_HAND_LABEL,
    ROBOT_SPEED_MM_S,
    WINDOW_NAME,
    classify_vertical_gesture,
    create_landmarker,
    draw_control_frame,
    draw_hand,
    find_hand,
    get_control_frame,
    get_edge_jog_directions,
    get_follow_directions,
    get_jog_step,
    get_palm_center,
    open_camera,
    stable_pose,
    update_suction,
)


def connect_robodk():
    """Connect to the active RoboDK station and return its robot and tool."""
    rdk = Robolink()
    robot = rdk.Item("", ITEM_TYPE_ROBOT)
    if not robot.Valid():
        raise RuntimeError("ไม่พบหุ่นยนต์ใน RoboDK Station ที่เปิดอยู่")
    tool = robot.getLink(ITEM_TYPE_TOOL)
    if not tool.Valid():
        raise RuntimeError("ไม่พบ Tool ที่เชื่อมอยู่กับหุ่นยนต์ใน RoboDK")
    return rdk, robot, tool, None


def get_control_center_pose(initial_pose):
    """Return a separate RoboDK pose matrix for the center of the control frame."""
    return robomath.Mat(initial_pose)


def pixel_to_robot(pixel, control_frame, origin_pose):
    """Map a camera pixel to an XY target relative to the RoboDK start pose."""
    left, top, right, bottom = control_frame
    pixel_x, pixel_y = pixel
    if not (left <= pixel_x <= right and top <= pixel_y <= bottom):
        return None

    image_x = (pixel_x - left) / (right - left)
    image_y = (pixel_y - top) / (bottom - top)
    robot_x = origin_pose[0, 3] + (0.5 - image_y) * CONTROL_FRAME_HEIGHT_MM
    robot_y = origin_pose[1, 3] + (image_x - 0.5) * CONTROL_FRAME_WIDTH_MM
    return float(robot_x), float(robot_y)


def get_follow_step(current_pose, target_pose, blocked_directions):
    """Calculate a bounded XY step toward a RoboDK target pose."""
    delta_x = target_pose[0] - current_pose[0, 3]
    delta_y = target_pose[1] - current_pose[1, 3]
    if abs(delta_x) <= FOLLOW_DEADBAND_MM:
        delta_x = 0.0
    elif ("UP" if delta_x > 0 else "DOWN") in blocked_directions:
        delta_x = 0.0
    if abs(delta_y) <= FOLLOW_DEADBAND_MM:
        delta_y = 0.0
    elif ("RIGHT" if delta_y > 0 else "LEFT") in blocked_directions:
        delta_y = 0.0

    distance = float((delta_x ** 2 + delta_y ** 2) ** 0.5)
    if distance == 0:
        return 0.0, 0.0
    step_scale = min(MAX_STEP_MM / distance, 1.0)
    return delta_x * step_scale, delta_y * step_scale


def is_inside_control_frame(pose, origin_pose):
    """Check that the robot TCP is within the configured XY control envelope."""
    return (
        abs(pose[0, 3] - origin_pose[0, 3]) <= CONTROL_FRAME_WIDTH_MM / 2
        and abs(pose[1, 3] - origin_pose[1, 3]) <= CONTROL_FRAME_HEIGHT_MM / 2
    )


def is_robot_pose_reachable(robot, pose):
    """Use RoboDK inverse kinematics to reject unreachable simulated targets."""
    solution = robot.SolveIK(pose)
    return bool(solution.list())


class RobotCommandWorker:
    """Process simulated movement and gripper commands outside the camera loop."""

    def __init__(self, robot, tool, initial_pose):
        self.robot = robot
        self.tool = tool
        self.condition = threading.Condition()
        self.commands = deque()
        self.busy = False
        self.stopping = False
        self.error = None
        self.rejected_move = None
        self.current_pose = robomath.Mat(initial_pose)
        self.suction_on = False
        self.gripped_object_name = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def is_idle(self):
        with self.condition:
            return not self.busy and not self.commands and self.error is None

    def get_current_pose(self):
        with self.condition:
            return robomath.Mat(self.current_pose)

    def get_gripper_state(self):
        with self.condition:
            return self.suction_on, self.gripped_object_name

    def pop_rejected_move(self):
        """Return and clear a MoveL target rejected by RoboDK."""
        with self.condition:
            target = self.rejected_move
            self.rejected_move = None
            return robomath.Mat(target) if target is not None else None

    def submit_move(self, pose):
        with self.condition:
            if self.busy or self.commands or self.stopping or self.error:
                return False
            self.commands.append(("move", robomath.Mat(pose)))
            self.condition.notify()
            return True

    def submit_suction(self, enabled):
        with self.condition:
            if self.stopping or self.error:
                return False
            if self.commands and self.commands[-1] == ("suction", enabled):
                return True
            self.commands.append(("suction", enabled))
            self.condition.notify()
            return True

    def _run(self):
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
                    self.robot.MoveL(value, blocking=True)
                    actual_pose = self.robot.Pose()
                    attached_name = None
                elif value:
                    attached = self.tool.AttachClosest()
                    attached_name = attached.Name() if attached.Valid() else None
                    actual_pose = None
                else:
                    self.tool.DetachAll()
                    attached_name = None
                    actual_pose = None
            except TargetReachError as error:
                if command == "move":
                    with self.condition:
                        self.rejected_move = robomath.Mat(value)
                        self.busy = False
                        self.condition.notify_all()
                    continue
                with self.condition:
                    self.error = error
                    self.commands.clear()
                    self.busy = False
                    self.stopping = True
                    self.condition.notify_all()
                return
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
        with self.condition:
            self.stopping = True
            self.commands.clear()
            self.condition.notify_all()
        self.thread.join()


__all__ = [
    "CONTROL_FRAME_HEIGHT_MM",
    "CONTROL_FRAME_WIDTH_MM",
    "LEFT_HAND_LABEL",
    "MAX_STEP_MM",
    "MAX_Z_STEP_MM",
    "MOVE_INTERVAL_SEC",
    "POSE_HISTORY_FRAMES",
    "RIGHT_HAND_LABEL",
    "ROBOT_SPEED_MM_S",
    "RobotCommandWorker",
    "WINDOW_NAME",
    "classify_vertical_gesture",
    "connect_robodk",
    "create_landmarker",
    "draw_control_frame",
    "draw_hand",
    "find_hand",
    "get_control_center_pose",
    "get_control_frame",
    "get_edge_jog_directions",
    "get_follow_directions",
    "get_follow_step",
    "get_jog_step",
    "get_palm_center",
    "is_inside_control_frame",
    "is_robot_pose_reachable",
    "open_camera",
    "pixel_to_robot",
    "stable_pose",
    "update_suction",
]
