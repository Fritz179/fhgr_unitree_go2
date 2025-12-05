#!/usr/bin/env python3.11

import time
from collections import deque

import cv2 as cv
from ultralytics import YOLO

import mediapipe as mp
import numpy as np

from pathlib import Path

def draw_footer(img, lines):
    """
    Adds a footer below the image showing one or more lines of text.
    """
    footer = np.full((len(lines) * 25 + 40, img.shape[1], 3), (0, 0, 0), dtype=np.uint8)

    y = 25
    for (on, key, text) in lines:
        color = (0, 255, 0) if on else (0, 0, 255)

        first = f"[{key}]"
        cv.putText(footer, first, (5, y),
            cv.FONT_HERSHEY_DUPLEX, 0.9, color, 1, cv.LINE_4)

        cv.putText(footer, text, (70, y),
           cv.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 1, cv.LINE_4)
        
        y += 30

    return np.vstack((img, footer))


class AIClient:
    def __init__(self):
        # Configuration
        self.imgsz = 640
        self.conf = 0.25
        self.device = 0 

        # Load models
        THIS_DIR = Path(__file__).resolve().parent

        self.model = YOLO(THIS_DIR / "hagrid.pt")
        self.keypoints_model = YOLO(THIS_DIR / "hand_keypoints.pt")

        # Mediapipe Hands setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Which connections to draw for hand landmarks
        self.hand_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17),
        ]

        # FPS smoothing
        self.t0 = time.time()
        self.fps_hist = deque(maxlen=30)

        print("Press 'q' to quit.")

        self.draw_landmark = True
        self.draw_hand_direction = True
        self.draw_handgrid = True
        self.draw_keypoints = False
        self.draw_framerate = False
        self.draw_instructions = True
        self.draw_action = True

        self.active_pose_counts = [0] * len(self.model.names)


    def draw_hand_landmarks(self, drawings, keypoints):
        if drawings is None:
            return

        for start, end in self.hand_connections:
            if start < len(keypoints) and end < len(keypoints):
                x1, y1 = map(int, keypoints[start][:2])
                x2, y2 = map(int, keypoints[end][:2])
                cv.line(drawings, (x1, y1), (x2, y2), (0, 255, 0), 2)

        for x, y, _ in keypoints:
            cv.circle(drawings, (int(x), int(y)), 4, (0, 0, 255), -1)


    def get_index_direction(self, keypoints, drawings, frame_shape, hand_width):
        if not keypoints or frame_shape is None or len(frame_shape) < 2:
            return None

        INDEX_BASE = 5
        INDEX_TIP = 8

        if len(keypoints) <= INDEX_TIP:
            return None

        base_x, base_y, base_z = keypoints[INDEX_BASE]
        tip_x, tip_y, tip_z = keypoints[INDEX_TIP]

        x = tip_x - base_x
        y = tip_y - base_y
        z = tip_z - base_z

        direction_length = np.sqrt(x*x + y*y + z*z*2)  # Weight z more for depth
        if direction_length > 0:
            # Normalize and scale for consistent visual length
            arrow_scale = 100  # Fixed arrow length in pixels
            x = (x / direction_length) * arrow_scale
            y = (y / direction_length) * arrow_scale
            z = (z / direction_length) * arrow_scale

            def slow_exp(x, k=5):
                sign = 1 if x >= 0 else -1
                x = abs(x)
                t = x / 100.0
                return 100 * (np.exp(k * t) - 1) / (np.exp(k) - 1) * sign

            x = slow_exp(x)
            y = slow_exp(y)
            z = slow_exp(z)
            
            # print(f"Index finger direction: X={norm_x:.3f}, Y={norm_y:.3f}, Z={norm_z:.3f}")


        if drawings is not None:
            start_point = (int(tip_x), int(tip_y))
            end_point = (int(tip_x + (tip_x - base_x)), int(tip_y + (tip_y - base_y)))
            
            # Draw main arrow
            cv.arrowedLine(drawings, start_point, end_point, (255, 0, 0), 3)
            
            # Add text label (white on black, same style as footer)
            label = f"X :{x:.0f}, Y :{y:.0f}"
            label_size = cv.getTextSize(label, cv.FONT_HERSHEY_DUPLEX, 1, 1)[0]
            cv.rectangle(drawings, (end_point[0] + 5, end_point[1] - label_size[1] - 15), 
                        (end_point[0] + 15 + label_size[0], end_point[1] - 5), (0, 0, 0), -1)
            cv.putText(drawings, label, (end_point[0] + 10, end_point[1] - 10), 
                    cv.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1, cv.LINE_4)

        h, w = frame_shape[:2]
        if w == 0 or h == 0:
            return None

        return ((float(x), float(y), float(z)), (float(tip_x / w), float(tip_y / h), float(hand_width / w)))


    def run_landmarks(self, original, drawings, box):
        rgb = cv.cvtColor(original, cv.COLOR_BGR2RGB)

        img_h, img_w = original.shape[:2]
        x1, y1, x2, y2 = box

        x1 = max(0, min(int(x1), img_w))
        x2 = max(0, min(int(x2), img_w))
        y1 = max(0, min(int(y1), img_h))
        y2 = max(0, min(int(y2), img_h))

        if x2 <= x1 or y2 <= y1:
            return None

        crop = rgb[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        hands_result = self.hands.process(crop)
        if not hands_result.multi_hand_landmarks:
            return None

        hand_landmarks = hands_result.multi_hand_landmarks[0]
        keypoints = []
        for lm in hand_landmarks.landmark:
            abs_x = x1 + lm.x * (x2 - x1)
            abs_y = y1 + lm.y * (y2 - y1)
            keypoints.append((abs_x, abs_y, lm.z))

        if drawings is not None:
            self.draw_hand_landmarks(drawings, keypoints)

        return keypoints
        


    # Handgrid
    def run_handgrid(self, original, drawings):
        results = self.model.predict(
            original,
            imgsz=self.imgsz,
            conf=self.conf,
            device=self.device,
            verbose=False
        )

        detections = []
        best_direction = None
        best_score = -1.0

        if results and len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                name = self.model.names.get(cls_id, str(cls_id))
                score = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                detections.append(cls_id)
                
                if drawings is not None:
                    # Draw bounding box
                    cv.rectangle(drawings, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw label with confidence (white on black, same style as footer)
                    label = f"{name} {score:.2f}"
                    label_size = cv.getTextSize(label, cv.FONT_HERSHEY_DUPLEX, 1, 1)[0]
                    cv.rectangle(drawings, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), (0, 0, 0), -1)
                    cv.putText(drawings, label, (x1, y1 - 5), cv.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1, cv.LINE_4)

                keypoints = self.run_landmarks(original, drawings if self.draw_landmark else None, (x1, y1, x2, y2))
                if keypoints:
                    hand_width = x2 - x1
                    direction = self.get_index_direction(
                        keypoints,
                        drawings if self.draw_hand_direction else None,
                        original.shape,
                        hand_width
                    )

                    if direction and score > best_score:
                        best_direction = direction
                        best_score = score

        # Get all recognized poses
        current_poses = []
        for cls_id in detections:
            current_poses.append(cls_id)

        # Remove non reaccuring poses from active counts
        for pose in range(len(self.active_pose_counts)):
            if not pose in current_poses:
                self.active_pose_counts[pose] = 0

        # Increase count for current poses
        for pose in current_poses:
            self.active_pose_counts[pose] += 1

        active_poses = []
        for pose in range(len(self.active_pose_counts)):
            if self.active_pose_counts[pose] >= 5:  # require 5 consecutive frames
                active_poses.append(self.model.names[pose])

        # For now just return the first active pose
        pose = active_poses[0] if len(active_poses) > 0 else None
        pose = None if pose == "no_gesture" else pose

        return pose, best_direction


    # hand keypoints
    def run_hand_keypoints(self, original, drawings):
        results = self.keypoints_model.predict(
            original,
            imgsz=self.imgsz,
            conf=self.conf,
            device=self.device,
            verbose=False
        )

        if results and len(results) > 0:
            for keypoints in results[0].keypoints:
                # keypoints.xy contains the x,y coordinates for each keypoint
                kpts = keypoints.xy[0]  # Get keypoints for first detection
                
                # Draw keypoints as circles
                for i, (x, y) in enumerate(kpts):
                    if x > 0 and y > 0:  # Only draw if keypoint is visible
                        cv.circle(drawings, (int(x), int(y)), 8, (0, 0, 255), -1)
                        # small black background + white index to match footer style
                        label = str(i)

                        cv.putText(drawings, label, (int(x) + 6, int(y) - 4),
                                cv.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1, cv.LINE_4)


    def draw_fps(self, drawings):
        t1 = time.time()
        fps = 1.0 / (t1 - self.t0)
        self.fps_hist.append(fps)
        fps_avg = sum(self.fps_hist) / len(self.fps_hist)

        text = f"FPS: {fps_avg:.1f}"
        text_size = cv.getTextSize(text, cv.FONT_HERSHEY_DUPLEX, 1, 1)[0]
        cv.rectangle(drawings, (0, 0), (text_size[0] + 16, text_size[1] + 16), (0, 0, 0), -1)
        cv.putText(drawings, text, (8, text_size[1] + 8), cv.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1, cv.LINE_4)


    def update(self, original):
        self.t0 = time.time()

        # draw on a copy of the original frame
        drawings = original.copy()

        # Handgrid detection first (returns pose and best direction incl. width)
        pose, index = self.run_handgrid(original, drawings if (self.draw_handgrid or self.draw_landmark or self.draw_hand_direction) else None)

        """ Available Poses:
            grabbing, grip, holy, point, call, three3, 
            timeout, xsign, hand_heart, hand_heart2, little_finger, 
            middle_finger, take_picture, dislike, fist, four, 
            like, mute, ok, one, palm, peace, peace_inverted, 
            rock, stop, stop_inverted, three, three2, two_up, 
            two_up_inverted, three_gun, thumb_index, thumb_index2, no_gesture
        """

        if self.draw_framerate:
            self.draw_fps(drawings)

        # Hand keypoints detection
        if self.draw_keypoints:
            self.run_hand_keypoints(original, drawings)


        key = cv.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:
            return "quit", index, drawings
        elif key == ord('y'):
            self.draw_landmark = not self.draw_landmark
        elif key == ord('x'):
            self.draw_hand_direction = not self.draw_hand_direction
        elif key == ord('c'):
            self.draw_handgrid = not self.draw_handgrid
        elif key == ord('v'):
            self.draw_keypoints = not self.draw_keypoints
        elif key == ord('b'):
            self.draw_framerate = not self.draw_framerate
        elif key == ord('n'):
            self.draw_instructions = not self.draw_instructions
        elif key == ord('m'):
            self.draw_action = not self.draw_action
        elif pose == None and key != 255:
            pose = chr(key)

        
        if self.draw_action:
            text = f"Pose: {pose}"
            size = cv.getTextSize(text, cv.FONT_HERSHEY_DUPLEX, 1, 1)[0]

            # bottom-left placement, black background, white text (footer style)
            cv.rectangle(drawings, (0, drawings.shape[0]-size[1] - 35), 
                        (size[0] + 20, drawings.shape[0] - 10), (0,0,0), -1)
            
            cv.putText(drawings, text, (10, drawings.shape[0]-20), 
                    cv.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1, cv.LINE_4)
            
        if self.draw_instructions:
            drawings = draw_footer(drawings, [
                (self.draw_landmark, 'Y', 'Mediapipe Hand Landmarks'),
                (self.draw_hand_direction, 'X', 'Index Finger Pointing Direction'),
                (self.draw_handgrid, 'C', 'Hand Pose Detection (YOLO, Hagrid)'),
                (self.draw_keypoints, 'V', 'Hand Keypoints (YOLO, OpenPose)'),
                (self.draw_framerate, 'B', 'Framerate Display'),
                (not self.draw_instructions, 'N', 'Hide Footer'),
                (self.draw_action, 'M', 'Draw Resulting Command'),
                (False, 'Q', 'Quit')
            ])


        return pose, index, drawings
