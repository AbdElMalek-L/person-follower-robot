"""
Human-following robot (YOLO + QR + DeepSORT) with live preview window.

Features:
- Detects persons using YOLOv8
- Detects QR codes using OpenCV
- Matches QR to nearest person
- Tracks the identified person using DeepSORT
- Displays preview with bounding boxes, IDs, and control info
- Prints movement commands (left/right/forward)
"""

import cv2
import numpy as np
import math
import time
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Optional serial control (uncomment if needed)
# import serial

# ------------------ Config ------------------
YOLO_MODEL = "yolov8n.pt"
VIDEO_SOURCE = 0  # camera index or path to video file
MIN_QR_PERSON_MATCH_DIST = 200
TARGET_LOST_TIMEOUT = 2.0
KP_ANG = 0.004
KP_DIST = 0.005
DESIRED_PERSON_AREA = 20000
FRAME_W, FRAME_H = 640, 480
# SERIAL_PORT = "COM3"
# SERIAL_BAUD = 115200
# --------------------------------------------

def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

def bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    return max(0, (x2 - x1) * (y2 - y1))

def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def main():
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    model = YOLO(YOLO_MODEL)
    qr = cv2.QRCodeDetector()
    tracker = DeepSort(max_age=30, n_init=3, max_cosine_distance=0.2)

    # Optional: serial connection
    # ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0.1)

    target_id = None
    target_last_seen = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.4)[0]
        person_boxes = []

        for box in results.boxes:
            cls = int(box.cls)
            if cls == 0:  # person
                x1, y1, x2, y2 = box.xyxy[0]
                conf = float(box.conf)
                person_boxes.append(([x1, y1, x2, y2], conf, "person"))

        tracks = tracker.update_tracks(person_boxes, frame=frame)
        data, bbox, _ = qr.detectAndDecode(frame)

        qr_bbox, qr_center = None, None
        if data and bbox is not None:
            pts = np.int32(bbox).reshape(-1, 2)
            qr_bbox = [pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()]
            qr_center = bbox_center(qr_bbox)
            cv2.polylines(frame, [pts], True, (0, 255, 255), 2)
            cv2.putText(frame, f"QR: {data}", (pts[0][0], pts[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # If QR seen, link to nearest person
        if qr_center:
            nearest_id, nearest_dist = None, 1e9
            for tr in tracks:
                if not tr.is_confirmed():
                    continue
                tid = tr.track_id
                x1, y1, x2, y2 = tr.to_ltrb()
                p_center = bbox_center([x1, y1, x2, y2])
                d = distance(p_center, qr_center)
                if d < nearest_dist:
                    nearest_dist, nearest_id = d, tid
            if nearest_dist < MIN_QR_PERSON_MATCH_DIST:
                target_id = nearest_id
                target_last_seen = time.time()

        target_bbox = None
        for tr in tracks:
            if not tr.is_confirmed():
                continue
            x1, y1, x2, y2 = map(int, tr.to_ltrb())
            tid = tr.track_id
            color = (0, 255, 0) if tid == target_id else (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID {tid}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            if tid == target_id:
                target_bbox = [x1, y1, x2, y2]
                target_last_seen = time.time()

        if target_id and target_last_seen and (time.time() - target_last_seen > TARGET_LOST_TIMEOUT):
            print("[WARN] Target lost")
            target_id = None
            target_bbox = None

        # Draw center point
        cx, cy = FRAME_W // 2, FRAME_H // 2
        cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)

        if target_bbox:
            tx1, ty1, tx2, ty2 = target_bbox
            tcx, tcy = bbox_center(target_bbox)
            cv2.line(frame, (cx, cy), (tcx, tcy), (0, 255, 0), 2)

            err_x = tcx - cx
            area = bbox_area(target_bbox)
            err_dist = DESIRED_PERSON_AREA - area

            turn = KP_ANG * err_x
            fwd = KP_DIST * err_dist

            # movement text
            turn_dir = "left" if err_x < 0 else "right"
            move_dir = "forward" if err_dist > 0 else "backward"
            cv2.putText(frame, f"TURN: {turn_dir} | MOVE: {move_dir}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)

            print(f"[CMD] Turn: {turn:.3f} | Forward: {fwd:.3f}")

            # send over serial if enabled
            # cmd = f"T{int(turn*100):03d}F{int(fwd*100):03d}\n"
            # ser.write(cmd.encode())

        cv2.imshow("Human Following Robot", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    # ser.close()


if __name__ == "__main__":
    main()
