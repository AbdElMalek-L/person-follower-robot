"""
Realtime webcam HOG human detector.

Run:
    python src/hog_detector_webcam.py

Press 'q' to quit.
"""

import time
import argparse
import cv2


def main(device: int, width: int, speed: str):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        print(f"Unable to open webcam device {device}")
        return

    # try to set a reasonable width
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)

    avg_fps = 0.0
    alpha = 0.9  # smoothing for fps display
    frame_count = 0

    print("Starting webcam. Press 'q' to quit.")
    while True:
        start = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Frame grab failed, stopping")
            break

        h, w = frame.shape[:2]
        if w < width:
            # upscale to target width keeping aspect ratio
            new_h = int(width * (h / float(w)))
            frame = cv2.resize(frame, (width, new_h))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if speed == 'fast':
            rects, weights = hog.detectMultiScale(gray, padding=(8, 8), scale=1.02)
        else:
            rects, weights = hog.detectMultiScale(gray, winStride=(4, 4), padding=(8, 8), scale=1.02)

        for i, (x, y, fw, fh) in enumerate(rects):
            wgt = weights[i] if len(weights) > i else 1.0
            if wgt < 0.13:
                continue
            color = (0, 255, 0) if wgt > 0.7 else ((50, 122, 255) if wgt > 0.3 else (0, 0, 255))
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), color, 2)

        # FPS calculation
        end = time.time()
        fps = 1.0 / (end - start) if (end - start) > 0 else 0.0
        if frame_count == 0:
            avg_fps = fps
        else:
            avg_fps = alpha * avg_fps + (1 - alpha) * fps
        frame_count += 1

        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, 'High confidence', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(frame, 'Moderate confidence', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 122, 255), 1)
        cv2.putText(frame, 'Low confidence', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

        cv2.imshow('Webcam HOG detector', frame)

        # wait 1 ms and check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', type=int, default=0, help='webcam device index (default 0)')
    parser.add_argument('-w', '--width', type=int, default=640, help='target frame width (default 640)')
    parser.add_argument('-s', '--speed', choices=['fast', 'slow'], default='fast', help='detection mode (fast/slow)')
    args = parser.parse_args()
    main(args.device, args.width, args.speed)
