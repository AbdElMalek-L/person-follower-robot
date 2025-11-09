import cv2
from ultralytics import YOLO
from pyzbar import pyzbar
import numpy as np

# --- CONFIGURATION ---
# 1. The data your QR code must contain
TARGET_QR_DATA = "FOLLOW_ME" 

# 2. Target "area" for the robot to maintain distance.
#    - Make this SMALLER to follow from FARTHER away.
#    - Make this LARGER to follow from CLOSER.
#    (You will need to tune this value!)
TARGET_AREA_PX = 60000 

# 3. Tolerances for robot control (adjust as needed)
AREA_TOLERANCE_PX = 5000  # How much the area can deviate before moving
CENTER_TOLERANCE_PX = 30  # How much the center can deviate before turning
# --- END CONFIGURATION ---


def main():
    # Load the YOLOv8 model
    # 'yolov8n.pt' is the smallest, fastest model.
    # You can use 'yolov8s.pt' or 'yolov8m.pt' for better accuracy but slower speed.
    print("Loading YOLO model...")
    model = YOLO('yolov8n.pt')
    print("Model loaded.")

    # Open a connection to the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    # Get frame dimensions (will be used for robot control)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_center_x = frame_width // 2

    # --- STATE VARIABLES ---
    target_acquired = False   # Are we currently tracking a target?
    tracker = None            # The OpenCV tracker object
    target_bbox = None        # The bounding box (x, y, w, h) of the target
    # --- END STATE ---

    print("Starting main loop... Press 'q' to quit.")

    while True:
        # Read a new frame
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame.")
            break

        # --- MODE 1: ACQUISITION (Looking for a new target) ---
        if not target_acquired:
            
            # 1. Run YOLO detection to find all people
            #    We only care about class 0, which is 'person'
            yolo_results = model(frame, classes=[0], verbose=False)
            
            # 2. Run QR code detection
            qr_codes = pyzbar.decode(frame)

            # Store potential targets
            potential_targets = []

            for qr in qr_codes:
                # Check if this QR code is our target
                qr_data = qr.data.decode('utf-8')
                if qr_data == TARGET_QR_DATA:
                    
                    # Get the bounding box of the QR code
                    (qr_x, qr_y, qr_w, qr_h) = qr.rect
                    # Calculate the center of the QR code
                    qr_center_x = qr_x + qr_w / 2
                    qr_center_y = qr_y + qr_h / 2
                    
                    # Draw a box around the *found* QR code
                    cv2.rectangle(frame, (qr_x, qr_y), (qr_x + qr_w, qr_y + qr_h), (255, 0, 0), 2)
                    cv2.putText(frame, "TARGET QR", (qr_x, qr_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    # 3. Link QR code to a YOLO 'person' detection
                    for res in yolo_results:
                        for box in res.boxes:
                            # Get person bounding box in xyxy format
                            (p_x1, p_y1, p_x2, p_y2) = box.xyxy[0].cpu().numpy().astype(int)
                            
                            # Check if the QR code's center is *inside* this person's box
                            if p_x1 < qr_center_x < p_x2 and p_y1 < qr_center_y < p_y2:
                                # This is our target!
                                
                                # Convert (x1, y1, x2, y2) to (x, y, w, h) for the tracker
                                target_bbox = (p_x1, p_y1, p_x2 - p_x1, p_y2 - p_y1)
                                potential_targets.append(target_bbox)
                                
                                # Draw a special "candidate" box
                                cv2.rectangle(frame, (p_x1, p_y1), (p_x2, p_y2), (0, 255, 255), 2)
                                cv2.putText(frame, "Target Candidate", (p_x1, p_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # 4. Acquire the Target
            if potential_targets:
                # For simplicity, we just take the first one found
                target_bbox = potential_targets[0] 
                
                # --- TARGET ACQUIRED! ---
                # Initialize the OpenCV tracker
                tracker = cv2.TrackerCSRT_create()
                tracker.init(frame, target_bbox)
                target_acquired = True
                
                print(f"*** Target Acquired at {target_bbox} ***")
                
                # Draw the final "ACQUIRED" box
                (x, y, w, h) = target_bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(frame, "TARGET ACQUIRED", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            else:
                # If no target, just show the "hunting" status
                cv2.putText(frame, "STATUS: Hunting for Target QR", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


        # --- MODE 2: TRACKING (Following the acquired target) ---
        else:
            # We have a target, so we don't run YOLO or QR detection.
            # We just update the tracker. This is *much* faster.
            track_success, new_box = tracker.update(frame)

            if track_success:
                # --- ROBOT CONTROL LOGIC ---
                # This is where you send commands to your robot
                
                (x, y, w, h) = [int(v) for v in new_box]
                
                # Draw the tracking box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "STATUS: Tracking Target", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # 1. STEERING (Turn Left/Right)
                box_center_x = x + w / 2
                
                if box_center_x < frame_center_x - CENTER_TOLERANCE_PX:
                    print("ROBOT COMMAND: TURN LEFT")
                    cv2.putText(frame, "ROBOT: TURN LEFT", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                elif box_center_x > frame_center_x + CENTER_TOLERANCE_PX:
                    print("ROBOT COMMAND: TURN RIGHT")
                    cv2.putText(frame, "ROBOT: TURN RIGHT", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                else:
                    print("ROBOT COMMAND: GO STRAIGHT")
                    cv2.putText(frame, "ROBOT: GO STRAIGHT", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # 2. DISTANCE (Move Forward/Backward)
                current_area = w * h
                
                if current_area < TARGET_AREA_PX - AREA_TOLERANCE_PX:
                    print("ROBOT COMMAND: MOVE FORWARD")
                    cv2.putText(frame, "ROBOT: MOVE FORWARD", (x, y + h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                elif current_area > TARGET_AREA_PX + AREA_TOLERANCE_PX:
                    print("ROBOT COMMAND: MOVE BACKWARD")
                    cv2.putText(frame, "ROBOT: MOVE BACKWARD", (x, y + h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                else:
                    print("ROBOT COMMAND: STOP / HOLD DISTANCE")
                    cv2.putText(frame, "ROBOT: HOLD DISTANCE", (x, y + h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            else:
                # --- TRACKING FAILED! ---
                # The tracker lost the target.
                # We need to reset and go back to Acquisition mode.
                target_acquired = False
                tracker = None
                target_bbox = None
                
                print("--- Tracking failed, re-acquiring... ---")
                
                # Display a "lost" message
                cv2.putText(frame, "STATUS: Target Lost! Re-acquiring...", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


        # Display the resulting frame
        cv2.imshow("Human Following Robot - View", frame)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    print("Shutting down...")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()