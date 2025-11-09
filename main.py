import cv2
import numpy as np
from ultralytics import YOLO
from pyzbar import pyzbar
import time

class PersonFollower:
    def __init__(self, camera_id=0, yolo_model='yolov8n.pt'):
        """
        Initialize the person following system
        
        Args:
            camera_id: Camera device ID (default 0 for webcam)
            yolo_model: YOLO model path (yolov8n.pt, yolov8s.pt, etc.)
        """
        self.cap = cv2.VideoCapture(camera_id)
        self.model = YOLO(yolo_model)
        
        # Tracking state
        self.target_qr_code = None
        self.target_person_bbox = None
        self.is_tracking = False
        self.person_id = None
        
        # Detection parameters
        self.qr_detection_cooldown = 2.0  # seconds
        self.last_qr_detection = 0
        self.confidence_threshold = 0.5
        
        # Movement parameters (for robot control)
        self.frame_center = None
        self.dead_zone = 50  # pixels from center
        
    def detect_qr_codes(self, frame):
        """Detect QR codes in the frame"""
        qr_codes = pyzbar.decode(frame)
        return qr_codes
    
    def get_person_detections(self, frame):
        """Get all person detections from YOLO"""
        results = self.model(frame, classes=[0], verbose=False)  # class 0 is 'person'
        return results[0].boxes
    
    def find_person_near_qr(self, qr_rect, person_boxes, frame_shape):
        """Find the person detection closest to the QR code"""
        qr_center = np.array([
            qr_rect.left + qr_rect.width / 2,
            qr_rect.top + qr_rect.height / 2
        ])
        
        min_distance = float('inf')
        closest_person = None
        
        for box in person_boxes:
            if box.conf[0] < self.confidence_threshold:
                continue
                
            bbox = box.xyxy[0].cpu().numpy()
            person_center = np.array([
                (bbox[0] + bbox[2]) / 2,
                (bbox[1] + bbox[3]) / 2
            ])
            
            distance = np.linalg.norm(qr_center - person_center)
            
            # Check if QR code is within person bounding box
            if (bbox[0] <= qr_center[0] <= bbox[2] and 
                bbox[1] <= qr_center[1] <= bbox[3]):
                # QR code is inside person bbox - strongest match
                return bbox, 0
            
            if distance < min_distance:
                min_distance = distance
                closest_person = bbox
        
        # Return closest person if within reasonable distance
        if closest_person is not None and min_distance < 200:
            return closest_person, min_distance
        
        return None, float('inf')
    
    def track_person(self, person_boxes):
        """Track the target person across frames"""
        if self.target_person_bbox is None:
            return None
        
        prev_center = np.array([
            (self.target_person_bbox[0] + self.target_person_bbox[2]) / 2,
            (self.target_person_bbox[1] + self.target_person_bbox[3]) / 2
        ])
        
        min_distance = float('inf')
        best_match = None
        
        for box in person_boxes:
            if box.conf[0] < self.confidence_threshold:
                continue
                
            bbox = box.xyxy[0].cpu().numpy()
            curr_center = np.array([
                (bbox[0] + bbox[2]) / 2,
                (bbox[1] + bbox[3]) / 2
            ])
            
            distance = np.linalg.norm(prev_center - curr_center)
            
            if distance < min_distance:
                min_distance = distance
                best_match = bbox
        
        # Update tracking if person found within reasonable distance
        if best_match is not None and min_distance < 150:
            return best_match
        
        return None
    
    def calculate_robot_command(self, bbox, frame_shape):
        """
        Calculate robot movement commands based on person position
        Returns: (move_direction, turn_direction, distance_estimate)
        """
        if bbox is None:
            return "STOP", "CENTER", "UNKNOWN"
        
        # Calculate person center
        person_center_x = (bbox[0] + bbox[2]) / 2
        person_center_y = (bbox[1] + bbox[3]) / 2
        
        # Frame center
        frame_center_x = frame_shape[1] / 2
        frame_center_y = frame_shape[0] / 2
        
        # Calculate horizontal offset
        offset_x = person_center_x - frame_center_x
        
        # Determine turn direction
        if abs(offset_x) < self.dead_zone:
            turn = "CENTER"
        elif offset_x > 0:
            turn = "RIGHT"
        else:
            turn = "LEFT"
        
        # Estimate distance based on bounding box size
        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        frame_area = frame_shape[0] * frame_shape[1]
        bbox_ratio = bbox_area / frame_area
        
        if bbox_ratio > 0.3:
            distance = "CLOSE"
            move = "STOP"
        elif bbox_ratio > 0.15:
            distance = "MEDIUM"
            move = "SLOW"
        else:
            distance = "FAR"
            move = "FORWARD"
        
        return move, turn, distance
    
    def draw_overlay(self, frame, person_boxes, qr_codes):
        """Draw detection and tracking overlays"""
        h, w = frame.shape[:2]
        
        # Draw frame center crosshair
        cv2.line(frame, (w//2 - 20, h//2), (w//2 + 20, h//2), (255, 255, 255), 1)
        cv2.line(frame, (w//2, h//2 - 20), (w//2, h//2 + 20), (255, 255, 255), 1)
        
        # Draw all person detections
        for box in person_boxes:
            bbox = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            
            color = (100, 100, 100)  # Gray for non-target persons
            thickness = 1
            
            cv2.rectangle(frame, 
                         (int(bbox[0]), int(bbox[1])), 
                         (int(bbox[2]), int(bbox[3])), 
                         color, thickness)
            cv2.putText(frame, f'{conf:.2f}', 
                       (int(bbox[0]), int(bbox[1] - 5)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw QR codes
        for qr in qr_codes:
            points = qr.polygon
            if len(points) == 4:
                pts = np.array([[p.x, p.y] for p in points], np.int32)
                cv2.polylines(frame, [pts], True, (0, 255, 255), 2)
                cv2.putText(frame, qr.data.decode('utf-8'),
                           (qr.rect.left, qr.rect.top - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw tracked person
        if self.target_person_bbox is not None and self.is_tracking:
            bbox = self.target_person_bbox
            cv2.rectangle(frame,
                         (int(bbox[0]), int(bbox[1])),
                         (int(bbox[2]), int(bbox[3])),
                         (0, 255, 0), 3)
            
            # Draw tracking label
            label = f"TRACKING: {self.person_id}"
            cv2.putText(frame, label,
                       (int(bbox[0]), int(bbox[1] - 10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw center line from frame center to person
            person_center_x = int((bbox[0] + bbox[2]) / 2)
            person_center_y = int((bbox[1] + bbox[3]) / 2)
            cv2.line(frame, (w//2, h//2), 
                    (person_center_x, person_center_y), 
                    (0, 255, 0), 2)
        
        return frame
    
    def run(self):
        """Main loop for the person following system"""
        print("Person Following System Started")
        print("Show a QR code to register a person to follow")
        print("Press 'q' to quit, 'r' to reset tracking")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            h, w = frame.shape[:2]
            current_time = time.time()
            
            # Detect persons in frame
            person_boxes = self.get_person_detections(frame)
            
            # QR code detection (with cooldown to avoid flickering)
            qr_codes = []
            if current_time - self.last_qr_detection > self.qr_detection_cooldown:
                qr_codes = self.detect_qr_codes(frame)
                
                if qr_codes and not self.is_tracking:
                    # New QR code detected - register person
                    qr = qr_codes[0]
                    self.target_qr_code = qr.data.decode('utf-8')
                    self.person_id = self.target_qr_code
                    
                    # Find person near QR code
                    person_bbox, distance = self.find_person_near_qr(
                        qr.rect, person_boxes, frame.shape
                    )
                    
                    if person_bbox is not None:
                        self.target_person_bbox = person_bbox
                        self.is_tracking = True
                        self.last_qr_detection = current_time
                        print(f"Now tracking person with ID: {self.person_id}")
            
            # Track the target person
            if self.is_tracking:
                new_bbox = self.track_person(person_boxes)
                
                if new_bbox is not None:
                    self.target_person_bbox = new_bbox
                    
                    # Calculate robot commands
                    move, turn, distance = self.calculate_robot_command(
                        self.target_person_bbox, frame.shape
                    )
                    
                    # Display commands
                    cmd_text = f"CMD: {move} | {turn} | {distance}"
                    cv2.putText(frame, cmd_text, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Here you would send commands to your robot
                    # self.send_robot_command(move, turn, distance)
                else:
                    # Lost tracking
                    cv2.putText(frame, "LOST TRACKING - Searching...", 
                               (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "WAITING FOR QR CODE", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Draw overlays
            frame = self.draw_overlay(frame, person_boxes, qr_codes)
            
            # Status info
            status = f"Tracking: {self.is_tracking} | ID: {self.person_id}"
            cv2.putText(frame, status, (10, h - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Person Follower', frame)
            
            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset tracking
                self.is_tracking = False
                self.target_person_bbox = None
                self.target_qr_code = None
                self.person_id = None
                print("Tracking reset")
        
        self.cleanup()
    
    def cleanup(self):
        """Release resources"""
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Initialize and run the person follower
    follower = PersonFollower(camera_id=0, yolo_model='yolov8n.pt')
    follower.run()