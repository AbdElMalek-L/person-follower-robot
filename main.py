import cv2
import numpy as np
from ultralytics import YOLO
from pyzbar import pyzbar
import time
import json
import os
from datetime import datetime

class PersonFollower:
    def __init__(self, camera_id=0, yolo_model='yolov8n.pt', auth_file='authorized_persons.json'):
        """
        Initialize the person following system
        
        Args:
            camera_id: Camera device ID (default 0 for webcam)
            yolo_model: YOLO model path (yolov8n.pt, yolov8s.pt, etc.)
            auth_file: JSON file to store authorized persons
        """
        self.cap = cv2.VideoCapture(camera_id)
        self.model = YOLO(yolo_model)
        
        # Authorized persons database
        self.auth_file = auth_file
        self.authorized_persons = self.load_authorized_persons()
        
        # Tracking state
        self.target_qr_code = None
        self.target_person_bbox = None
        self.is_tracking = False
        self.current_person = None
        
        # Detection parameters
        self.qr_detection_cooldown = 2.0  # seconds
        self.last_qr_detection = 0
        self.confidence_threshold = 0.5
        
        # Movement parameters (for robot control)
        self.frame_center = None
        self.dead_zone = 50  # pixels from center
        
        # Registration mode
        self.registration_mode = False
        
    def load_authorized_persons(self):
        """Load authorized persons from JSON file"""
        if os.path.exists(self.auth_file):
            try:
                with open(self.auth_file, 'r') as f:
                    data = json.load(f)
                    print(f"Loaded {len(data)} authorized persons")
                    return data
            except Exception as e:
                print(f"Error loading authorized persons: {e}")
                return {}
        return {}
    
    def save_authorized_persons(self):
        """Save authorized persons to JSON file"""
        try:
            with open(self.auth_file, 'w') as f:
                json.dump(self.authorized_persons, f, indent=2)
            print(f"Saved {len(self.authorized_persons)} authorized persons")
        except Exception as e:
            print(f"Error saving authorized persons: {e}")
    
    def register_person(self, qr_code, name=None):
        """Register a new authorized person"""
        if qr_code not in self.authorized_persons:
            person_data = {
                'qr_code': qr_code,
                'name': name or f"Person_{len(self.authorized_persons) + 1}",
                'registered_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'last_seen': None,
                'tracking_count': 0
            }
            self.authorized_persons[qr_code] = person_data
            self.save_authorized_persons()
            print(f"Registered new person: {person_data['name']} (QR: {qr_code})")
            return True
        else:
            print(f"Person already registered: {self.authorized_persons[qr_code]['name']}")
            return False
    
    def remove_person(self, qr_code):
        """Remove an authorized person"""
        if qr_code in self.authorized_persons:
            person_name = self.authorized_persons[qr_code]['name']
            del self.authorized_persons[qr_code]
            self.save_authorized_persons()
            print(f"Removed person: {person_name}")
            return True
        return False
    
    def is_authorized(self, qr_code):
        """Check if a QR code is authorized"""
        return qr_code in self.authorized_persons
    
    def update_person_tracking(self, qr_code):
        """Update tracking statistics for a person"""
        if qr_code in self.authorized_persons:
            self.authorized_persons[qr_code]['last_seen'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.authorized_persons[qr_code]['tracking_count'] += 1
            self.save_authorized_persons()
    
    def list_authorized_persons(self):
        """Return list of authorized persons"""
        return self.authorized_persons
    
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
                return bbox, 0
            
            if distance < min_distance:
                min_distance = distance
                closest_person = bbox
        
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
        
        if best_match is not None and min_distance < 150:
            return best_match
        
        return None
    
    def calculate_robot_command(self, bbox, frame_shape):
        """Calculate robot movement commands based on person position"""
        if bbox is None:
            return "STOP", "CENTER", "UNKNOWN"
        
        person_center_x = (bbox[0] + bbox[2]) / 2
        frame_center_x = frame_shape[1] / 2
        
        offset_x = person_center_x - frame_center_x
        
        if abs(offset_x) < self.dead_zone:
            turn = "CENTER"
        elif offset_x > 0:
            turn = "RIGHT"
        else:
            turn = "LEFT"
        
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
            
            color = (100, 100, 100)
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
            qr_data = qr.data.decode('utf-8')
            is_auth = self.is_authorized(qr_data)
            
            points = qr.polygon
            if len(points) == 4:
                pts = np.array([[p.x, p.y] for p in points], np.int32)
                color = (0, 255, 0) if is_auth else (0, 255, 255)
                cv2.polylines(frame, [pts], True, color, 2)
                
                label = qr_data
                if is_auth:
                    person_name = self.authorized_persons[qr_data]['name']
                    label = f"{person_name} ✓"
                elif self.registration_mode:
                    label = f"{qr_data} [REGISTER]"
                
                cv2.putText(frame, label,
                           (qr.rect.left, qr.rect.top - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw tracked person
        if self.target_person_bbox is not None and self.is_tracking:
            bbox = self.target_person_bbox
            cv2.rectangle(frame,
                         (int(bbox[0]), int(bbox[1])),
                         (int(bbox[2]), int(bbox[3])),
                         (0, 255, 0), 3)
            
            person_name = self.current_person['name'] if self.current_person else "Unknown"
            label = f"TRACKING: {person_name}"
            cv2.putText(frame, label,
                       (int(bbox[0]), int(bbox[1] - 10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            person_center_x = int((bbox[0] + bbox[2]) / 2)
            person_center_y = int((bbox[1] + bbox[3]) / 2)
            cv2.line(frame, (w//2, h//2), 
                    (person_center_x, person_center_y), 
                    (0, 255, 0), 2)
        
        # Draw authorized persons list
        y_offset = 60
        cv2.putText(frame, f"Authorized: {len(self.authorized_persons)}", 
                   (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def print_menu(self):
        """Print control menu"""
        print("\n" + "="*50)
        print("PERSON FOLLOWER - CONTROL MENU")
        print("="*50)
        print("Main Controls:")
        print("  q - Quit program")
        print("  r - Reset tracking")
        print("  t - Toggle registration mode")
        print("  l - List authorized persons")
        print("  d - Delete a person")
        print("\nRegistration Mode:")
        print("  - Show QR code to register/track person")
        print("  - Authorized QR codes: Green border")
        print("  - Unauthorized QR codes: Yellow border")
        print("="*50 + "\n")
    
    def run(self):
        """Main loop for the person following system"""
        self.print_menu()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            h, w = frame.shape[:2]
            current_time = time.time()
            
            # Detect persons in frame
            person_boxes = self.get_person_detections(frame)
            
            # QR code detection
            qr_codes = []
            if current_time - self.last_qr_detection > self.qr_detection_cooldown:
                qr_codes = self.detect_qr_codes(frame)
                
                if qr_codes:
                    qr = qr_codes[0]
                    qr_data = qr.data.decode('utf-8')
                    
                    # Registration mode
                    if self.registration_mode:
                        if not self.is_authorized(qr_data):
                            self.register_person(qr_data)
                        self.last_qr_detection = current_time
                    
                    # Tracking mode (only authorized persons)
                    elif not self.is_tracking and self.is_authorized(qr_data):
                        self.target_qr_code = qr_data
                        self.current_person = self.authorized_persons[qr_data]
                        
                        person_bbox, distance = self.find_person_near_qr(
                            qr.rect, person_boxes, frame.shape
                        )
                        
                        if person_bbox is not None:
                            self.target_person_bbox = person_bbox
                            self.is_tracking = True
                            self.update_person_tracking(qr_data)
                            self.last_qr_detection = current_time
                            print(f"Now tracking: {self.current_person['name']}")
                    
                    # Unauthorized QR code
                    elif not self.is_tracking and not self.is_authorized(qr_data):
                        print(f"Unauthorized QR code: {qr_data}")
                        self.last_qr_detection = current_time
            
            # Track the target person
            if self.is_tracking:
                new_bbox = self.track_person(person_boxes)
                
                if new_bbox is not None:
                    self.target_person_bbox = new_bbox
                    
                    move, turn, distance = self.calculate_robot_command(
                        self.target_person_bbox, frame.shape
                    )
                    
                    cmd_text = f"CMD: {move} | {turn} | {distance}"
                    cv2.putText(frame, cmd_text, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "LOST TRACKING - Searching...", 
                               (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                mode = "REGISTRATION MODE" if self.registration_mode else "SHOW QR CODE"
                color = (255, 165, 0) if self.registration_mode else (0, 255, 255)
                cv2.putText(frame, mode, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw overlays
            frame = self.draw_overlay(frame, person_boxes, qr_codes)
            
            # Status info
            tracking_status = f"✓ {self.current_person['name']}" if self.is_tracking else "Waiting"
            status = f"Tracking: {tracking_status} | Auth: {len(self.authorized_persons)}"
            cv2.putText(frame, status, (10, h - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Person Follower', frame)
            
            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.is_tracking = False
                self.target_person_bbox = None
                self.target_qr_code = None
                self.current_person = None
                print("Tracking reset")
            elif key == ord('t'):
                self.registration_mode = not self.registration_mode
                mode = "ENABLED" if self.registration_mode else "DISABLED"
                print(f"Registration mode: {mode}")
            elif key == ord('l'):
                print("\n--- Authorized Persons ---")
                if self.authorized_persons:
                    for qr_code, person in self.authorized_persons.items():
                        print(f"  {person['name']} (QR: {qr_code})")
                        print(f"    Registered: {person['registered_date']}")
                        print(f"    Last seen: {person['last_seen']}")
                        print(f"    Tracking count: {person['tracking_count']}")
                else:
                    print("  No authorized persons")
                print("-------------------------\n")
            elif key == ord('d'):
                print("\nEnter QR code to delete (or 'cancel'):")
                for qr_code, person in self.authorized_persons.items():
                    print(f"  {qr_code} - {person['name']}")
                # Note: In a real application, you'd implement proper input handling
        
        self.cleanup()
    
    def cleanup(self):
        """Release resources"""
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    follower = PersonFollower(camera_id=0, yolo_model='yolov8n.pt')
    follower.run()