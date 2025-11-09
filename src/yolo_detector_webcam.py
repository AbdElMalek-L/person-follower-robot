import cv2
from ultralytics import YOLO
import numpy as np
import time

def main():
    # Initialize YOLOv8 model - using the pre-trained model optimized for people detection
    model = YOLO('yolov8n.pt')  # 'n' for nano model, you can use 's', 'm', 'l', or 'x' for larger models
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Check if webcam opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Press 'q' to quit")
    
    while True:
        start_time = time.time()  # Start time for FPS calculation
        
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Run YOLOv8 inference on the frame
        # Only detect person class (class 0 in COCO dataset)
        results = model(frame, classes=[0])  
        
        # Process detections
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].astype(int)
                # Get confidence score
                conf = box.conf[0]
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Add confidence score
                cv2.putText(frame, f'Person: {conf:.2f}', 
                          (x1, y1 - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 
                          0.5, (0, 255, 0), 2)
        
        # Calculate and display FPS
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f'FPS: {fps:.2f}', 
                   (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('YOLOv8 People Detection', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()