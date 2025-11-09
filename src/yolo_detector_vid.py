import cv2
from ultralytics import YOLO
import numpy as np
import time
import argparse

def process_video(video_path, output_path=None):
    # Initialize YOLOv8 model
    model = YOLO('yolov8n.pt')  # 'n' for nano model, you can use 's', 'm', 'l', or 'x' for larger models
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Initialize video writer if output path is specified
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    print("Processing video... Press 'q' to quit")
    
    frame_count = 0
    total_time = 0
    
    while True:
        start_time = time.time()
        
        # Read frame
        ret, frame = cap.read()
        if not ret:
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
        process_time = time.time() - start_time
        total_time += process_time
        frame_count += 1
        avg_fps = frame_count / total_time
        
        cv2.putText(frame, f'FPS: {avg_fps:.2f}', 
                   (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 255, 0), 2)
        
        # Write frame to output video if specified
        if out:
            out.write(frame)
        
        # Display the frame
        cv2.imshow('YOLOv8 People Detection', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='YOLOv8 Video People Detection')
    parser.add_argument('input', help='Path to input video file')
    parser.add_argument('--output', help='Path to output video file (optional)')
    args = parser.parse_args()
    
    process_video(args.input, args.output)

if __name__ == '__main__':
    main()