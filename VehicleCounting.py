# Vehicle Counting using YOLOv8 and OpenCV
# -----------------------------------------

# Requirements: 
# pip install ultralytics opencv-python-headless numpy pillow

import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


from tkinter import Tk, filedialog

Tk().withdraw()  # لا تظهر نافذة Tk الرئيسية
FILE_PATH = filedialog.askopenfilename(title="Select image or video file")

#FILE_PATH =r"photoCars.jpeg"
#FILE_PATH =r"C:\Users\aljaz\OneDrive\سطح المكتب\CarCounting\videoCars.mp4"
MODEL_PATH = "yolov8m.pt" 
VEHICLE_CLASSES = ['car', 'truck', 'bus', 'motorbike', 'bicycle']
CONFIDENCE = 0.5        


# Load YOLO model
model = YOLO(MODEL_PATH)

# Get file extension
ext = os.path.splitext(FILE_PATH)[1].lower()

# ---------- image processing ----------

if ext in ['.jpg', '.jpeg', '.png']:
    # Load image
    img = np.array(Image.open(FILE_PATH).convert("RGB"))
    
    # Detect vehicles
    results = model(img)
    
    # used to Show image with bounding boxes
    results[0].show()
    
    # Count vehicles
    count = sum(
        1 for box in results[0].boxes
        if model.names[int(box.cls)] in VEHICLE_CLASSES
    )
    print(f"Number of vehicles detected: {count}")


# ---------- video processing ----------

elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
    cap = cv2.VideoCapture(FILE_PATH)
    frame_count = 0
    vehicle_classes = ['car', 'truck', 'bus', 'motorbike', 'bicycle']

   

    unique_centers = []  
    DIST_THRESHOLD = 50  

    while True:
        ret, frame = cap.read()
        if not ret:
            break

       
        results = model(frame) 
        annotated = results[0].plot()
        frame_count += 1
        count_in_frame = 0
        #total_unique = 0


        if hasattr(results[0], 'boxes') and results[0].boxes is not None:
            cls_ids = results[0].boxes.cls
            boxes = results[0].boxes.xyxy.cpu().numpy()

            for i, c in enumerate(cls_ids):
                if model.names[int(c)] in vehicle_classes:
                    x1, y1, x2, y2 = boxes[i]
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    
                    is_new = True
                    for (px, py) in unique_centers:
                        if abs(cx - px) < DIST_THRESHOLD and abs(cy - py) < DIST_THRESHOLD:
                            is_new = False
                            break

                    if is_new:
                        unique_centers.append((cx, cy))
                    count_in_frame += 1

        total_unique = len(unique_centers)

      
        cv2.putText(
            annotated,
            f"Vehicles in frame: {count_in_frame}, Total unique: {total_unique}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow("Vehicle Tracking", annotated)
        print(f"Frame {frame_count}: Vehicles in this frame: {count_in_frame} | Total unique: {total_unique}", end="\r")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n Video finished. Estimated total unique vehicles: {total_unique}")