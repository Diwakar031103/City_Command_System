# Import required libraries
from ultralytics import YOLO
import cv2
import os
import time

"""---------------- CREATE FOLDER ---------------- """

# Create folder to store violation images (if not exists)
if not os.path.exists("violations"):
    os.makedirs("violations")

""" ---------------- LOAD MODEL ---------------- """

# YOLOv8 nano model (fast & lightweight)
model = YOLO("yolov8n.pt")

""" ---------------- LOAD VIDEO ----------------- """

# Open traffic video file
cap = cv2.VideoCapture("traffic.mp4")

""" ---------------- INITIAL SETTINGS ---------------- """
# Y-coordinate of stop line
line_y = 400

 # Store IDs of vehicles already counted
violated_ids = set()

 # Total violations counter
violation_count = 0

# Start time for traffic signal logic
start_time = time.time()


""" ---------------- LOOP ----------------- """

while True:
    ret, frame = cap.read()
    if not ret:                     # Break loop when video ends
        break
""" ---------------- SIGNAL LOGIC ----------------- """

# Calculate elapsed time since start
    elapsed = int(time.time() - start_time)

# Simple signal: 5 sec RED, 5 sec GREEN (loop)
    if elapsed % 10 < 5:
        light_state = "RED"
        line_color = (0,0,255)
    else:
        light_state = "GREEN"
        line_color = (0,255,0)

""" --------------- OBJECT DETECTION ---------------- """

 # Track objects across frames (assigns unique IDs)
    results = model.track(frame, persist=True)

""" ---------------- DRAW STOP LINE ---------------- """

    # Vehicles crossing this line during RED = violation
    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), line_color, 3)

""" ---------------- PROCESS DETECTIONS ---------------- """

    for box in results[0].boxes:
        if box.id is None: # Skip if no tracking ID assigned
            continue

        # Extract object ID and bounding box
        obj_id = int(box.id[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

""" ---------------- VIOLATION LOGIC ----------------"""
        # If vehicle crosses line during RED signal
        if light_state == "RED" and y2 > line_y:

            # Check if this vehicle already counted
            if obj_id not in violated_ids:
                violated_ids.add(obj_id)
                violation_count += 1

                 # Save full frame as proof image
                cv2.imwrite(f"violations/violation_{obj_id}.jpg", frame)
            
             # Draw RED bounding box for violation
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 3)
             # Label as violation
            cv2.putText(frame, "VIOLATION", (x1, y1),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        else:
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)   # Draw GREEN box for normal vehicles

""" ---------------- DISPLAY STATS ----------------- """

    # Show total violations count
    cv2.putText(frame, f"Total Violations: {violation_count}", (20,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

     # Show current traffic light state
    cv2.putText(frame, f"Light: {light_state}", (20,100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, line_color, 2)

""" ---------------- DISPLAY WINDOW ----------------- """

    # Show processed video output
    cv2.imshow("Traffic System", frame)

    # Press ESC key to exit manually
    if cv2.waitKey(1) == 27:
        break
""" ----------------- CLEANUP -------------------"""

# Release video and close windows
cap.release()
cv2.destroyAllWindows()