# Import required libraries
import streamlit as st
import cv2
import tempfile
import os
import time
from ultralytics import YOLO

# ---------------- UI ----------------

# Streamlit UI setup for traffic monitoring dashboard
st.set_page_config(page_title="Traffic AI Pro", layout="wide")
st.title("🚦 Smart Traffic AI System ")

# ---------------- MODEL ----------------

# Cache model so it loads only once (improves performance)
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")     # Lightweight YOLOv8 model

model = load_model()

# ---------------- CREATE STORAGE FOLDER ----------------
# Folder to store violation proof images
if not os.path.exists("violations"):
    os.makedirs("violations")

# ---------------- VIDEO INPUT ----------------

# Upload traffic video file
uploaded_file = st.file_uploader("Upload Traffic Video", type=["mp4", "avi", "mov"])

# ---------------- TRAFFIC SIGNAL SETTINGS ----------------

# User can control signal timing dynamically
red_time = st.slider("🔴 Red Light Time", 5, 30, 10)
yellow_time = st.slider("🟡 Yellow Light Time", 2, 10, 3)
green_time = st.slider("🟢 Green Light Time", 5, 30, 10)

# ---------------- START ----------------
if uploaded_file:

    st.info("Processing video... ⏳")

    tfile = tempfile.NamedTemporaryFile(delete=False) # Save uploaded video temporarily
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)     # Read video using OpenCV

    if not cap.isOpened():                   # Error handling if video fails to load
        st.error("❌ Video load error")
        st.stop()
# ---------------- INITIAL SETTINGS ----------------

    line_y = 400               # Detection line (stop line)
    violated_ids = set()       # To avoid duplicate counting
    violation_count = 0         # Total violations counter

    stframe = st.empty()          # Streamlit placeholders for live updates
    stats = st.empty()

    start_time = time.time()      # Start time for signal cycle

    # ---------------- PROCESS LOOP ----------------
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (900, 500))       # Resize frame for consistent processing

        # ---------------- SIGNAL LOGIC ----------------

        # Calculate total signal cycle time
        cycle = red_time + yellow_time + green_time
        elapsed = int(time.time() - start_time) % cycle        # Get current time in cycle (looping)

        if elapsed < red_time:                  # Determine signal state based on time
            light_state = "RED"
            color = (0, 0, 255)

        elif elapsed < red_time + yellow_time:
            light_state = "YELLOW"
            color = (0, 255, 255)

        else:
            light_state = "GREEN"
            color = (0, 255, 0)

        # ---------------- DRAW STOP LINE ----------------

         # This line represents where vehicles must stop
        cv2.line(frame, (0, line_y), (900, line_y), color, 2)

        # ---------------- TRAFFIC LIGHT VISUAL BOX ----------------

        cv2.putText(frame, f"Signal: {light_state}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # ---------------- RAFFIC LIGHT VISUAL BOX  ----------------

        # Draw UI box showing red-yellow-green lights
        x_offset = frame.shape[1] - 80
        y_offset = 30

        cv2.rectangle(frame, (x_offset, y_offset),
                      (x_offset + 50, y_offset + 140),
                      (255, 255, 255), 2)

        r = (0,0,255) if light_state == "RED" else (50,50,50)
        y = (0,255,255) if light_state == "YELLOW" else (50,50,50)
        g = (0,255,0) if light_state == "GREEN" else (50,50,50)

        # Draw circles for signal lights
        cv2.circle(frame, (x_offset+25, y_offset+25), 12, r, -1)
        cv2.circle(frame, (x_offset+25, y_offset+70), 12, y, -1)
        cv2.circle(frame, (x_offset+25, y_offset+115), 12, g, -1)

        # ----------------  OBJECT DETECTION & TRACKING ----------------

        # Track vehicles across frames using YOLOv8 tracking
        results = model.track(frame, persist=True, conf=0.5)

        if results and results[0].boxes is not None:

            for box in results[0].boxes:

                if box.id is None:             # Skip if no tracking ID
                    continue

                obj_id = int(box.id.item())    # Unique object ID
                cls_id = int(box.cls.item())   # Class ID

                label = model.names[int(cls_id)]

                if label not in ["car", "truck", "bus", "motorcycle"]:    # Filter only vehicle classes
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates

                # ---------------- VIOLATION ----------------

                # If vehicle crosses line during RED signal
                if light_state == "RED" and y2 > line_y:

                    if obj_id not in violated_ids:         # Avoid duplicate counting for same vehicle
                        violated_ids.add(obj_id)
                        violation_count += 1

                        crop = frame[y1:y2, x1:x2]        # Save cropped image as proof
                        if crop.size > 0:
                            cv2.imwrite(f"violations/violation_{obj_id}.jpg", crop)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)          # Draw RED box for violation
                    cv2.putText(frame, f"VIOLATION ID:{obj_id}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                else:                                                                 # Draw GREEN box for normal vehicles
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} ID:{obj_id}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # ---------------- DISPLAY STATS ----------------

        cv2.putText(frame, f"Violations: {violation_count}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)

        # Show processed frame
        stframe.image(frame, channels="BGR")

          # Show live violation count
        stats.metric("Total Violations", violation_count)

        # Small delay to reduce CPU usage
        time.sleep(0.03)

     # Release video after processing
    cap.release()
     # Final success message
    st.success(f"✅ Done! Total Violations: {violation_count}")