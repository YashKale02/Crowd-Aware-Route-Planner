#app_video.py
import streamlit as st
import cv2
import tempfile
import numpy as np

from detection import load_model, detect_people, generate_crowd_map
from pathfinding import astar
from visualize import draw_path, get_smooth_path
from config import CONF_THRESHOLD, IOU_THRESHOLD, GRID_SIZE

st.set_page_config(page_title="Crowd Pathfinding", layout="wide")
st.title("🎥 Crowd-Aware Pathfinding (Video Input)")

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Controls")
uploaded_video = st.sidebar.file_uploader("📂 Upload Video", type=["mp4", "avi", "mov"])

# Parameters
CONF_THRESHOLD = st.sidebar.slider("Detection Confidence", 0.1, 1.0, CONF_THRESHOLD, 0.05)
GRID_SIZE = st.sidebar.slider("Grid Size", 5, 50, GRID_SIZE, 5)
CROWD_WEIGHT = st.sidebar.slider("Crowd Weight", 1.0, 100.0, 25.0, 1.0)
frame_skip = st.sidebar.slider("Process Every Nth Frame", 1, 10, 3)

st.sidebar.subheader("📍 Coordinates")
start_x = st.sidebar.number_input("Start X", 0, 1920, 100)
start_y = st.sidebar.number_input("Start Y", 0, 1080, 100)
goal_x = st.sidebar.number_input("Goal X", 0, 1920, 800)
goal_y = st.sidebar.number_input("Goal Y", 0, 1080, 400)

@st.cache_resource
def get_model():
    return load_model()

model = get_model()

if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    cap = cv2.VideoCapture(tfile.name)

    stframe = st.empty()
    progress_bar = st.progress(0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_counter, processed = 0, 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame_counter += 1
        if frame_counter % frame_skip != 0: continue

        # Resize and Detect
        frame = cv2.resize(frame, (960, 540))
        h, w = frame.shape[:2]
        boxes = detect_people(model, frame)

        # Mapping & Cost Calculation
        crowd_map = generate_crowd_map(frame, boxes)
        weighted_map = np.where(crowd_map > 0, crowd_map * CROWD_WEIGHT, 1)

        # Convert Pixels to Grid
        sy, sx = int(start_y / (h/GRID_SIZE)), int(start_x / (w/GRID_SIZE))
        gy, gx = int(goal_y / (h/GRID_SIZE)), int(goal_x / (w/GRID_SIZE))
        
        # Pathfinding
        path = astar(weighted_map, (sy, sx), (gy, gx), crowd_weight=CROWD_WEIGHT)

        # Visualization
        if path:
            smooth_path = get_smooth_path(frame, path)
            frame = draw_path(frame, path, smooth_path)

        cv2.circle(frame, (int(start_x), int(start_y)), 8, (0, 255, 0), -1)
        cv2.circle(frame, (int(goal_x), int(goal_y)), 8, (0, 0, 255), -1)

        stframe.image(frame, channels="BGR", use_container_width=True)
        
        processed += 1
        if total_frames > 0:
            progress_bar.progress(min(processed / (total_frames / frame_skip), 1.0))

    cap.release()
    st.success("Processing Complete!")
else:
    st.info("Please upload a video.")