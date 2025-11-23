# app.py

import streamlit as st
import cv2
import numpy as np
import tempfile
from streamlit_image_coordinates import streamlit_image_coordinates

# Import all your existing logic
from detection import load_model, detect_people, generate_crowd_map
from graph import normalize_grid
from pathfinding import astar
from visualize import draw_detections, draw_path, get_smooth_path
import config

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Crowd-Aware Pathfinding")
st.title("🏃‍♂️ Crowd-Aware Pathfinding")

# --- Model Caching ---
# Cache the YOLO model so it only loads once
@st.cache_resource
def get_yolo_model():
    st.info("Loading YOLOv10 model... This may take a moment.")
    return load_model()

model = get_yolo_model()

# --- Helper Function ---
def get_first_frame(video_file):
    """Reads the first frame of an uploaded video file."""
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        cap = cv2.VideoCapture(tfile.name)
        ret, frame = cap.read()
        cap.release()
        if ret:
            # We must convert from BGR (OpenCV) to RGB (Streamlit)
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), frame
    return None, None

# --- Session State Initialization ---
# This holds our app's "memory"
if 'start_point' not in st.session_state:
    st.session_state.start_point = None
if 'end_point' not in st.session_state:
    st.session_state.end_point = None
if 'run_processing' not in st.session_state:
    st.session_state.run_processing = False

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Controls")
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

    # Add sliders for tuning your algorithm
    st.session_state.crowd_weight = st.slider(
        "Crowd Weight (in pathfinding.py)", 
        min_value=5.0, max_value=50.0, value=10.0, step=1.0,
        help="Higher values make the path avoid people more aggressively."
    )
    
    st.session_state.kernel_size = st.slider(
        "Buffer Zone Size (in main.py)", 
        min_value=3, max_value=9, value=3, step=2,
        help="Size of the 'personal space' buffer. (e.g., 3 means 3x3 kernel)."
    )

    if st.button("Reset Start/End Points"):
        st.session_state.start_point = None
        st.session_state.end_point = None
        st.session_state.run_processing = False
        st.experimental_rerun()

# --- Main Page Layout ---
col1, col2 = st.columns([1, 1])

with col1:
    st.header("1. Select Start & End")
    
    if uploaded_file and not st.session_state.run_processing:
        first_frame_rgb, first_frame_bgr = get_first_frame(uploaded_file)
        
        if first_frame_rgb is not None:
            # This component displays the image and waits for a click
            with st.container():
                st.info("Click on the image to set your start and end points.")
                value = streamlit_image_coordinates(first_frame_rgb, key="locator")

                if value:
                    point = (value['x'], value['y'])
                    # Logic to set start and then end
                    if st.session_state.start_point is None:
                        st.session_state.start_point = point
                        st.experimental_rerun()
                    elif st.session_state.end_point is None:
                        # Only set end point if it's different from start
                        if point != st.session_state.start_point:
                            st.session_state.end_point = point
                            st.experimental_rerun()
            
            # Display the selected points
            st.write(f"**Start Point:** {st.session_state.start_point}")
            st.write(f"**End Point:** {st.session_state.end_point}")

            if st.session_state.start_point and st.session_state.end_point:
                st.success("Start and End points selected!")
                if st.button("🚀 Run Pathfinding on Video"):
                    st.session_state.run_processing = True
                    st.experimental_rerun()
        
    elif not uploaded_file:
        st.info("Please upload a video in the sidebar to begin.")
    
    elif st.session_state.run_processing:
        st.info("Processing video... See the result on the right. \n\n (Press 'Reset' in sidebar to stop.)")


with col2:
    st.header("2. Processed Video")
    
    # This is where the video will play
    video_placeholder = st.empty()

    if not st.session_state.run_processing:
         video_placeholder.info("The processed video will appear here after you select points and click 'Run'.")


# --- Main Processing Loop ---
# This runs only when the "Run" button has been pressed
if st.session_state.run_processing and uploaded_file:
    
    # Get the static start/end points
    start_pixel = st.session_state.start_point
    end_pixel = st.session_state.end_point
    
    # Get the algorithm tuning parameters
    CROWD_WEIGHT = st.session_state.crowd_weight
    K_SIZE = st.session_state.kernel_size
    kernel = np.ones((K_SIZE, K_SIZE), np.uint8)

    # Save the uploaded file to a temporary path
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties for coordinate mapping
    # We use the *original* video dimensions for all calculations
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # We need the dimensions of the *displayed* image from col1 to scale our clicks
    # (Assuming first_frame_bgr is still in scope from the col1 logic)
    h_display, w_display = first_frame_bgr.shape[:2]

    # --- Convert Pixel Coordinates to Grid Coordinates ---
    # This scales the click (from displayed image) to the original video dimensions
    # Then converts to grid coordinates
    
    # Scale x, y from displayed image to original video
    scale_x = w_orig / w_display
    scale_y = h_orig / h_display
    
    sx_orig, sy_orig = int(start_pixel[0] * scale_x), int(start_pixel[1] * scale_y)
    ex_orig, ey_orig = int(end_pixel[0] * scale_x), int(end_pixel[1] * scale_y)
    
    # Convert original pixel coords to grid coords
    start_node = (int((sy_orig / h_orig) * config.GRID_SIZE), int((sx_orig / w_orig) * config.GRID_SIZE))
    goal_node = (int((ey_orig / h_orig) * config.GRID_SIZE), int((ex_orig / w_orig) * config.GRID_SIZE))

    while cap.isOpened() and st.session_state.run_processing:
        ret, frame = cap.read()
        if not ret:
            st.write("Video processing finished.")
            st.session_state.run_processing = False
            break

        # --- 1. Run your FULL pipeline (from main.py) ---
        boxes = detect_people(model, frame)
        crowd_map = generate_crowd_map(frame, boxes)
        
        dilated_map = cv2.dilate(crowd_map, kernel, iterations=1)
        normalized_cost_map = normalize_grid(dilated_map)
        
        # --- 2. Run A* (with tuned weight) ---
        # Modify astar to accept the crowd weight
        # For now, we modify the map directly (a small hack)
        # A better way would be to pass CROWD_WEIGHT to astar
        weighted_cost_map = normalized_cost_map * CROWD_WEIGHT
        
        path = astar(weighted_cost_map, start_node, goal_node)
        
        # --- 3. Draw on the frame ---
        annotated_frame = draw_detections(frame.copy(), boxes)

        if path:
            # We pass 'frame' to get_smooth_path for correct dimensions
            smooth_pts_cv2 = get_smooth_path(frame, path)
            annotated_frame = draw_path(annotated_frame, path, smooth_pts_cv2)
        
        # --- 4. Display in Streamlit ---
        # Convert back to RGB for display and show in the placeholder
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(annotated_frame_rgb, use_column_width=True)

    cap.release()