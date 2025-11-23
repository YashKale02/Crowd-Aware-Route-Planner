# app_image.py

import streamlit as st
import cv2
import numpy as np
import io

# Import all your existing logic
from detection import load_model, detect_people, generate_crowd_map
from graph import normalize_grid
from pathfinding import astar
from visualize import draw_detections, draw_path, get_smooth_path, resize_for_display, draw_start_end_points
import config
from streamlit_image_coordinates import streamlit_image_coordinates

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Image Route Finder")
st.title("🏃‍♂️ Image Route Finder")

# --- Model Caching ---
@st.cache_resource
def get_yolo_model():
    st.info("Loading YOLOv10 model... This may take a moment.")
    return load_model()

model = get_yolo_model()

# --- Session State Initialization ---
if 'start_point' not in st.session_state:
    st.session_state.start_point = None
if 'end_point' not in st.session_state:
    st.session_state.end_point = None
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'original_image_bgr' not in st.session_state:
    st.session_state.original_image_bgr = None
if 'original_image_rgb' not in st.session_state:
    st.session_state.original_image_rgb = None

# --- Helper Function ---
def load_image():
    """Loads and resizes an uploaded image from session_state."""
    
    image_file = st.session_state.file_uploader
    
    if image_file is not None:
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        img_bgr = resize_for_display(img_bgr)
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        st.session_state.original_image_bgr = img_bgr
        st.session_state.original_image_rgb = img_rgb
    
    st.session_state.start_point = None
    st.session_state.end_point = None
    st.session_state.processed_image = None

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Controls")
    
    uploaded_file = st.file_uploader(
        "Upload an image", 
        type=["png", "jpg", "jpeg"],
        on_change=load_image,
        key="file_uploader"
    )

    st.session_state.crowd_weight = st.slider(
        "Crowd Weight", 
        min_value=5.0, max_value=50.0, value=10.0, step=1.0,
        help="Higher values make the path avoid people more aggressively."
    )
    
    st.session_state.kernel_size = st.slider(
        "Buffer Zone Size", 
        min_value=3, max_value=9, value=3, step=2,
        help="Size of the 'personal space' buffer. (e.g., 3 means 3x3 kernel)."
    )

    if st.button("Reset Start/End Points"):
        st.session_state.start_point = None
        st.session_state.end_point = None
        st.session_state.processed_image = None
        st.rerun()

# --- Main Page Layout ---
col1, col2 = st.columns([1, 1])

with col1:
    st.header("1. Select Start & End")
    
    if st.session_state.original_image_rgb is not None:
        # --- MODIFIED: Display the image with current start/end points marked ---
        # Create a copy so we don't draw on the original cached image
        display_image_with_points = st.session_state.original_image_bgr.copy()
        
        # Draw small circles at the start and end points
        # Using a distinct color like green for selection feedback
        if st.session_state.start_point:
            cv2.circle(display_image_with_points, st.session_state.start_point, 5, (0, 255, 0), -1) # Green circle
        if st.session_state.end_point:
            cv2.circle(display_image_with_points, st.session_state.end_point, 5, (0, 255, 0), -1) # Green circle

        # Convert to RGB for Streamlit display
        display_image_with_points_rgb = cv2.cvtColor(display_image_with_points, cv2.COLOR_BGR2RGB)

        with st.container():
            st.info("Click on the image to set your start and end points.")
            # Use the image with points drawn for selection
            value = streamlit_image_coordinates(display_image_with_points_rgb, key="locator")

            if value:
                point = (value['x'], value['y'])
                if st.session_state.start_point is None:
                    st.session_state.start_point = point
                    st.rerun()
                elif st.session_state.end_point is None and point != st.session_state.start_point:
                    st.session_state.end_point = point
                    st.rerun()
        
        st.write(f"**Start Point:** {st.session_state.start_point}")
        st.write(f"**End Point:** {st.session_state.end_point}")

        if st.session_state.start_point and st.session_state.end_point:
            st.success("Points selected!")
            if st.button("🚀 Run Pathfinding"):
                
                image_to_process = st.session_state.original_image_bgr.copy()
                
                CROWD_WEIGHT = st.session_state.crowd_weight
                K_SIZE = st.session_state.kernel_size
                kernel = np.ones((K_SIZE, K_SIZE), np.uint8)
                
                h, w = image_to_process.shape[:2]

                boxes = detect_people(model, image_to_process)
                
                crowd_map = generate_crowd_map(image_to_process, boxes)
                dilated_map = cv2.dilate(crowd_map, kernel, iterations=1)
                normalized_cost_map = normalize_grid(dilated_map)
                
                weighted_cost_map = normalized_cost_map * CROWD_WEIGHT
                
                start_pixel = st.session_state.start_point
                end_pixel = st.session_state.end_point
                
                start_node = (int((start_pixel[1] / h) * config.GRID_SIZE), int((start_pixel[0] / w) * config.GRID_SIZE))
                goal_node = (int((end_pixel[1] / h) * config.GRID_SIZE), int((end_pixel[0] / w) * config.GRID_SIZE))

                path = astar(weighted_cost_map, start_node, goal_node)
                
                annotated_image = draw_detections(image_to_process, boxes)
                if path:
                    smooth_pts_cv2 = get_smooth_path(annotated_image, path)
                    annotated_image = draw_path(annotated_image, path, smooth_pts_cv2)
                
                # --- NEW: Draw final black start/end points on the processed image ---
                annotated_image = draw_start_end_points(annotated_image, start_pixel, end_pixel)
                
                st.session_state.processed_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                st.rerun()

    else:
        st.info("Please upload an image in the sidebar to begin.")

with col2:
    st.header("2. Processed Image")
    
    if st.session_state.processed_image is not None:
        st.image(st.session_state.processed_image, use_container_width=True)
    else:
        st.info("The final path will appear here after you select points and click 'Run'.")