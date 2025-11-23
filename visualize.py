# visualize.py

import cv2
import numpy as np
from scipy.interpolate import splprep, splev
from config import MAX_DISPLAY_WIDTH, GRID_SIZE

# --- NEW FUNCTION: Overlay path on video frames ---
def overlay_path(frame, path, grid_size):
    """
    Draws the A* path on a given video frame.
    Converts grid coordinates to pixel coordinates and draws yellow lines.
    Used in app2.py for video visualization.
    """
    if not path or frame is None:
        return frame

    h, w = frame.shape[:2]
    cell_h, cell_w = h / grid_size, w / grid_size

    pixel_points = []
    for (gy, gx) in path:
        px = int((gx + 0.5) * cell_w)
        py = int((gy + 0.5) * cell_h)
        pixel_points.append((px, py))

    # Draw the path as connected yellow lines
    for i in range(1, len(pixel_points)):
        cv2.line(frame, pixel_points[i - 1], pixel_points[i], (0, 255, 255), 2)

    # Draw small circles along the path for better visibility
    for (px, py) in pixel_points:
        cv2.circle(frame, (px, py), 3, (255, 255, 0), -1)

    return frame

def resize_for_display(image):
    h, w = image.shape[:2]
    if w > MAX_DISPLAY_WIDTH:
        scale = MAX_DISPLAY_WIDTH / w
        new_w, new_h = int(w * scale), int(h * scale)
        image = cv2.resize(image, (new_w, new_h))
    return image

def draw_detections(image, boxes):
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return image

def get_smooth_path(image, path, num_points=100):
    if not path or len(path) < 4:
        # st.warning("Path too short to smooth, drawing jagged path.") # Streamlit not available here
        return None 

    h, w = image.shape[:2]
    
    pixel_points = []
    for (gy, gx) in path:
        x = int((gx + 0.5) * w / GRID_SIZE)
        y = int((gy + 0.5) * h / GRID_SIZE)
        pixel_points.append((x, y))

    pts = np.array(pixel_points).T 
    
    smoothing_factor = len(path) * 2.5 
        
    tck, u = splprep([pts[0], pts[1]], s=smoothing_factor, k=3)
    
    if tck is None:
        return None

    u_new = np.linspace(u.min(), u.max(), num_points)
    x_new, y_new = splev(u_new, tck)

    smooth_pts_list = [(int(x), int(y)) for x, y in zip(x_new, y_new)]
    smooth_pts_cv2 = np.array([smooth_pts_list], dtype=np.int32)
    
    return smooth_pts_cv2


def draw_path(image, path, smooth_pts_cv2):
    """
    Draws the path on the image.
    If smooth_pts_cv2 is provided, it draws the smooth path.
    Otherwise, it draws the jagged A* path.
    """
    
    # --- Draw the SMOOTH path in LIGHTER RED ---
    if smooth_pts_cv2 is not None:
        # BGR for OpenCV: (Blue, Green, Red)
        LIGHT_RED = (80, 80, 255) # This is a lighter red, adjust as needed
        cv2.polylines(image, [smooth_pts_cv2], isClosed=False, color=LIGHT_RED, thickness=2, lineType=cv2.LINE_AA)
        return image

    # --- Fallback: Draw the original JAGGED path (e.g., in yellow for distinction) ---
    if not path or len(path) < 2:
        return image

    h, w = image.shape[:2]
    pixel_points = []
    
    for (gy, gx) in path:
        x = int((gx + 0.5) * w / GRID_SIZE)
        y = int((gy + 0.5) * h / GRID_SIZE)
        pixel_points.append((x, y))
    
    pts = np.array([pixel_points], dtype=np.int32)
    YELLOW = (0, 255, 255) # Fallback to yellow for jagged path
    cv2.polylines(image, [pts], isClosed=False, color=YELLOW, thickness=1, lineType=cv2.LINE_AA)
    
    return image

# --- NEW FUNCTION TO DRAW START/END POINTS ---
def draw_start_end_points(image, start_pixel, end_pixel):
    """
    Draws black circles at the start and end pixel coordinates.
    """
    BLACK = (0, 0, 0)
    RADIUS = 8 # Size of the dot
    THICKNESS = -1 # -1 means filled circle
    
    if start_pixel:
        cv2.circle(image, start_pixel, RADIUS, BLACK, THICKNESS)
    if end_pixel:
        cv2.circle(image, end_pixel, RADIUS, BLACK, THICKNESS)
        
    return image

def select_points(image):
    # This function is not used in the Streamlit app directly for selection,
    # but it might be kept for local debugging or future features.
    # The Streamlit app uses streamlit_image_coordinates instead.
    points = []
    window_name = "Select Start and End (Click 2 points, then press any key)"
    
    def callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            print(f"Selected point {len(points)}: {(x, y)}")
            cv2.imshow(window_name, image)
            
    cv2.imshow(window_name, image)
    cv2.setMouseCallback(window_name, callback)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return points if len(points) == 2 else None