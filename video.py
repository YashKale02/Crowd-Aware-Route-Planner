import cv2
import numpy as np
from config import GRID_SIZE, YOLO_MODEL_PATH
from detection import load_model, detect_people, generate_crowd_map
from pathfinding import astar
from visualize import draw_detections, draw_path, get_smooth_path

VIDEO_SOURCE = 0  # Change to filename if needed
START_POINT = (50, 50)
END_POINT = (400, 400)

def main():
    print(f"Loading model from {YOLO_MODEL_PATH}...")
    model = load_model()
    
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("Error opening video.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        h, w = frame.shape[:2]

        # 1. Detection
        boxes = detect_people(model, frame)
        annotated_frame = draw_detections(frame.copy(), boxes)

        # 2. Mapping
        crowd_map = generate_crowd_map(frame, boxes)
        
        # Dilation (Safety Buffer)
        kernel = np.ones((5, 5), np.uint8)
        dilated_map = cv2.dilate(crowd_map, kernel, iterations=1)
        
        max_val = np.max(dilated_map)
        cost_map = dilated_map / max_val if max_val > 0 else dilated_map

        # 3. Coordinate Conversion
        sx = int((START_POINT[0] / w) * GRID_SIZE)
        sy = int((START_POINT[1] / h) * GRID_SIZE)
        ex = int((END_POINT[0] / w) * GRID_SIZE)
        ey = int((END_POINT[1] / h) * GRID_SIZE)

        # Bounds Check
        sx, sy = max(0, min(sx, GRID_SIZE-1)), max(0, min(sy, GRID_SIZE-1))
        ex, ey = max(0, min(ex, GRID_SIZE-1)), max(0, min(ey, GRID_SIZE-1))

        # 4. Pathfinding
        grid_path = astar(cost_map, (sy, sx), (ey, ex))

        if grid_path:
            smooth_path = get_smooth_path(annotated_frame, grid_path)
            annotated_frame = draw_path(annotated_frame, grid_path, smooth_path)

        # Draw Endpoints
        cv2.circle(annotated_frame, START_POINT, 8, (255, 0, 0), -1)
        cv2.circle(annotated_frame, END_POINT, 8, (0, 255, 0), -1)

        cv2.imshow('Crowd Navigation', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()