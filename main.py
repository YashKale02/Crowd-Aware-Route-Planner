# main.py

import cv2
import numpy as np
from detection import load_model, detect_people, generate_crowd_map
from graph import normalize_grid
from pathfinding import astar
from visualize import draw_detections, draw_path, select_points, resize_for_display, get_smooth_path
from config import GRID_SIZE

def main():
    model = load_model()
    image = cv2.imread("data/input/image2.jpg")

    # Detect people
    boxes = detect_people(model, image)
    annotated = draw_detections(image.copy(), boxes)
    annotated = resize_for_display(annotated)

    # Select start & end points
    points = select_points(annotated)
    if not points or len(points) < 2:
        print("Please select two points.")
        return
    start, end = points[:2]

    # Generate crowd map
    crowd_map = generate_crowd_map(image, boxes)
    crowd_map = normalize_grid(crowd_map)

    # 2. NEW: Create the buffer zone (Path Clearance)
    #    'kernel_size' controls how big the buffer is. 
    #    A (3,3) kernel affects 1 cell in every direction.
    #    A (5,5) kernel affects 2 cells. Start with (3,3).
    kernel_size = (5, 5) 
    kernel = np.ones(kernel_size, np.uint8)
    
    # 'dilate' will take the max value (the crowd count) and 
    # spread it to the neighboring cells defined by the kernel.
    dilated_map = cv2.dilate(crowd_map, kernel, iterations=1)

    # 3. Normalize the NEW 'dilated_map'
    normalized_cost_map = normalize_grid(dilated_map)

    # Convert to grid coordinates
    h_annot, w_annot = annotated.shape[:2] 
    sx, sy = int((start[0]/w_annot)*GRID_SIZE), int((start[1]/h_annot)*GRID_SIZE)
    ex, ey = int((end[0]/w_annot)*GRID_SIZE), int((end[1]/h_annot)*GRID_SIZE)
    start_node = (sy, sx)
    goal_node = (ey, ex)
    
    # Run A* pathfinding
    path = astar(crowd_map, start_node, goal_node)
    if not path:
        print("No path found!")
        return

    # --- NEW: Generate the smooth path ---
    # We pass 'annotated' to get the correct dimensions for pixel conversion
    smooth_pts_cv2 = get_smooth_path(annotated, path) 

    # --- MODIFIED: Draw path ---
    # Pass both the original 'path' and the new 'smooth_pts_cv2'
    final = draw_path(annotated, path, smooth_pts_cv2)
    
    cv2.imshow("Optimized Route", final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()