# detection.py
import numpy as np
from ultralytics import YOLO
from config import YOLO_MODEL_PATH, CONF_THRESHOLD, IOU_THRESHOLD, GRID_SIZE

def load_model():
    return YOLO(YOLO_MODEL_PATH)

#finds all the bounding boxes for people in an image.
def detect_people(model, image):
    results = model.predict(source=image, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
    person_boxes = []
    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) == 0:  # class 0 = person
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                person_boxes.append((x1, y1, x2, y2))
    return person_boxes

# Generates a crowd density map based on detected people
def generate_crowd_map(image, boxes):
    h, w = image.shape[:2]
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=float)
    for (x1, y1, x2, y2) in boxes:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        gx = int((cx / w) * GRID_SIZE)
        gy = int((cy / h) * GRID_SIZE)
        if 0 <= gx < GRID_SIZE and 0 <= gy < GRID_SIZE:
            grid[gy, gx] += 1  # Add crowd cost
    return grid
