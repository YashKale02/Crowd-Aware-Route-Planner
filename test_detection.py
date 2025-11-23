import cv2
from ultralytics import YOLO

model = YOLO("yolov8x.pt")
image_path = "data/input/image1.png"
video_path = "data/input/testvideo.mp4"

results = model.predict(source=image_path, 
                        conf=0.35, 
                        iou=0.45, 
                        imgsz=1280, 
                        classes=[0], 
                        verbose=False)
annotated_image = results[0].plot()  # YOLO draws boxes

# ---- Resize for display ----
max_display_width = 1000  # adjust as you like
h, w = annotated_image.shape[:2]  # .shape gives (height, width, channels) and we don't need channels so we take first two
scale = max_display_width / w  # scale factor to fit width e.g. original width is 4000, max is 1000, scale is 0.25 of original
new_w, new_h = int(w * scale), int(h * scale) #calculate new dimensions
resized = cv2.resize(annotated_image, (new_w, new_h))

cv2.imshow("Detections", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()