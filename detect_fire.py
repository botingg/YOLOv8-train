import cv2
from ultralytics import YOLO

# 確定你的模型路徑
model = YOLO(r"C:\Users\User\Downloads\yolov8_fire\runs\detect\train\weights\best.pt")

# 測試影片
cap = cv2.VideoCapture(r"C:\Users\User\Downloads\yolov8_fire\burn.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # 推理
    results = model(frame)
    
    # 直接畫出結果，如果模型學會了，框就會出現
    annotated_frame = results[0].plot()
    
    cv2.imshow("Detection", annotated_frame)
    if cv2.waitKey(1) == 27: break

cap.release()
cv2.destroyAllWindows()