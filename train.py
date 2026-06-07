from ultralytics import YOLO

# 載入 YOLOv8 nano（最適合 Raspberry Pi）
model = YOLO("yolov8n.pt")

# 開始訓練
model.train(
    data="C:/Users/User/Downloads/yolov8_fire/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device="cpu"  # 有GPU用0，沒GPU改cpu
)