from ultralytics import YOLO
import os

# 1. 檢查資料集是否存在
data_yaml = "C:/Users/User/Downloads/yolov8_fire/data.yaml"
if not os.path.exists(data_yaml):
    # 自動建立 data.yaml
    with open(data_yaml, "w") as f:
        f.write("path: C:/Users/User/Downloads/yolov8_fire/dataset\n")
        f.write("train: train/images\n")
        f.write("val: val/images\n")
        f.write("names:\n  0: fire\n")
    print("已自動建立 data.yaml")

# 2. 載入模型 (使用 yolov8n.pt 是最輕量且最快訓練的)
model = YOLO("yolov8n.pt")

# 3. 開始訓練 (強烈建議訓練 50 個 Epoch，這是讓模型學會火的必要次數)
print("開始訓練中，請耐心等待...")
model.train(
    data=data_yaml, 
    epochs=50, 
    imgsz=640, 
    batch=8, 
    device="cpu" # 如果你有顯卡，這會快很多
)

print("訓練完成！模型儲存在 runs/detect/train/weights/best.pt")