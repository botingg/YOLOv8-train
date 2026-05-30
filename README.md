# YOLOv8 + Raspberry Pi 4 火災偵測系統完整開發指南

## 專案目標

本專案旨在建置一套基於 YOLOv8 與 Raspberry Pi 4 的即時火災偵測系統，利用 Edge AI 技術於本地端完成影像分析，降低雲端運算依賴，提高即時性與系統可靠度。

---

# 1. 專案架構

```text
yolov8-pi-project/
│
├── dataset/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   ├── labels/
│       ├── train/
│       ├── val/
│
├── models/
│   ├── yolov8n.pt
│   ├── best.pt
│   ├── best.onnx
│
├── runs/
│
├── scripts/
│   ├── train.py
│   ├── export_onnx.py
│   ├── detect_pc.py
│   ├── detect_pi.py
│   ├── make_csv.py
│
├── data.yaml
├── requirements.txt
└── README.md
```

---

# 2. 開發環境建置

## 建立 Conda 環境

```bash
conda create -n yolo python=3.10
conda activate yolo
```

## 安裝必要套件

```bash
pip install ultralytics
pip install opencv-python
pip install numpy
pip install pandas
pip install matplotlib
```

## GPU版本（選用）

```bash
pip install torch torchvision torchaudio
```

---

# 3. Dataset 準備

YOLO Dataset格式：

```text
dataset/
├── images/
│   ├── train
│   └── val
│
├── labels/
│   ├── train
│   └── val
```

---

## Label格式

每個標註檔內容：

```text
class_id x_center y_center width height
```

範例：

```text
0 0.52 0.48 0.30 0.40
1 0.20 0.33 0.10 0.15
```

所有座標皆需正規化至0~1。

---

# 4. 建立 data.yaml

```yaml
path: dataset

train: images/train
val: images/val

nc: 2

names:
  0: fire
  1: smoke
```

---

# 5. 訓練 YOLOv8

## train.py

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0
)
```

## 執行訓練

```bash
python train.py
```

---

# 6. 訓練結果分析

輸出目錄：

```text
runs/detect/train/
│
├── weights/
│   ├── best.pt
│   └── last.pt
│
├── confusion_matrix.png
├── results.png
├── PR_curve.png
└── F1_curve.png
```

重要指標：

- Precision
- Recall
- mAP50
- mAP50-95

---

# 7. 模型轉換 ONNX

## export_onnx.py

```python
from ultralytics import YOLO

model = YOLO("best.pt")

model.export(
    format="onnx",
    imgsz=320
)
```

執行：

```bash
python export_onnx.py
```

產生：

```text
best.onnx
```

---

# 8. Raspberry Pi 4 環境建置

## 更新系統

```bash
sudo apt update
sudo apt upgrade
```

## 安裝套件

```bash
sudo apt install python3-pip

pip3 install numpy
pip3 install pandas
pip3 install opencv-python
pip3 install onnxruntime
```

---

# 9. Raspberry Pi 即時推論

## detect_pi.py

```python
import cv2
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession("best.onnx")

input_name = session.get_inputs()[0].name

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    img = cv2.resize(frame, (320,320))

    img = img.astype(np.float32)/255.0
    img = np.transpose(img,(2,0,1))
    img = np.expand_dims(img,0)

    outputs = session.run(
        None,
        {input_name: img}
    )

    cv2.imshow("Detection",frame)

    if cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows()
```

---

# 10. CSV紀錄系統

## 建立CSV

```python
import csv

with open("log.csv","w",newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "timestamp",
        "class",
        "confidence"
    ])
```

---

## 寫入紀錄

```python
import csv
import time

def log_detection(cls,conf):

    with open("log.csv","a",newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            cls,
            conf
        ])
```

---

# 11. FPS優化技巧

## 降低解析度

```python
imgsz = 320
```

---

## 使用 YOLOv8n

```text
YOLOv8n
```

最適合 Raspberry Pi 4

---

## Frame Skip

```python
if frame_id % 2 != 0:
    continue
```

---

## 關閉GUI

避免大量使用：

```python
cv2.imshow()
```

---

## ONNX Runtime

比 PyTorch 快許多。

---

# 12. 系統架構圖

```text
Camera
   │
   ▼
Raspberry Pi 4
   │
   ▼
ONNX Runtime
   │
   ▼
YOLOv8n Detection
   │
   ├── Alarm
   │
   ├── CSV Log
   │
   └── Dashboard
```

---

# 13. 預期效能分析

| 模型 | 解析度 | FPS |
|--------|--------|--------|
| YOLOv8n | 640 | 1~2 |
| YOLOv8n | 320 | 3~6 |
| ONNX | 320 | 5~10 |

---

# 14. 升級架構

## Coral TPU

```text
Pi + Coral TPU
```

推論速度：

```text
10~30 FPS
```

---

## Jetson Nano

```text
Jetson Nano
```

具備 CUDA 支援。

---

## Jetson Orin Nano

```text
40~100 FPS
```

適合工業級部署。

---

# 15. 專題應用價值

## 智慧家庭

- 火災偵測
- 入侵偵測
- 老人跌倒偵測

---

## 智慧工廠

- 設備監控
- 安全警報
- 異常事件分析

---

## 智慧交通

- 車流分析
- 事故偵測
- 違規辨識

---

# 16. 為什麼 Raspberry Pi 4 不適合大模型

## Raspberry Pi 4 硬體限制

| 項目 | 規格 |
|--------|--------|
| CPU | Cortex-A72 4 Core |
| GPU | VideoCore VI |
| RAM | 2GB / 4GB / 8GB |
| CUDA | 無 |
| Tensor Core | 無 |

---

## 模型大小比較

| 模型 | 適用程度 |
|--------|--------|
| YOLOv8n | ✔ |
| YOLOv8s | ⚠ |
| YOLOv8m | ✘ |
| YOLOv8l | ✘ |
| YOLOv8x | ✘ |

---

## 原因1：計算量過大

CNN每層都需要大量卷積運算。

模型越大：

```text
運算量暴增
↓
FPS下降
↓
延遲增加
```

---

## 原因2：RAM不足

Feature Map過大：

```text
RAM耗盡
↓
Swap
↓
SD Card存取
↓
速度暴跌
```

---

## 原因3：缺乏GPU

Pi只能依靠CPU運算。

相較桌機：

```text
效能差距可達10~50倍
```

---

# 17. 未來展望

## Tiny AI

未來模型將持續縮小：

- YOLOv10 Nano
- MobileNet Detector
- Tiny Detector

---

## AI加速器

未來可結合：

- Coral TPU
- Hailo-8
- Jetson系列

---

## 模型壓縮

技術包含：

### Quantization

```text
FP32
↓
INT8
```

---

### Pruning

```text
移除無效權重
```

---

### Knowledge Distillation

```text
大模型教小模型
```

---

## IoT整合

```text
Camera
  ↓
Edge AI
  ↓
MQTT
  ↓
Cloud
  ↓
Dashboard
```

---

# 18. 結論

本系統利用 YOLOv8n 與 Raspberry Pi 4 建立低成本 Edge AI 火災偵測平台。

透過：

- YOLOv8n
- ONNX Runtime
- Raspberry Pi 4
- CSV Logging

可實現即時火災監測與事件紀錄。

由於 Raspberry Pi 4 的 CPU、記憶體及 AI 加速能力有限，因此必須在模型準確率與推論速度之間取得平衡。

YOLOv8n 搭配 ONNX Runtime 與 320×320 解析度，是目前在 Raspberry Pi 4 上最具實務價值且穩定的部署方案。

未來可透過：

- Coral TPU
- Jetson Orin
- Quantization
- Pruning
- MQTT Cloud Integration

進一步提升系統效能與工業應用價值。
