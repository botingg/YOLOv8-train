import cv2
import time
from ultralytics import YOLO
from pathlib import Path

# =====================================
# 載入 YOLO 模型
# =====================================
model = YOLO("./weights/best.pt")

# =====================================
# 影片列表
# =====================================
video_list = [
    "burn.mp4",
    "burn1.mp4",
    "burn2.mp4"
]

# =====================================
# 建立輸出資料夾
# =====================================
Path("outputs").mkdir(exist_ok=True)

# =====================================
# 開始處理影片
# =====================================
for video_path in video_list:

    print(f"\nProcessing: {video_path}")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Cannot open {video_path}")
        continue

    # 影片資訊
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 輸出檔名
    output_path = f"outputs/output_{video_path}"

    # mp4 encoder
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    # FPS 計算
    prev_time = time.time()

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # =====================================
        # YOLO inference
        # =====================================
        results = model(
            frame,
            imgsz=640,
            conf=0.4,
            verbose=False
        )

        # 畫框
        annotated_frame = results[0].plot()

        # =====================================
        # FPS 計算
        # =====================================
        current_time = time.time()

        fps_display = 1 / (current_time - prev_time)

        prev_time = current_time

        # 顯示 FPS
        cv2.putText(
            annotated_frame,
            f"FPS: {fps_display:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # 顯示影片
        cv2.imshow("YOLO Fire Detection", annotated_frame)

        # 寫入影片
        out.write(annotated_frame)

        # ESC 離開
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    out.release()

    print(f"Saved: {output_path}")

cv2.destroyAllWindows()

print("\nAll videos completed.")