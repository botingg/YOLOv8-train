import cv2
import os

videos = ["burn.mp4", "burn1.mp4", "burn2.mp4"]

os.makedirs("dataset/images", exist_ok=True)

count = 0

for v in videos:
    cap = cv2.VideoCapture(v)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % 5 == 0:
            cv2.imwrite(f"dataset/images/img_{count}.jpg", frame)

        count += 1

cap.release()

print("Done extracting frames")
