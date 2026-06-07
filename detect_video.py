from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "runs" / "detect" / "train-7" / "weights" / "best.pt"
VIDEOS = [
    ROOT / "burn.mp4",
    ROOT / "burn1.mp4",
    ROOT / "burn2.mp4",
]
OUTPUT_DIR = ROOT / "outputs"


def find_fire_by_color(frame, previous_frame=None):
    """Return boxes around bright red/orange/yellow regions that look like fire."""
    bgr_b, bgr_g, bgr_r = cv2.split(frame)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)

    red_dominant = bgr_r.astype(np.int16) > bgr_b.astype(np.int16) + 40
    orange_core = (
        (hue <= 35)
        & (saturation >= 90)
        & (value >= 150)
        & (bgr_r >= 140)
        & (bgr_g >= 60)
        & (bgr_b <= 130)
        & red_dominant
    )
    bright_yellow_core = (
        (hue <= 45)
        & (saturation >= 80)
        & (value >= 190)
        & (bgr_r >= 180)
        & (bgr_g >= 120)
        & (bgr_b <= 140)
    )
    flame_seed = (
        (hue <= 45)
        & (saturation >= 55)
        & (value >= 205)
        & (bgr_r >= 205)
        & (bgr_g >= 135)
        & (bgr_b <= 155)
        & (bgr_r.astype(np.int16) > bgr_b.astype(np.int16) + 45)
    )

    warm_mask = ((orange_core | bright_yellow_core).astype(np.uint8)) * 255
    seed_mask = (flame_seed.astype(np.uint8)) * 255

    seed_kernel = np.ones((15, 15), np.uint8)
    nearby_seed = cv2.dilate(seed_mask, seed_kernel, iterations=1)
    mask = cv2.bitwise_and(warm_mask, nearby_seed)

    if previous_frame is not None:
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
        motion = cv2.absdiff(current_gray, previous_gray)
        motion = cv2.threshold(motion, 10, 255, cv2.THRESH_BINARY)[1]
        motion = cv2.dilate(motion, np.ones((21, 21), np.uint8), iterations=1)
        motion_filtered = cv2.bitwise_and(mask, motion)
        if cv2.countNonZero(motion_filtered) > 30:
            mask = motion_filtered

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = frame.shape[0] * frame.shape[1]
    min_area = max(250, int(frame_area * 0.0008))
    height_limit = int(frame.shape[0] * 0.08)

    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w < 12 or h < 12:
            continue
        if x <= 2 or x + w >= frame.shape[1] - 2:
            continue
        if y < height_limit:
            continue

        aspect = w / float(h)
        fill_ratio = area / float(w * h)
        if aspect > 1.9 or aspect < 0.18:
            continue
        if fill_ratio < 0.12:
            continue

        boxes.append((x, y, x + w, y + h, area))

    return merge_overlapping_boxes(boxes)


def merge_overlapping_boxes(boxes):
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda box: box[4], reverse=True)
    merged = []

    for x1, y1, x2, y2, _ in boxes:
        current = np.array([x1, y1, x2, y2], dtype=np.int32)
        did_merge = False

        for index, existing in enumerate(merged):
            if box_iou(current, existing) > 0.08 or boxes_are_close(current, existing):
                merged[index] = np.array(
                    [
                        min(existing[0], current[0]),
                        min(existing[1], current[1]),
                        max(existing[2], current[2]),
                        max(existing[3], current[3]),
                    ],
                    dtype=np.int32,
                )
                did_merge = True
                break

        if not did_merge:
            merged.append(current)

    merged_boxes = [tuple(map(int, box)) for box in merged]
    if not merged_boxes:
        return []

    areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in merged_boxes]
    max_area = max(areas)
    return [
        box
        for box, area in zip(merged_boxes, areas)
        if area >= max_area * 0.35
    ]


def box_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection == 0:
        return 0.0

    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / float(area_a + area_b - intersection)


def boxes_are_close(a, b):
    horizontal_gap = max(0, max(a[0], b[0]) - min(a[2], b[2]))
    vertical_gap = max(0, max(a[1], b[1]) - min(a[3], b[3]))
    return horizontal_gap < 35 and vertical_gap < 35


def draw_boxes(frame, boxes, label):
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(
            frame,
            label,
            (x1, max(28, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )


def detect_yolo_boxes(model, frame, conf):
    result = model.predict(frame, conf=conf, imgsz=960, verbose=False)[0]
    boxes = []

    if result.boxes is None:
        return boxes

    for xyxy, score in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
        x1, y1, x2, y2 = map(int, xyxy)
        boxes.append((x1, y1, x2, y2, float(score)))

    return boxes


def process_video(model, video_path, conf=0.08):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / f"{video_path.stem}_fire_detected.mp4"
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    yolo_frames = 0
    color_frames = 0
    frame_index = 0
    previous_frame = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        yolo_boxes = detect_yolo_boxes(model, frame, conf)
        if yolo_boxes:
            boxes = [(x1, y1, x2, y2) for x1, y1, x2, y2, _ in yolo_boxes]
            draw_boxes(frame, boxes, "fire")
            yolo_frames += 1
        else:
            boxes = find_fire_by_color(frame, previous_frame)
            draw_boxes(frame, boxes, "fire")
            if boxes:
                color_frames += 1

        writer.write(frame)
        previous_frame = frame.copy()
        frame_index += 1

        if frame_index % 100 == 0:
            print(f"{video_path.name}: {frame_index}/{total_frames} frames")

    cap.release()
    writer.release()

    print(
        f"Saved {output_path} | YOLO frames: {yolo_frames}, "
        f"color fallback frames: {color_frames}"
    )
    return output_path


def main():
    if not WEIGHTS.exists():
        raise FileNotFoundError(f"Missing model weights: {WEIGHTS}")

    model = YOLO(str(WEIGHTS))
    print(f"Using model: {WEIGHTS}")
    print(f"Classes: {model.names}")

    for video_path in VIDEOS:
        process_video(model, video_path)


if __name__ == "__main__":
    main()
