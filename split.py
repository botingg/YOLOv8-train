import os
import shutil
import random
from pathlib import Path

# 設定路徑
base_path = Path(r"C:\Users\User\Downloads\yolov8_fire\dataset")
src_images = base_path / "images"
train_img_path = base_path / "train/images"
train_lbl_path = base_path / "train/labels"
val_img_path = base_path / "val/images"
val_lbl_path = base_path / "val/labels"

# 建立目標資料夾
for p in [train_img_path, train_lbl_path, val_img_path, val_lbl_path]:
    p.mkdir(parents=True, exist_ok=True)

# 取得所有圖片檔 (假設都是 .jpg)
files = [f for f in src_images.glob("*.jpg")]
random.shuffle(files)

# 設定切分比例 (80% 訓練, 20% 驗證)
split_idx = int(len(files) * 0.8)
train_files = files[:split_idx]
val_files = files[split_idx:]

def move_files(file_list, dest_img_dir, dest_lbl_dir):
    for f in file_list:
        # 移動圖片
        shutil.move(str(f), str(dest_img_dir / f.name))
        
        # 移動對應的標籤檔 (假設檔名相同，副檔名為 .txt)
        lbl_file = f.with_suffix(".txt")
        if lbl_file.exists():
            shutil.move(str(lbl_file), str(dest_lbl_dir / lbl_file.name))

# 執行分配
move_files(train_files, train_img_path, train_lbl_path)
move_files(val_files, val_img_path, val_lbl_path)

print(f"處理完成！已將 {len(files)} 張圖片分為訓練集與驗證集。")