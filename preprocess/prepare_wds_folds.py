import os
import random
import io
import math
import shutil
import json
import multiprocessing
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import webdataset as wds  # 引入 wds 库

# ================= 配置区域 =================
SOURCE_DIR = "/home/jiangwentao/Repos/foundation-agriculture/datasets/AgriNet-1K/all"
OUTPUT_DIR = "/home/jiangwentao/Repos/foundation-agriculture/datasets/AgriNet-1K/wds_folds"
NUM_FOLDS = 4
TARGET_SIZE = 256
SHARD_SIZE = 2000
NUM_WORKERS = 16 
# ===========================================

def safe_write_shard(args):
    """
    使用 webdataset.TarWriter 写入 Shard
    """
    shard_path, image_data, class_to_idx = args
    shard_path = Path(shard_path)
    tmp_path = shard_path.with_suffix('.tar.tmp')
    
    # 简单的存在性检查 (断点续传)
    if shard_path.exists() and shard_path.stat().st_size > 1024:
        return len(image_data)

    written_count = 0
    try:
        # 使用 wds.TarWriter 打开文件
        # encoder=True 使用默认编码器 (自动处理 img -> jpg bytes, str -> utf-8 bytes)
        with wds.TarWriter(str(tmp_path)) as sink:
            for unique_id, img_path, class_name in image_data:
                try:
                    if class_name not in class_to_idx:
                        print(f"[Warning] Class {class_name} not found in map!")
                        continue
                    
                    label_idx = class_to_idx[class_name]
                    
                    # 1. 图片预处理 (Resize)
                    with Image.open(img_path) as img:
                        img = img.convert('RGB')
                        w, h = img.size
                        ratio = TARGET_SIZE / min(w, h)
                        new_size = (int(w * ratio), int(h * ratio))
                        img = img.resize(new_size, Image.BILINEAR)
                        
                        # 转换为二进制流
                        img_byte_arr = io.BytesIO()
                        img.save(img_byte_arr, format='JPEG', quality=95)
                        img_bytes = img_byte_arr.getvalue()
                        
                    # 2. 准备 Key (保留原始文件名)
                    key = Path(img_path).stem 

                    # 3. 构建样本并写入
                    sample = {
                        "__key__": key,
                        "jpg": img_bytes,
                        "cls": str(class_name)
                    }
                    
                    sink.write(sample)
                    written_count += 1
                    
                except Exception as e:
                    print(f"[Warning] Skipped {img_path}: {e}")
                    continue
        
        # 写入完成，重命名
        if written_count > 0:
            os.rename(tmp_path, shard_path)
            return written_count
        else:
            if tmp_path.exists(): os.remove(tmp_path)
            return 0

    except Exception as e:
        print(f"[Error] Failed shard {shard_path}: {e}")
        if tmp_path.exists(): os.remove(tmp_path)
        return 0

def main():
    source = Path(SOURCE_DIR)
    output = Path(OUTPUT_DIR)
    
    # 防止意外覆盖，建议手动清理或确认
    # if output.exists(): shutil.rmtree(output) 
    output.mkdir(parents=True, exist_ok=True)

    # 1. 建立类别映射
    print(f"Scanning classes in {source}...")
    # 确保排序稳定
    classes = sorted([d.name for d in source.iterdir() if d.is_dir()])
    
    if not classes:
        print(f"Error: No class directories found in {source}")
        return

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    print(f"Found {len(classes)} classes.")
    
    # 保存 class map
    with open(output / "class_map.json", "w") as f:
        json.dump(class_to_idx, f, indent=4)
    print(f"Class map saved to {output}/class_map.json")

    # 2. 收集所有图片
    all_images_info = []
    class_dirs = [source / cls for cls in classes]
    
    print("Listing images...")
    # 使用 tqdm 显示扫描进度
    for cls_dir in tqdm(class_dirs):
        cls_name = cls_dir.name
        # 扫描常见图片后缀
        for img_file in cls_dir.iterdir():
            if img_file.is_file() and img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                all_images_info.append((None, img_file, cls_name)) # 第一个参数 unique_id 暂时不用，占位
    
    total_images = len(all_images_info)
    print(f"Total images: {total_images}. Shuffling...")
    
    if total_images == 0:
        print("Error: No images found.")
        return

    # 3. 随机打乱
    random.seed(42)
    random.shuffle(all_images_info)
    
    # 4. 构建任务列表 (按 Fold 切分 -> 按 Shard 切分)
    fold_size = math.ceil(total_images / NUM_FOLDS)
    tasks = []
    
    for fold_idx in range(NUM_FOLDS):
        start_idx = fold_idx * fold_size
        end_idx = min((fold_idx + 1) * fold_size, total_images)
        
        fold_subset = all_images_info[start_idx:end_idx]
        
        fold_dir = output / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        
        num_shards = math.ceil(len(fold_subset) / SHARD_SIZE)
        
        for i in range(num_shards):
            s_start = i * SHARD_SIZE
            s_end = min((i + 1) * SHARD_SIZE, len(fold_subset))
            shard_data = fold_subset[s_start:s_end]
            
            # 格式化文件名: shard_00000.tar
            shard_path = fold_dir / f"shard_{i:05d}.tar"
            
            tasks.append((shard_path, shard_data, class_to_idx))

    print(f"Starting generation of {len(tasks)} shards with {NUM_WORKERS} workers...")
    
    # 5. 多进程执行
    # 注意：Pool 放在 with 语句中自动管理资源
    with multiprocessing.Pool(NUM_WORKERS) as pool:
        # 使用 imap_unordered 可以让进度条更平滑地更新
        results = list(tqdm(pool.imap_unordered(safe_write_shard, tasks), total=len(tasks)))

    # 6. 最终统计
    print("\nVerifying final counts...")
    final_counts = []
    for i in range(NUM_FOLDS):
        # 简单的数学统计，实际可以用 wds 命令校验
        start = i * fold_size
        end = min((i + 1) * fold_size, total_images)
        count = end - start
        final_counts.append(count)
        print(f"Fold {i}: ~{count} images")

    print("\n" + "="*50)
    print("DONE.")
    print("="*50)
    counts_str = " ".join([str(c) for c in final_counts])
    print(f"FOLD_COUNTS=({counts_str})")

if __name__ == "__main__":
    main()