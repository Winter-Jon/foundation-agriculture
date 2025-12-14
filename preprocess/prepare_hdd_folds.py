import os
import tarfile
import random
import io
import math
import shutil
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import multiprocessing

# ================= 配置区域 =================
SOURCE_DIR = "/home/jiangwentao/Repos/foundation-agriculture/datasets/AgriNet-1K/all"
OUTPUT_DIR = "/home/jiangwentao/Repos/foundation-agriculture/datasets/AgriNet-1K/wds_fold_2"
NUM_FOLDS = 2
TARGET_SIZE = 256
SHARD_SIZE = 2000
NUM_WORKERS = 16 
# ===========================================

def safe_write_shard(args):
    """
    写入 Shard
    - Key: 使用原始文件名 (Nxxxxx_Pxxxxxx)，timm 会将其识别为样本名
    - Label: 必须写入 int 字符串，否则 timm Reader 会报错
    """
    shard_path, image_data, class_to_idx = args
    shard_path = Path(shard_path)
    tmp_path = shard_path.with_suffix('.tar.tmp')
    
    # 简单的存在性检查
    if shard_path.exists() and shard_path.stat().st_size > 1024:
        return len(image_data)

    written_count = 0
    try:
        with tarfile.open(tmp_path, "w") as tar:
            for unique_id, img_path, class_name in image_data:
                try:
                    # 1. 准备 Label (适配 timm: 必须是 int)
                    if class_name not in class_to_idx:
                        print(f"[Warning] Class {class_name} not found in map!")
                        continue
                    label_idx = class_to_idx[class_name]
                    
                    # 2. 处理图片
                    with Image.open(img_path) as img:
                        img = img.convert('RGB')
                        w, h = img.size
                        ratio = TARGET_SIZE / min(w, h)
                        new_size = (int(w * ratio), int(h * ratio))
                        img = img.resize(new_size, Image.BILINEAR)
                        
                        img_byte_arr = io.BytesIO()
                        img.save(img_byte_arr, format='JPEG', quality=95)
                        img_bytes = img_byte_arr.getvalue()
                        
                        # ---------------- 关键修改 ----------------
                        # Key: 使用原始文件名 (去后缀)
                        # 例如: N05014_P000001
                        # 这样 timm 的 filenames() 方法返回的就是这个名字
                        base_name = Path(img_path).stem
                        key = base_name
                        # ----------------------------------------
                        
                        # 写入图片
                        info = tarfile.TarInfo(name=f"{key}.jpg")
                        info.size = len(img_bytes)
                        tar.addfile(info, io.BytesIO(img_bytes))
                        
                        # ---------------- 关键修改 ----------------
                        # Label: 写入数字索引的字符串
                        # 对应 reader_wds.py 中的 class_label = int(sample[target_key])
                        cls_str = str(label_idx)
                        cls_bytes = cls_str.encode('utf-8')
                        cls_info = tarfile.TarInfo(name=f"{key}.cls")
                        cls_info.size = len(cls_bytes)
                        tar.addfile(cls_info, io.BytesIO(cls_bytes))
                        # ----------------------------------------
                        
                        written_count += 1
                except Exception as e:
                    print(f"[Warning] Skipped {img_path}: {e}")
                    continue
        
        # 校验
        verify_success = True
        try:
            with tarfile.open(tmp_path, "r") as tar:
                for _ in tar: pass
        except:
            verify_success = False

        if verify_success:
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
    
    if output.exists():
        print(f"Checking output dir {output}...")
        # 建议清理旧数据，避免混淆
        # shutil.rmtree(output) 

    # 1. 建立类别映射 (Class Mapping)
    print(f"Scanning classes in {source}...")
    # 排序非常重要，保证 N01001 对应 0, N01002 对应 1...
    classes = sorted([d.name for d in source.iterdir() if d.is_dir()])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    print(f"Found {len(classes)} classes.")
    
    # 保存映射表：训练完后，你需要这个表把预测出的 数字 0 转回 N01001
    output.mkdir(parents=True, exist_ok=True)
    with open(output / "class_map.json", "w") as f:
        json.dump(class_to_idx, f, indent=4)
    print(f"Class map saved to {output}/class_map.json")

    # 2. 收集所有图片
    all_images_info = []
    class_dirs = [source / cls for cls in classes]
    
    print("Listing images...")
    for cls_dir in tqdm(class_dirs):
        cls_name = cls_dir.name
        for img_file in cls_dir.iterdir():
            if img_file.is_file() and not img_file.name.startswith('.'):
                all_images_info.append((img_file, cls_name))
    
    total_images = len(all_images_info)
    print(f"Total images: {total_images}. Shuffling...")
    
    # 3. 随机打乱
    random.seed(42)
    random.shuffle(all_images_info)
    
    # 4. 辅助索引 (用于切分Fold，不用于Key)
    indexed_images = []
    for i, (path, cls) in enumerate(all_images_info):
        indexed_images.append((i, path, cls))
        
    # 5. 分配任务
    fold_size = math.ceil(total_images / NUM_FOLDS)
    tasks = []
    
    for fold_idx in range(NUM_FOLDS):
        start_idx = fold_idx * fold_size
        end_idx = min((fold_idx + 1) * fold_size, total_images)
        
        fold_subset = indexed_images[start_idx:end_idx]
        
        fold_dir = output / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        
        num_shards = math.ceil(len(fold_subset) / SHARD_SIZE)
        
        for i in range(num_shards):
            s_start = i * SHARD_SIZE
            s_end = min((i + 1) * SHARD_SIZE, len(fold_subset))
            shard_data = fold_subset[s_start:s_end]
            
            shard_path = fold_dir / f"shard_{i:05d}.tar"
            
            tasks.append((shard_path, shard_data, class_to_idx))

    print(f"Starting generation of {len(tasks)} shards...")
    
    # 6. 执行
    fold_stats = {i: 0 for i in range(NUM_FOLDS)}
    with multiprocessing.Pool(NUM_WORKERS) as pool:
        pbar = tqdm(total=len(tasks))
        for i, count in enumerate(pool.imap_unordered(safe_write_shard, tasks)):
            pbar.update(1)
        pbar.close()

    # 重新统计
    print("\nVerifying final counts...")
    final_counts = []
    for i in range(NUM_FOLDS):
        start = i * fold_size
        end = min((i + 1) * fold_size, total_images)
        count = end - start
        final_counts.append(count)
        print(f"Fold {i}: ~{count} images")

    print("\n" + "="*50)
    print("DONE. COPY BELOW ARRAY:")
    print("="*50)
    counts_str = " ".join([str(c) for c in final_counts])
    print(f"FOLD_COUNTS=({counts_str})")

if __name__ == "__main__":
    main()