import os
import json
import io
import math
import multiprocessing
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import webdataset as wds

# ================= 配置区域 =================
INPUT_JSON_PATH = "./split_info.json"
OUTPUT_WDS_DIR = "/home/jiangwentao/Repos/foundation-agriculture/datasets/AgriNet-1K/wds_split"
TARGET_SIZE = 256
SHARD_SIZE = 2000
NUM_WORKERS = 16 
# ===========================================

def safe_write_shard(args):
    """多进程写入 Worker"""
    shard_path, items, class_to_idx = args
    shard_path = Path(shard_path)
    tmp_path = shard_path.with_suffix('.tar.tmp')
    
    if shard_path.exists() and shard_path.stat().st_size > 1024:
        return len(items) # Skip existing

    written = 0
    try:
        with wds.TarWriter(str(tmp_path)) as sink:
            for item in items:
                try:
                    path = item['path']
                    cls = item['cls']
                    key = item['stem']
                    
                    if cls not in class_to_idx: continue

                    # 图片处理
                    with Image.open(path) as img:
                        img = img.convert('RGB')
                        w, h = img.size
                        ratio = TARGET_SIZE / min(w, h)
                        new_size = (int(w * ratio), int(h * ratio))
                        img = img.resize(new_size, Image.BILINEAR)
                        
                        buf = io.BytesIO()
                        img.save(buf, format='JPEG', quality=95)
                        img_bytes = buf.getvalue()

                    sample = {
                        "__key__": key,
                        "jpg": img_bytes,
                        "cls": cls
                    }
                    sink.write(sample)
                    written += 1
                except Exception as e:
                    print(f"Error processing {item.get('path')}: {e}")
                    continue
        
        if written > 0:
            os.rename(tmp_path, shard_path)
            return written
        else:
            if tmp_path.exists(): os.remove(tmp_path)
            return 0
    except Exception as e:
        print(f"Shard failed {shard_path}: {e}")
        if tmp_path.exists(): os.remove(tmp_path)
        return 0

def main():
    # 1. 读取 JSON
    print(f"Loading split info from {INPUT_JSON_PATH}...")
    with open(INPUT_JSON_PATH, 'r') as f:
        data = json.load(f)
        
    class_map = data['class_map']
    splits = data['splits']
    
    output_root = Path(OUTPUT_WDS_DIR)
    output_root.mkdir(parents=True, exist_ok=True)
    
    # 保存 Class Map 到输出目录方便后续使用
    with open(output_root / "class_map.json", "w") as f:
        json.dump(class_map, f, indent=4)

    # 2. 准备任务
    tasks = []
    
    for split_name, items in splits.items():
        if not items: continue
        
        print(f"Preparing tasks for {split_name.upper()} ({len(items)} images)...")
        split_dir = output_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        # 在 Step 1 中已经决定了归属，这里不需要再 Shuffle，
        # 但为了 Shard 内部数据的随机性，建议再 Shuffle 一次
        import random
        random.shuffle(items)
        
        num_shards = math.ceil(len(items) / SHARD_SIZE)
        for i in range(num_shards):
            chunk = items[i*SHARD_SIZE : (i+1)*SHARD_SIZE]
            shard_name = f"{split_name}-{i:05d}.tar"
            tasks.append((split_dir / shard_name, chunk, class_map))
            
    print(f"Total shards to write: {len(tasks)}")
    
    # 3. 多进程执行
    with multiprocessing.Pool(NUM_WORKERS) as pool:
        list(tqdm(pool.imap_unordered(safe_write_shard, tasks), total=len(tasks)))
        
    print("\nStep 2 Done. WebDataset generated.")

if __name__ == "__main__":
    main()