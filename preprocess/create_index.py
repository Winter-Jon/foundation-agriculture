import os
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# ================= 配置 =================
CLEAN_DATASET_ROOT = "datasets/AgriNet-1K/all"  # 修改为机器A上的路径
INDEX_OUTPUT_FILE = "index.json"           #生成的指纹文件
NUM_THREADS = 16
# =======================================

def calculate_md5(file_path):
    """计算文件MD5"""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception:
        return None

def main():
    # 1. 扫描所有文件路径
    all_files = []
    print(f"正在扫描目录结构: {CLEAN_DATASET_ROOT}")
    for root, dirs, files in os.walk(CLEAN_DATASET_ROOT):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                all_files.append(os.path.join(root, file))

    print(f"共发现 {len(all_files)} 个文件，开始计算哈希...")

    # 2. 并行计算哈希并建立索引
    hash_index = {} # {md5: relative_path}
    
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        # 提交任务
        futures = {executor.submit(calculate_md5, fp): fp for fp in all_files}
        
        for future in tqdm(futures, total=len(all_files), desc="Generating Index"):
            fpath = futures[future]
            md5_val = future.result()
            
            if md5_val:
                # 关键：只保存相对路径（例如 "n01440764/processed_img_01.jpg"）
                rel_path = os.path.relpath(fpath, CLEAN_DATASET_ROOT)
                hash_index[md5_val] = rel_path

    # 3. 保存指纹文件
    print(f"索引构建完成，包含 {len(hash_index)} 个唯一哈希值。")
    with open(INDEX_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(hash_index, f)
    
    print(f"文件已保存至: {INDEX_OUTPUT_FILE}")
    print(">>> 请将此 JSON 文件发送到机器 B <<<")

if __name__ == "__main__":
    main()