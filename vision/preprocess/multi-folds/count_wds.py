import tarfile
import multiprocessing
from pathlib import Path
from tqdm import tqdm

# ================= 配置 =================
# 指向存放真实 tar 包的物理路径 (HDD)
DATA_ROOT = Path("datasets/AgriNet-1K/wds_split")
NUM_WORKERS = 64  # 并行读取进程数
# =======================================

def count_tar_images(tar_path):
    """统计单个 tar 包内的 jpg 图片数量"""
    count = 0
    try:
        # 使用 'r' 模式读取，仅读取 header
        with tarfile.open(tar_path, "r") as tar:
            for member in tar:
                # 假设图片后缀是 .jpg (根据之前的生成脚本)
                # WebDataset 中通常包含 .jpg 和 .cls，我们只数 .jpg
                if member.name.endswith('.jpg') or member.name.endswith('.jpeg') or member.name.endswith('.png'):
                    count += 1
    except Exception as e:
        print(f"Error reading {tar_path}: {e}")
        return 0
    return count

def main():
    if not DATA_ROOT.exists():
        print(f"Error: Path {DATA_ROOT} not found.")
        return

    # 1. 获取所有的 Fold 和其下的 tar 文件
    folds = sorted([d for d in DATA_ROOT.iterdir() if d.is_dir()])
    
    fold_counts = {}
    total_images = 0

    print(f"Found {len(folds)} folds. Starting scan using {NUM_WORKERS} workers...")
    print("This may take a few minutes on HDD...")

    for fold_dir in folds:
        fold_name = fold_dir.name
        tar_files = sorted(list(fold_dir.glob("*.tar")))
        
        if not tar_files:
            print(f"Warning: {fold_name} is empty.")
            fold_counts[fold_name] = 0
            continue

        # 多进程统计当前 Fold
        with multiprocessing.Pool(NUM_WORKERS) as pool:
            # 使用 imap 获取结果
            results = list(tqdm(pool.imap(count_tar_images, tar_files), 
                              total=len(tar_files), 
                              desc=f"Scanning {fold_name}",
                              leave=False))
        
        count = sum(results)
        fold_counts[fold_name] = count
        total_images += count
        
        print(f"  -> {fold_name}: {count} images ({len(tar_files)} shards)")

    print("=" * 50)
    print("SUMMARY FOR CONFIG")
    print("=" * 50)
    print(f"Total Dataset Size: {total_images}")
    print("-" * 50)
    
    # 2. 打印可以直接复制到脚本里的参数建议
    print("Use these numbers for your --train-num-samples and --val-num-samples:\n")
    
    for fold_dir in folds:
        fold_name = fold_dir.name
        val_count = fold_counts[fold_name]
        train_count = total_images - val_count
        
        print(f"Running {fold_name} as VALIDATION:")
        print(f"   --train-num-samples {train_count}")
        print(f"   --val-num-samples   {val_count}")
        print("-" * 30)

if __name__ == "__main__":
    main()