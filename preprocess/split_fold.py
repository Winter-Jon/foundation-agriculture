import os
import random
import argparse
from pathlib import Path
from tqdm import tqdm
import math

def create_folds(source_dir, output_dir, k_folds=5, seed=42):
    """
    将ImageNet格式的数据集划分为K折，使用软连接。
    """
    source_path = Path(source_dir).resolve() #以此获取绝对路径
    output_path = Path(output_dir).resolve()

    if not source_path.exists():
        raise FileNotFoundError(f"Source directory not found: {source_path}")

    # 设置随机种子以保证复现性
    random.seed(seed)

    # 获取所有类别文件夹 (假设source下全是类别文件夹)
    classes = sorted([d for d in source_path.iterdir() if d.is_dir()])
    
    print(f"Found {len(classes)} classes in {source_path}")
    print(f"Preparing to split into {k_folds} folds using symbolic links...")

    # 遍历每个类别进行处理
    for class_dir in tqdm(classes, desc="Processing Classes"):
        class_name = class_dir.name
        
        # 获取该类别下的所有图片文件 (过滤隐藏文件)
        images = sorted([f for f in class_dir.iterdir() if f.is_file() and not f.name.startswith('.')])
        
        # 打乱顺序
        random.shuffle(images)
        
        num_images = len(images)
        fold_size = math.ceil(num_images / k_folds)

        # 为每一折创建数据
        for fold_idx in range(k_folds):
            # 计算当前Fold作为Validation集的索引范围
            start_idx = fold_idx * fold_size
            end_idx = min((fold_idx + 1) * fold_size, num_images)
            
            # 切分数据
            val_images = images[start_idx:end_idx]
            # 剩下的就是训练集 (取差集)
            # 注意：这里使用列表推导式可能在数据量极大时稍慢，但在ImageNet Class级别(几千张)是完全没问题的
            train_images = [img for i, img in enumerate(images) if i < start_idx or i >= end_idx]

            # 定义目标路径结构: output/fold_X/train/class_name 和 output/fold_X/val/class_name
            fold_root = output_path / f"fold_{fold_idx}"
            train_dest = fold_root / "train" / class_name
            val_dest = fold_root / "val" / class_name

            # 创建目录
            train_dest.mkdir(parents=True, exist_ok=True)
            val_dest.mkdir(parents=True, exist_ok=True)

            # 创建软连接的辅助函数
            def make_symlinks(file_list, destination_dir):
                for src_file in file_list:
                    link_name = destination_dir / src_file.name
                    # 如果连接已存在，跳过（防止重复运行报错）
                    if not link_name.exists():
                        try:
                            os.symlink(src_file, link_name)
                        except OSError as e:
                            print(f"Error linking {src_file} to {link_name}: {e}")

            # 执行连接创建
            make_symlinks(train_images, train_dest)
            make_symlinks(val_images, val_dest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split ImageNet dataset into K-folds using symlinks.")
    parser.add_argument("--source", type=str, required=True, help="Path to the 'all' directory containing class folders.")
    parser.add_argument("--output", type=str, required=True, help="Path where the folds will be created.")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds (default: 5).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()

    create_folds(args.source, args.output, args.folds, args.seed)