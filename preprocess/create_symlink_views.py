import os
from pathlib import Path
from tqdm import tqdm

# ================= 配置 =================
# 1. 这里填你上一步生成的包含真实 tar 包的路径
PHYSICAL_ROOT = Path("/home/jiangwentao/Repos/foundation-agriculture/datasets/AgriNet-1K/wds_folds")

# 2. 这里填你想创建软链接的目标路径
VIEW_ROOT = Path("/home/jiangwentao/Repos/foundation-agriculture/datasets/AgriNet-1K/wds_views")

NUM_FOLDS = 5
# =======================================

def create_links(src_files, dst_dir):
    """
    在 dst_dir 下创建指向 src_files 的软链接，
    并强制重命名为连续的 shard_00000.tar, shard_00001.tar ...
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    # 清理旧链接（防止重复运行导致混淆）
    for existing_link in dst_dir.glob("*.tar"):
        if existing_link.is_symlink():
            existing_link.unlink()

    for i, src_file in enumerate(src_files):
        # 必须使用绝对路径，否则软链接会失效
        src_abs = src_file.resolve()
        
        # 创建连续编号的新文件名
        link_name = dst_dir / f"shard_{i:05d}.tar"
        
        try:
            os.symlink(src_abs, link_name)
        except OSError as e:
            print(f"Error linking {src_abs} -> {link_name}: {e}")

def main():
    if not PHYSICAL_ROOT.exists():
        print(f"Error: Physical root {PHYSICAL_ROOT} does not exist.")
        return

    # 1. 扫描所有的物理文件
    # 结构假设为: PHYSICAL_ROOT / fold_X / shard_XXXXX.tar
    folds_data = {}
    for i in range(NUM_FOLDS):
        fold_path = PHYSICAL_ROOT / f"fold_{i}"
        # 获取该 fold 下所有 tar 包并排序
        tars = sorted(list(fold_path.glob("*.tar")))
        folds_data[i] = tars
        print(f"Physical Fold {i}: found {len(tars)} shards.")

    print("-" * 40)

    # 2. 创建 5 个 View
    for target_fold in range(NUM_FOLDS):
        print(f"Creating View for Fold {target_fold} (Target Fold is Validation)...")
        
        view_dir = VIEW_ROOT / f"fold_{target_fold}"
        
        # --- 准备验证集 (Validation) ---
        # 验证集就是当前的 fold_X
        val_files = folds_data[target_fold]
        create_links(val_files, view_dir / "val")
        print(f"  - Validation: Linked {len(val_files)} shards from fold_{target_fold}")
        
        # --- 准备训练集 (Train) ---
        # 训练集是除了 target_fold 以外的所有 folds
        train_files = []
        for i in range(NUM_FOLDS):
            if i != target_fold:
                train_files.extend(folds_data[i])
        
        create_links(train_files, view_dir / "train")
        print(f"  - Train:      Linked {len(train_files)} shards from folds {[x for x in range(NUM_FOLDS) if x != target_fold]}")

    print("-" * 40)
    print(f"Done! Views created at: {VIEW_ROOT}")
    print("Example structure:")
    print(f"  {VIEW_ROOT}/fold_0/train/shard_00000.tar -> Points to Physical Fold 1")
    print(f"  {VIEW_ROOT}/fold_0/val/shard_00000.tar   -> Points to Physical Fold 0")

if __name__ == "__main__":
    main()