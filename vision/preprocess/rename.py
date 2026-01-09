import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# ================= 配置区域 =================
# 数据集根目录
DATA_ROOT = "datasets/AgriNet-1K/all"
# 起始索引 (通常从 1 开始: P00001)
START_INDEX = 1
# ===========================================

def main():
    root = Path(DATA_ROOT)
    if not root.exists():
        print(f"Error: {root} does not exist.")
        return

    print(f"Scanning classes in {root}...")
    
    # 获取所有类别文件夹
    classes = sorted([d for d in root.iterdir() if d.is_dir()])
    
    print(f"Found {len(classes)} classes. Starting file renaming...")
    print("Format: {FolderName}_P{Index:05d}.jpg")
    
    # 进度条
    pbar = tqdm(classes, desc="Processing Classes")
    
    for class_dir in pbar:
        class_name = class_dir.name # e.g., N05014
        
        pbar.set_description(f"Processing {class_name}")
        
        # 1. 获取文件夹内所有文件并排序
        images = sorted([
            f for f in class_dir.iterdir() 
            if f.is_file() and not f.name.startswith('.')
        ])
        
        # 2. 遍历重命名
        for i, img_path in enumerate(images, start=START_INDEX):
            # 构造新文件名: N05014_P000001.jpg
            new_filename = f"{class_name}_P{i:05d}.jpg"
            new_file_path = class_dir / new_filename
            
            # 如果文件名已经符合格式（例如之前运行过中断了），跳过
            if img_path.name == new_filename:
                continue
                
            try:
                # 检查是否需要转换格式 (非 jpg 转 jpg)
                suffix = img_path.suffix.lower()
                if suffix not in ['.jpg', '.jpeg']:
                    # 转换格式并保存
                    with Image.open(img_path) as img:
                        img = img.convert('RGB')
                        img.save(new_file_path, quality=95)
                    # 删除旧格式文件
                    os.remove(img_path)
                else:
                    # 如果已经是 jpg
                    # 为了防止 Linux 文件系统大小写不敏感导致的问题，
                    # 或者重命名冲突（A.jpg -> B.jpg，但B.jpg原本存在），
                    # 直接重命名。如果目标已存在，说明有逻辑错误或重复文件，这里做个简单检查
                    
                    if new_file_path.exists():
                        # 这种情况极少见，除非文件夹里原本就有 P000001 这种名字
                        # 为了安全，先重命名为一个临时文件，再改回来（但在有序处理中通常不需要）
                        print(f"[Warning] Target {new_filename} exists, skipping {img_path.name}")
                        continue
                        
                    os.rename(img_path, new_file_path)
                        
            except Exception as e:
                print(f"\n[Error] Failed to process {img_path}: {e}")

    print("="*50)
    print("RENAMING COMPLETE")
    print("="*50)
    print(f"Dataset Root: {DATA_ROOT}")

if __name__ == "__main__":
    main()