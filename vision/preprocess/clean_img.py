import os
import argparse
from pathlib import Path
from PIL import Image, ImageFile
from tqdm import tqdm
import multiprocessing

# ================= 配置区域 =================
# 原始数据集根目录
DATA_ROOT = "datasets/AgriNet-1K/all"
# 删除日志保存位置
LOG_FILE = "deleted_images_log.txt"
# 并行进程数 (HDD 建议 8-16)
NUM_WORKERS = 128
# ===========================================

# 确保 PIL 不会加载截断的图片，而是抛出错误
ImageFile.LOAD_TRUNCATED_IMAGES = False

def check_and_remove(file_path):
    """
    尝试完全加载图片。如果失败，删除文件。
    返回: (status, message/path)
    status: 'ok', 'corrupt', 'error'
    """
    try:
        file_path = Path(file_path)
        
        # 1. 尝试打开并强制加载数据
        # 很多损坏只有在 load() 时才会暴露，比如 broken data stream
        with Image.open(file_path) as img:
            img.load() 
            # 也可以转换一下格式确保解码器工作
            # img.convert('RGB') 
            
        return ('ok', None)

    except (OSError, SyntaxError, UserWarning) as e:
        # 捕获 PIL 常见的图片损坏错误
        try:
            # 物理删除
            if file_path.exists():
                os.remove(file_path)
            return ('corrupt', str(file_path))
        except Exception as del_e:
            return ('error', f"Found corrupt {file_path} but failed to delete: {del_e}")
            
    except Exception as e:
        # 其他未知错误
        return ('error', f"Unknown error checking {file_path}: {e}")

def main():
    root_dir = Path(DATA_ROOT)
    
    if not root_dir.exists():
        print(f"Error: Path {root_dir} does not exist.")
        return

    print(f"Scanning directory: {root_dir} ...")
    
    # 扫描所有常见的图片扩展名
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.JPG', '.JPEG', '.PNG'}
    image_files = []
    
    # 递归查找所有图片
    # 使用 rglob 可能会比较慢，如果你知道只有两层目录，可以用嵌套循环优化
    # 这里为了通用使用 rglob
    for ext in extensions:
        image_files.extend(list(root_dir.rglob(f"*{ext}")))
        
    print(f"Found {len(image_files)} images. Starting integrity check...")
    print(f"Using {NUM_WORKERS} workers. This will involve full file reading, please wait...")

    deleted_count = 0
    errors_count = 0
    
    # 准备写入日志
    with open(LOG_FILE, "w") as log:
        log.write(f"Corrupt images deletion log - {root_dir}\n")
        log.write("="*50 + "\n")
        
        with multiprocessing.Pool(NUM_WORKERS) as pool:
            # 使用 tqdm 显示进度
            for status, msg in tqdm(pool.imap_unordered(check_and_remove, image_files), total=len(image_files)):
                
                if status == 'corrupt':
                    deleted_count += 1
                    # 打印到控制台
                    # print(f"\n[Deleted] {msg}") 
                    # 写入日志
                    log.write(f"{msg}\n")
                    log.flush() # 确保实时写入
                    
                elif status == 'error':
                    errors_count += 1
                    print(f"\n[System Error] {msg}")

    print("\n" + "="*50)
    print("CLEANING COMPLETE")
    print("="*50)
    print(f"Total scanned: {len(image_files)}")
    print(f"Corrupt & Deleted: {deleted_count}")
    print(f"System/Permission Errors: {errors_count}")
    print(f"Log saved to: {os.path.abspath(LOG_FILE)}")
    
    if deleted_count > 0:
        print("\n*** IMPORTANT ***")
        print("Since files have been deleted, you MUST re-run the 'generate_wds_final.py' script")
        print("to regenerate the .tar shards without the corrupt files.")

if __name__ == "__main__":
    main()