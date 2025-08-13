import os
import random
import shutil
from tqdm import tqdm

def prepare_dataset(source_dir, target_dir, val_split=0.2):
    """
    自动将源数据划分为训练集和验证集，并整理到目标文件夹。

    参数:
    source_dir (str): 原始图片所在的文件夹 (例如 '.../train/')
    target_dir (str): 目标根目录 (例如 './data')
    val_split (float): 验证集所占的比例
    """
    # 1. 创建目标文件夹结构
    train_path = os.path.join(target_dir, 'train')
    val_path = os.path.join(target_dir, 'val')

    train_cat_path = os.path.join(train_path, 'cat')
    train_dog_path = os.path.join(train_path, 'dog')
    val_cat_path = os.path.join(val_path, 'cat')
    val_dog_path = os.path.join(val_path, 'dog')

    # 使用 os.makedirs 一次性创建所有需要的目录
    for path in [train_cat_path, train_dog_path, val_cat_path, val_dog_path]:
        os.makedirs(path, exist_ok=True)
    
    # 2. 获取所有图片并分类
    all_files = os.listdir(source_dir)
    cat_files = [f for f in all_files if f.startswith('cat')]
    dog_files = [f for f in all_files if f.startswith('dog')]
    
    # 打乱文件列表
    random.shuffle(cat_files)
    random.shuffle(dog_files)
    
    print(f"找到 {len(cat_files)} 张猫的图片和 {len(dog_files)} 张狗的图片。")

    # 3. 分割并复制文件
    def split_and_copy(file_list, train_dest, val_dest):
        split_point = int(len(file_list) * (1 - val_split))
        train_files = file_list[:split_point]
        val_files = file_list[split_point:]

        print(f"正在复制 {len(train_files)} 个文件到 {train_dest}...")
        for file_name in tqdm(train_files, desc=f"Train {os.path.basename(train_dest)}"):
            shutil.copy(os.path.join(source_dir, file_name), os.path.join(train_dest, file_name))
            
        print(f"正在复制 {len(val_files)} 个文件到 {val_dest}...")
        for file_name in tqdm(val_files, desc=f"Val {os.path.basename(val_dest)}"):
            shutil.copy(os.path.join(source_dir, file_name), os.path.join(val_dest, file_name))

    print("\n--- 正在处理猫的图片 ---")
    split_and_copy(cat_files, train_cat_path, val_cat_path)

    print("\n--- 正在处理狗的图片 ---")
    split_and_copy(dog_files, train_dog_path, val_dog_path)

    print("\n数据集准备完成！")
    print("="*30)
    print(f"训练集猫数量: {len(os.listdir(train_cat_path))}")
    print(f"训练集狗数量: {len(os.listdir(train_dog_path))}")
    print(f"验证集猫数量: {len(os.listdir(val_cat_path))}")
    print(f"验证集狗数量: {len(os.listdir(val_dog_path))}")
    print("="*30)


if __name__ == '__main__':
    # !!! 重要：修改为你自己的路径 !!!
    # 这是你从Kaggle解压后的 `train` 文件夹的路径
    SOURCE_IMAGE_DIR = "/path/to/your/downloaded/dogs-vs-cats/train" 
    
    # 这是我们项目中的 `data` 目录
    TARGET_DATA_DIR = "./data"
    
    if not os.path.isdir(SOURCE_IMAGE_DIR) or len(os.listdir(SOURCE_IMAGE_DIR)) < 10:
         print(f"错误：请确保 '{SOURCE_IMAGE_DIR}' 是一个有效的目录，并且包含了Kaggle猫狗大战的图片。")
         print("你需要先从Kaggle下载并解压数据集，然后更新脚本中的 SOURCE_IMAGE_DIR 变量。")
    else:
        prepare_dataset(SOURCE_IMAGE_DIR, TARGET_DATA_DIR)
