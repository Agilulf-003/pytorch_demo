# image_classification/config.py

import torch

# ----- 基本配置 -----
PROJECT_NAME = "SimpleImageClassification"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----- 数据路径 -----
DATA_ROOT = "./data"
TRAIN_DIR = f"{DATA_ROOT}/train"
VAL_DIR = f"{DATA_ROOT}/val"

# ----- 模型相关 -----
# 模型保存路径
MODEL_SAVE_PATH = "./checkpoints"
# 预训练模型 (如果使用的话)
PRETRAINED_MODEL = None # or "resnet18", etc.

# ----- 训练超参数 -----
NUM_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001
IMAGE_SIZE = (128, 128) # 图片统一调整到的大小

# ----- 类别信息 -----
# 假设你的文件夹是 'cat' 和 'dog'
CLASSES = ['cat', 'dog']
NUM_CLASSES = len(CLASSES)
