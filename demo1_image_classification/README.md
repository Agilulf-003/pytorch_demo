# 最简单的 PyTorch 图像分类项目 Demo

这是一个基础的、结构化的 PyTorch 图像分类项目，用于演示一个完整的工作流程。

## 目录结构

```
image_classification/
├── data/                  # 数据
│   ├── train/             # 训练集 (每个类别一个文件夹)
│   │   ├── cat/
│   │   └── dog/
│   └── val/               # 验证集
│       ├── cat/
│       └── dog/
├── models/                # 模型定义
│   └── model.py
├── utils/                 # 工具函数
│   ├── dataset.py
│   └── transforms.py
├── checkpoints/           # 训练好的模型权重 (会自动创建)
├── config.py              # 全局配置文件
├── train.py               # 训练脚本
├── eval.py                # 评估脚本
└── README.md              # 本说明文件
```

## 环境依赖

你需要安装 PyTorch 和其他一些常用库。

```bash
pip install torch torchvision tqdm
```

## 如何使用

### 1. 准备数据

将你的训练和验证图片按照 `data/` 目录下的示例结构放好。

### 2. 配置参数

打开 `config.py` 文件，根据你的需求修改超参数，例如学习率 (`LEARNING_RATE`)、批次大小 (`BATCH_SIZE`)、训练轮数 (`NUM_EPOCHS`) 以及类别信息 (`CLASSES`)。

### 3. 开始训练

运行训练脚本来开始训练模型。训练好的模型权重会保存在 `checkpoints/` 目录下。

```bash
python train.py
```

### 4. 评估模型

训练完成后，运行评估脚本来测试模型在验证集上的性能。

```bash
python eval.py
```
**注意**: `eval.py` 默认加载最后一个 epoch 的模型权重，请确保 `config.py` 中的 `NUM_EPOCHS` 与 `train.py` 训练的轮数一致，或者手动修改 `eval.py` 中加载的模型路径。
