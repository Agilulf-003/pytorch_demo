image_classification/
├── data/                  # 数据相关
│   ├── train/             # 训练集（按类别分文件夹，如cat/dog）
│   ├── val/               # 验证集
│   └── test/              # 测试集（可选）
│
├── models/                # 模型定义
│   └── model.py           # 自定义模型（如SimpleCNN）
│
├── utils/                 # 工具函数
│   ├── dataset.py         # 数据加载和预处理
│   └── transforms.py      # 数据增强（可选）
│
├── config.py              # 配置文件（超参数、路径等）
├── train.py               # 训练脚本
├── eval.py                # 评估脚本（可选）
└── README.md              # 项目说明
