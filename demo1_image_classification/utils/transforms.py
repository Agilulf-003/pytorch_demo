# image_classification/utils/transforms.py

from torchvision import transforms
from config import IMAGE_SIZE

def get_train_transforms():
    """
    获取训练集的数据增强/转换流程
    """
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(), # 随机水平翻转
        transforms.RandomRotation(10),     # 随机旋转
        transforms.ToTensor(),             # 转换为Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 标准化
    ])

def get_val_transforms():
    """
    获取验证/测试集的数据转换流程
    """
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
