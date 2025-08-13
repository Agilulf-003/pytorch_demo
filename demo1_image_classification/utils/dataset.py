# image_classification/utils/dataset.py

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from utils.transforms import get_train_transforms, get_val_transforms
from config import TRAIN_DIR, VAL_DIR, BATCH_SIZE

def create_dataloaders():
    """
    创建训练和验证的DataLoader
    """
    # 获取数据转换
    train_transforms = get_train_transforms()
    val_transforms = get_val_transforms()

    # 创建Dataset
    # ImageFolder会自动根据文件夹名称分配标签
    train_dataset = ImageFolder(root=TRAIN_DIR, transform=train_transforms)
    val_dataset = ImageFolder(root=VAL_DIR, transform=val_transforms)

    # 创建DataLoader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True, # 训练集需要打乱
        num_workers=4, # 使用多进程加载数据
        pin_memory=True # 锁页内存，加快数据到GPU的传输
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False, # 验证集不需要打乱
        num_workers=4,
        pin_memory=True
    )

    print(f"类别映射: {train_dataset.class_to_idx}")
    return train_loader, val_loader

if __name__ == '__main__':
    # 测试一下DataLoader是否正常
    train_loader, val_loader = create_dataloaders()
    # 取一个batch的数据看看
    images, labels = next(iter(train_loader))
    print(f"Images batch shape: {images.size()}")
    print(f"Labels batch shape: {labels.size()}")
