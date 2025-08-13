# image_classification/eval.py

import torch
from tqdm import tqdm
import torch.nn as nn

# 从项目中导入模块
from config import DEVICE, MODEL_SAVE_PATH
from models.model import SimpleCNN
from utils.dataset import create_dataloaders

def evaluate_model(model, dataloader, criterion, device):
    """
    在给定数据集上评估模型
    """
    model.eval() # 设置模型为评估模式
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    # 不需要计算梯度
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating", unit="batch")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            progress_bar.set_postfix(loss=loss.item(), acc=f"{(predicted == labels).sum().item()/inputs.size(0):.4f}")


    eval_loss = running_loss / total_samples
    eval_acc = correct_predictions / total_samples
    return eval_loss, eval_acc

def main():
    """
    主评估函数
    """
    # 1. 创建DataLoader
    _, val_loader = create_dataloaders() # 这里我们只需要验证loader

    # 2. 加载模型
    model = SimpleCNN().to(DEVICE)
    # 加载训练好的权重，这里以最后一个epoch为例
    model_path = f"{MODEL_SAVE_PATH}/epoch_{10}.pth" # 假设我们训练了10个epoch
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        print("Please run train.py first to generate the model file.")
        return
        
    print(f"Model loaded from {model_path}")

    # 3. 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 4. 执行评估
    val_loss, val_acc = evaluate_model(model, val_loader, criterion, DEVICE)

    print("\n--- Evaluation Summary ---")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print("--------------------------")

if __name__ == "__main__":
    main()
