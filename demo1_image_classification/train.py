# image_classification/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

# 从项目中导入模块
from config import DEVICE, NUM_EPOCHS, LEARNING_RATE, MODEL_SAVE_PATH
from models.model import SimpleCNN
from utils.dataset import create_dataloaders

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    训练一个epoch
    """
    model.train() # 设置模型为训练模式
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    # 使用tqdm显示进度条
    progress_bar = tqdm(dataloader, desc="Training", unit="batch")

    for inputs, labels in progress_bar:
        # 将数据移动到指定设备
        inputs, labels = inputs.to(device), labels.to(device)

        # 1. 清零梯度. optimizer.zero_grad() 是为了清除上一步的梯度信息
        # 否则梯度会累加
        optimizer.zero_grad()

        # 2. 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 3. 反向传播. calculate gradients use backward()
        loss.backward()
# model.py
# Used 1 reference
# In PyTorch, calling loss.backward() computes the gradients of the loss with respect to all model parameters that have requires_grad=True.

# Why does this work?

# When you do the forward pass (outputs = model(inputs)), 
# PyTorch builds a computation graph that tracks all operations.
# The loss is a scalar value computed from the model's output 
# and the labels.
# Calling loss.backward() automatically computes the gradients 
# of the loss with respect to each parameter in the model b
# y traversing this computation graph using backpropagation.
# These gradients are then stored in the .grad attribute 
# of each parameter, ready for the optimizer to use in optimizer.step().
# Summary:
# loss.backward() triggers automatic differentiation, 
# so you get all necessary gradients for updating the model parameters.
     
        

        # 4. 更新权重
        optimizer.step()

        # 统计损失和准确率
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

        # 更新进度条信息
        progress_bar.set_postfix(loss=loss.item(), acc=f"{(predicted == labels).sum().item()/inputs.size(0):.4f}")

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc

def main():
    """
    主训练函数
    """
    # 打印设备信息
    print(f"Using device: {DEVICE}")

    # 1. 创建DataLoader
    train_loader, _ = create_dataloaders() # 在这里我们只用到了训练loader

    # 2. 初始化模型
    model = SimpleCNN().to(DEVICE)

    # 3. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 确保模型保存目录存在
    if not os.path.exists(MODEL_SAVE_PATH):
        os.makedirs(MODEL_SAVE_PATH)

    loss_list = []
    acc_list = []
    # 4. 开始训练循环
    for epoch in range(NUM_EPOCHS):
        print(f"--- Epoch {epoch+1}/{NUM_EPOCHS} ---")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)

        # record loss and accuracy in a list for plotting later

        loss_list.append(train_loss)
        acc_list.append(train_acc)  

        print(f"Epoch {epoch+1} Summary: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        # 每个epoch后保存模型
        torch.save(model.state_dict(), f"{MODEL_SAVE_PATH}/epoch_{epoch+1}.pth")
        print(f"Model saved to {MODEL_SAVE_PATH}/epoch_{epoch+1}.pth")

    # plotting the loss and accuracy trend
    epochs = list(range(1, NUM_EPOCHS + 1))
    plt.figure(figsize=(12, 5)) 

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_list, marker='o', label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.xticks(epochs)
    plt.legend()
    plt.grid(True)

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc_list, marker='o', color='orange', label='Train Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.xticks(epochs)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_SAVE_PATH, "training_curve.png"))  # Save plot
    plt.show()
    plt.close()

    # 训练结束
    print("--- Training Finished ---")


if __name__ == "__main__":
    main()
