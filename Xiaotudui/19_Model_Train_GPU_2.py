import time

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import CIFAR10Model

# 定义训练的设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 准备数据集
train_dataset = torchvision.datasets.CIFAR10("./dataset", train=True, transform=torchvision.transforms.ToTensor(),
                                             download=True)
test_dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                            download=True)
train_dataset_len = len(train_dataset)
test_dataset_len = len(test_dataset)
print("训练集样本数量：{}".format(train_dataset_len))
print("测试集样本数量：{}".format(test_dataset_len))

# 准备数据集加载器
train_dataloader = DataLoader(train_dataset, batch_size=64)
test_dataloader = DataLoader(test_dataset, batch_size=64)

# 搭建网络模型

model = CIFAR10Model.CIFAR10Model().to(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss().to(device)

# 优化器
# learning_rate = 0.01
# 1e-2 = 1 x 10^-2 = 0.01
learning_rate = 1e-2
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 设置训练网络的参数
# 记录训练次数
total_train_step = 0
# 记录测试次数
total_test_step = 0
# 训练轮数
epochs = 50

# 添加 TensorBoard
writer = SummaryWriter("logs")

start_time = time.time()
for epoch in range(epochs):
    print("--------第 {} 轮训练开始--------".format(epoch + 1))

    # 训练步骤开始
    model.train()
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)

        outputs = model(imgs)  # 前向传播
        loss = loss_fn(outputs, targets)  # 计算损失
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重（参数）

        total_train_step += 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print(end_time - start_time)
            print("训练次数：{}，Loss：{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    model.eval()
    total_test_loss = 0
    total_test_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)

            outputs = model(imgs)
            loss = loss_fn(outputs, targets)

            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_test_accuracy += accuracy

    print("整体测试集Loss：{}".format(total_test_loss))
    total_accuracy = total_test_accuracy / test_dataset_len * 100
    print("整体测试集正确率：{}%".format(total_accuracy))

    writer.add_scalar("test_loss", total_test_loss / 100, total_test_step)
    writer.add_scalar("test_accuracy", total_test_accuracy / test_dataset_len, total_test_step)
    total_test_step += 1

    torch.save(model, "model/CIFAR10Model_{}.pth".format(epoch + 1))
    print("模型已保存")

writer.close()
