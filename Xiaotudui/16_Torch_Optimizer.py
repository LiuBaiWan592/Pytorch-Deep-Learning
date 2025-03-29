import torch
from torch import nn

import CIFAR10Model

model = CIFAR10Model.CIFAR10Model()
loss_fn = nn.CrossEntropyLoss()  # 交叉熵损失
optim = torch.optim.SGD(params=model.parameters(), lr=0.01, )  # 随机梯度下降

for epoch in range(20):
    epoch_loss = 0
    for imgs, target in CIFAR10Model.dataloader:
        output = model(imgs)  # 前向传播
        loss = loss_fn(output, target)  # 计算损失
        optim.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        optim.step()  # 更新参数
        epoch_loss += loss
    print(f'Epoch {epoch}, Loss {epoch_loss}')
