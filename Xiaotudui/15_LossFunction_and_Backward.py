from torch import nn

import CIFAR10Model

loss = nn.CrossEntropyLoss()
model = CIFAR10Model.CIFAR10Model()
for data in CIFAR10Model.dataloader:
    imgs, target = data
    output = model(imgs)
    result_loss = loss(output, target)
    result_loss.backward()
    print("ok")
