import torchvision
from torch import nn
from torch.utils.data import DataLoader

import CIFAR10Model

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)

loss = nn.CrossEntropyLoss()
model = CIFAR10Model.CIFAR10Model()
for data in dataloader:
    imgs, target = data
    output = model(imgs)
    result_loss = loss(output, target)
    result_loss.backward()
    print("ok")
