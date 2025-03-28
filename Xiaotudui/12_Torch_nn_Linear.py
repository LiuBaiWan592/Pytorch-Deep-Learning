import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64, drop_last=True)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features=196608, out_features=10)

    def forward(self, input):
        output = self.linear(input)
        return output


model = Model()
writer = SummaryWriter("logs")
step = 0
for data in dataloader:
    imgs, targets = data
    print(imgs.shape)
    # input = torch.reshape(imgs, (1, 1, 1, -1))  # 展开图像为一维
    input = torch.flatten(imgs)
    print(input.shape)
    output = model(input)
    print(output.shape)
