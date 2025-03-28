import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)


# input = torch.tensor([[1, 2, 0, 3, 1],
#                       [0, 1, 2, 3, 1],
#                       [1, 2, 1, 0, 0],
#                       [5, 2, 3, 1, 1],
#                       [2, 1, 0, 1, 1]])
#
# input = torch.reshape(input, (-1, 1, 5, 5))
# print(input.shape)
# print(input.dtype)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.max_pool = MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, input):
        output = self.max_pool(input)
        return output


model = Model()
# output = model(input)
# print(output)

writer = SummaryWriter("logs")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    output = model(imgs)
    writer.add_images("output", output, step)
    step = step + 1

writer.close()
