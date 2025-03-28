import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.max_pool3 = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=64 * 4 * 4, out_features=64)
        self.linear2 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        x = self.conv3(x)
        x = self.max_pool3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        logits = self.linear2(x)
        return logits


class ModelSeq(nn.Module):
    def __init__(self):
        super(ModelSeq, self).__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(in_features=64 * 4 * 4, out_features=64),
            nn.Linear(in_features=64, out_features=10)
        )

    def forward(self, x):
        logits = self.model1(x)
        return logits


model = Model()
print(model)
input = torch.ones((64, 3, 32, 32))
output = model(input)
print(output.shape)

model_seq = ModelSeq()
print(model_seq)
output_seq = model_seq(input)
print(output_seq.shape)

writer = SummaryWriter("logs")
writer.add_graph(model_seq, input)
writer.close()
