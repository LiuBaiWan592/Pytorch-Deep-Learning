import torch
import torchvision
from torch import nn


# 模型加载方式 1
# model1 = torch.load('model/vgg16_method1.pth')
# print(model1)

# 模型加载方式 2
# vgg16 = torchvision.models.vgg16(weights=None)
# vgg16.load_state_dict(torch.load('model/vgg16_method2.pth'))
# print(vgg16)
# model2 = torch.load('model/vgg16_method2.pth')
# print(model2)

# 陷阱1
# 旧版需要将网络结构添加到文件中

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        return x


# my_model = torch.load('model/my_model_method1.pth')
# print(my_model)

# 可将模型实例化后添加参数
my_model2 = MyModel()
my_model2.load_state_dict(torch.load('model/my_model_method1.pth'))
print(my_model2)
