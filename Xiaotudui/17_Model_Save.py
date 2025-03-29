import torch
import torchvision
from torch import nn

vgg16 = torchvision.models.vgg16(weights=None)

# 模型保存方式1：保存网络模型的结构和参数
torch.save(vgg16, 'vgg16_method1.pth')

# 模型保存方式2：保存网络模型的参数为字典（官方推荐）
torch.save(vgg16.state_dict(), 'vgg16_method2.pth')


# 陷阱1
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        return x


my_model = MyModel()
torch.save(my_model.state_dict(), 'my_model_method1.pth')
