import torchvision

from torch import nn

# train_dataset = torchvision.datasets.ImageNet(root='./dataset', split='train', download=True,
#                                               transform=torchvision.transforms.ToTensor())

vgg16_preFalse = torchvision.models.vgg16(weights=None)
vgg16_preTrue = torchvision.models.vgg16(weights='IMAGENET1K_V1')

train_dataset = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True,
                                             transform=torchvision.transforms.ToTensor())

# 增加一层进行修改
vgg16_preTrue.classifier.add_module('add_linear', nn.Linear(1000, 10))
print(vgg16_preTrue)

# 直接修改原模型
vgg16_preFalse.classifier[6] = nn.Linear(4096, 10)
print(vgg16_preFalse)
