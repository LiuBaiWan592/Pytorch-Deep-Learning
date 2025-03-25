import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

dataset_transform = transforms.Compose([
    transforms.ToTensor()
])

train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=dataset_transform, download=True)

# print(test_set[0])
# print(test_set.classes)
#
# img, target = test_set[0]
# print(img)
# print(target)
#
# print(test_set.classes[target])

writer = SummaryWriter("logs")
for i in range(10):
    img, target = test_set[i]
    writer.add_image("Test_set", img, i)


writer.close()

