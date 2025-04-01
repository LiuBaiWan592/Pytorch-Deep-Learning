import torch
import torchvision
from PIL import Image

image_path = "./images/truck.png"
image = Image.open(image_path)
print(image)
image = image.convert('RGB')
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])

image = transform(image)
print(image.shape)

model = torch.load("model/CIFAR10Model_44.pth")
print(model)

image = torch.reshape(image, (1, 3, 32, 32))
model.eval()

with torch.no_grad():
    output = model(image.cuda())
print(output)

print(output.argmax(1))
