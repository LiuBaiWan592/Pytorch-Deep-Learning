from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs")
img = Image.open("images/Bronco.webp")
print(img)

# ToTensor
trans_toTensor = transforms.ToTensor()
img_tensor = trans_toTensor(img)
writer.add_image("ToTensor", img_tensor)

# Normalize
print(img_tensor[0][0][0])
# output[channel] = (input[channel] - mean[channel]) / std[channel]
trans_normalize = transforms.Normalize([0.3, 0.2, 0.1], [1, 2, 3])
img_norm = trans_normalize(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Normalize", img_norm, 2)

# Resize
print(type(img))
print(img.size)
trans_resize = transforms.Resize((512, 512))
# img(PIL) -> Resize -> img_resize(PIL)
img_resize = trans_resize(img)
# img_resize(PIL) -> ToTensor -> img_resize_tensor(Tensor)
img_resize_tensor = trans_toTensor(img_resize)
writer.add_image("Resize", img_resize_tensor, 0)
print(type(img_resize))
print(img_resize.size)

# Compose and Resize_2
trans_resize_2 = transforms.Resize(512)
# img(PIL) -> Resize -> img_resize(PIL)
# img_resize(PIL) -> ToTensor -> img_resize_tensor(Tensor)
trans_compose = transforms.Compose([trans_resize_2, trans_toTensor])
img_resize_2 = trans_compose(img)
writer.add_image("Resize", img_resize_2, 1)
print(img_resize_2.dtype)
print(img_resize_2.shape)

# RandomCrop
trans_randomCrop = transforms.RandomCrop([512, 1024])
trans_compose_2 = transforms.Compose([trans_randomCrop, trans_toTensor])
for i in range(10):
    img_randomCrop = trans_compose_2(img)
    writer.add_image("RandomCrop", img_randomCrop, i)


writer.close()


