from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import cv2

img_path = "dataset/Practice/train/ants_image/0013035.jpg"
img_cv = cv2.imread(img_path)
print(type(img_cv))
img = Image.open(img_path)
print(img)

writer = SummaryWriter("logs")

# 1. transforms的使用
tensor_trans = transforms.ToTensor()        # 实例化
img_tensor = tensor_trans(img)
# print(img_tensor)


writer.add_image("img_tensor", img_tensor)
writer.close()



