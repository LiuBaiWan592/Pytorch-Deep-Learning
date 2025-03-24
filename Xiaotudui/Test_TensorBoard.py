from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np


# tensorboard --logdir=logs --port=9999
writer = SummaryWriter("logs")

image_path = "dataset/Practice/train/bees_image/29494643_e3410f0d37.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)

print(img_array.shape)

writer.add_image("test", img_array, 2, dataformats='HWC')

# # y = x
# for i in range(100):
#     writer.add_scalar("y = x", i, i)

# y = 2x
for i in range(100):
    writer.add_scalar("y = 2x", 2*i, i)


writer.close()

