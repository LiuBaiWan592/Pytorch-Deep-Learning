import torch
from torch import nn

input = torch.tensor([1, 2, 3], dtype=torch.float32)
target = torch.tensor([1, 2, 5], dtype=torch.float32)

# 均值损失函数
l1_loss = nn.L1Loss(reduction="sum")

result_l1 = l1_loss(input, target)
print(result_l1)

# 均方差损失函数
mse_loss = nn.MSELoss()

result_MSE = mse_loss(input, target)
print(result_MSE)

# 交叉熵损失函数（多分类问题）
x = torch.tensor([0.1, 0.2, 0.3])
y = torch.tensor([1])
x = torch.reshape(x, (1, 3))

# loss(x, class) = -x * class + log[exp(x[0]) + exp(x[1]) + exp(x[2]) + ···]
cross_loss = nn.CrossEntropyLoss()
result_CrossEntropy = cross_loss(x, y)
print(result_CrossEntropy)
