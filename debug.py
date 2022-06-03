import torch

from model.engine.loss_functions import HingeLoss

loss_fn = HingeLoss(d_train=True)

input_tensor = torch.tensor([[-3, -2, -1, 0, 1, 2, 3], [-3, -2, -1, 0, 1, 2, 3], [-3, -2, -1, 0, 1, 2, 3]])
input_tensor = input_tensor.unsqueeze(0).unsqueeze(3).float()
print(input_tensor, input_tensor.shape)

output = loss_fn(input_tensor, 0)

print(output, output.shape)