import torch
import numpy as np

pred =[torch.tensor([[372.00000, 270.50000, 490.00000, 479.50000,   0.77246,  41.00000],
        [  3.00000,   8.00000, 399.75000, 489.50000,   0.74219,   0.00000]], device='cuda:0')]
print(pred)

det = pred[0]
print(det)

print(det[:, :4])