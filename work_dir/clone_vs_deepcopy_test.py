import numpy as np
import torch
from copy import deepcopy

print("start")

print(torch.cuda.max_memory_allocated("cuda:0"))

tensor_original = torch.randn((1024,1024), device="cuda:0")

print(torch.cuda.max_memory_allocated("cuda:0"))

tensor_deepcopy = deepcopy(tensor_original)

print(torch.cuda.max_memory_allocated("cuda:0"))

tensor_clone = torch.clone(tensor_original)

print(torch.cuda.max_memory_allocated("cuda:0"))

tensor_2 = tensor_original
print("o, d, c")
print(tensor_original[0,0])
print(tensor_deepcopy[0,0])
print(tensor_clone[0,0])

tensor_clone[0] = 1

tensor_2[0,0] = 1

print("o, d, c, 2")
print(tensor_original[0,0])
print(tensor_deepcopy[0,0])
print(tensor_clone[0,0])
print(tensor_2[0,0])