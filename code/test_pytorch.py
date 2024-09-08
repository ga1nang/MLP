import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary

#sample
X = torch.tensor([1.0, 2.0])
y = torch.tensor([0.0])


#model
model = nn.Sequential(
    nn.Linear(2, 1),
    nn.Linear(1, 1),
    nn.Sigmoid()
)

print(model)
summary(model, (5, 2))

for layer in model.children():
    print(layer.state_dict())