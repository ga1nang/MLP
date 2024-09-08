import matplotlib.pyplot as plt
import numpy as np

from torchvision.datasets import FashionMNIST
from torchvision import transforms
from torch.utils.data import DataLoader


#load train dataset
transform = transforms.Compose([transforms.ToTensor()])
trainset = FashionMNIST(root='data/fashion_mnist_pytoch',
                        train=True,
                        download=True, transform=transform)

#mini-batch
trainloader = DataLoader(trainset, 
                         batch_size=3500, 
                         num_workers=2, 
                         shuffle=True)
print(len(trainloader))


#plot trained image
# plt.figure(figsize=(2, 2))
# plt.imshow(img.squeeze(), cmap='gray')
# plt.axis('off')
# plt.show()

#plot randomed 100 images
indices = list(np.random.randint(60000, size=100))
fig = plt.figure(figsize=(10, 10))
columns = 10
rows = 10
for i in range(columns * rows):
    img, label = trainset[indices[i]]
    fig.add_subplot(rows, columns, i+1)
    plt.axis('off')
    plt.imshow(img.squeeze(), cmap='gray', vmin=0, vmax=1)
    
plt.show()