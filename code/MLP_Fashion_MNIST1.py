import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#functon show image
def imshow(img):
  img = img * 0.5 + 0.5
  np_img = img.numpy()
  plt.axis('off')
  plt.imshow(np.transpose(np_img, (1, 2, 0)))
  plt.show()


#function to evaluate
def evaluate(model, testloader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    test_loss = test_loss / len(testloader)
    return test_loss, accuracy


if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    trainset = torchvision.datasets.FashionMNIST(root='data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1024, num_workers=10, shuffle=True)

    testset = torchvision.datasets.FashionMNIST(root='data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1024, num_workers=10, shuffle=False)
    
    
    #show image
    # for i, (images, labels) in enumerate(trainloader, 0):
    #     imshow(torchvision.utils.make_grid(images[:8]))
    #     break
    
    
    #load trained model in gg colab
    
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu'), weights_only=True))
    
    #loss function
    criterion = nn.CrossEntropyLoss()
    
    #evaluate model
    test_loss, test_accuracy = evaluate(model, testloader, criterion)
    print(f'test_loss: {test_loss}')
    print(f'test_accuracy: {test_accuracy}')


