import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Check if CUDA is available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Function to display the image
def imshow(img):
    np_img = img.numpy()
    plt.axis('off')
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()
    
    
#function to compute loss and accuracy for test set
def evaluate(model, testloader, critetion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            
            outputs= model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        accuracy = 100 * correct / total
        test_loss = test_loss / len(testloader)
        return test_loss, accuracy


if __name__ == '__main__':
    # Define data transformation
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0,), (1.0,))])

    # Load the dataset
    trainset = torchvision.datasets.FashionMNIST(root='data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1024, num_workers=10, shuffle=True)

    testset = torchvision.datasets.FashionMNIST(root='data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1024, num_workers=10, shuffle=False)


    # Display some images
    for i, (images, labels) in enumerate(trainloader, 0):
        imshow(torchvision.utils.make_grid(images[:8]))  # Display 8 images from the batch
        break
    
    
    #model
    model = nn.Sequential(
        nn.Flatten(), nn.Linear(784, 10)
    )
    model = model.to(device)
    
    print(model)
    
    
    #generating a random tensor
    input_tensor = torch.rand(5, 28, 28).to(device)
    
    #feeding the tensor into the model
    output = model(input_tensor)
    print(output.shape)
    
    
    #cross-entropy loss, SGD algo
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    
    '''Training process'''
    #some parameters
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    max_epoch = 100
    
    #traing
    for epoch in range(max_epoch):
        running_loss = 0.0
        running_correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(trainloader, 0):
            inputs, labels = inputs.to(device), labels.to(device)
            
            #zero the parameter gradients
            optimizer.zero_grad()
            
            #forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            #determine class prediction and track accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            running_correct += (predicted == labels).sum().item()
            
            #backward pass and optimizer
            loss.backward()
            optimizer.step()
            
        
        epoch_accuracy = 100 * running_correct / total
        epoch_loss = running_loss / (i + 1)
        test_loss, test_accuracy = evaluate(model, testloader, criterion)
        print(f"Epoch [{epoch + 1}/{max_epoch}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
        
        #save data for plotting
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        
    
    '''Plotting training process graph'''
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].plot(train_losses, label='train_losses')
    ax[0].plot(test_losses, label='test_losses')
    ax[0].set_title('Loss in training process')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    
    ax[1].plot(train_accuracies, label='train_accuracy')
    ax[1].plot(test_accuracies, label='test_accuracy')
    ax[0].set_title('Accuracy in training process')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Accuracy')
    ax[1].legend()
    
    plt.show()
            
    
    
    
    