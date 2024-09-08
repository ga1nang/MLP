import numpy as np
import matplotlib.pyplot as plt
import gzip

from urllib import request


#download data
filenames = ["train-images-idx3-ubyte.gz",
             "train-labels-idx1-ubyte.gz",
             "t10k-images-idx3-ubyte.gz",
             "t10k-labels-idx1-ubyte.gz"]

folder = 'data\\fashion_mnist\\'
# base_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
# for name in filenames:
#     print("Downloading " + name + "...")
#     request.urlretrieve(base_url + name, folder+name)

#load training images and labels
with gzip.open(folder + filenames[0], 'rb') as f:
    X_train = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28*28)

with gzip.open(folder + filenames[1], 'rb') as f:
    y_train = np.frombuffer(f.read(), np.uint8, offset=8)
    
#load testing images and labels
with gzip.open(folder + filenames[2], 'rb') as f:
    X_test = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28*28)
    
with gzip.open(folder + filenames[3], 'rb') as f:
    y_test = np.frombuffer(f.read(), np.uint8, offset=8)
    

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

#get a random list of 100 elements from X_train
indices = list(np.random.randint(X_train.shape[0], size=100))

#plot the images
fig = plt.figure(figsize=(9, 9))
columns = 10
rows = 10
for i in range(columns * rows):
    img = X_train[indices[i]].reshape(28, 28)
    fig.add_subplot(rows, columns, i + 1)
    plt.axis('off')
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    
plt.show()