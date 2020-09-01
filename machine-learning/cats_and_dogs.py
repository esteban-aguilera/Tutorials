import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm  # its a progress bar for for-loops


# --------------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------------
EPOCHS = 3
BATCH_SIZE = 32


# --------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------
def main():
    # cc = CatsAndDogs()
    # cc.make_training_data()

    X_train, y_train, X_test, y_test = get_data()
    model = ConvClassifier()
    
    train(model, (X_train, y_train))
    print('\n')
    test(model, (X_test, y_test))


# --------------------------------------------------------------------------------
# functions
# --------------------------------------------------------------------------------
def train(model, train_data, path='models', filename='cats_and_dogs.pth'):
    X_train, y_train = train_data

    # define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()
    
    # load model
    if(os.path.exists(f'{path}/{filename}') is True):
        model.load_state_dict( torch.load(f'{path}/{filename}') )
    else:
        pass
    
    acc = 0
    # train model
    for epoch in range(EPOCHS):
        for i in tqdm( range(0, len(X_train), BATCH_SIZE) ):
            X = X_train[i:i+BATCH_SIZE].view(-1, 1, 50, 50)
            y = y_train[i:i+BATCH_SIZE].long()

            # zero the gradients
            model.zero_grad()

            # forward + backward + optimize
            output = model(X)
            loss = loss_function(output, y)
            loss.backward()
            optimizer.step()

            # calculate accuracy
            acc += (output.argmax(dim=1) == y).sum().item()

        acc = acc / len(X_train)

        print(f'\t epoch {epoch+1}: {loss.item()} \t acc: {100 * acc}%')
    
    # save trained model
    mkdir(path)
    torch.save(model.state_dict(), f'{path}/{filename}')
    

def test(model, test_data, path='models', filename='cats_and_dogs.pth'):
    X_test, y_test = test_data

    # load model
    model.load_state_dict( torch.load(f'{path}/{filename}') )

    acc = 0
    # test model
    with torch.no_grad():  # don't calculate gradients
        for i in tqdm( range(0, len(X_test), BATCH_SIZE) ):
            X = X_test[i:i+BATCH_SIZE].view(-1, 1, 50, 50)
            y = y_test[i:i+BATCH_SIZE].long()

            # create prediction
            outputs = model(X)
            
            # test prediction
            acc += (outputs.argmax(dim=1) == y).sum().item()
        
        print(acc, len(y_test))
        acc = acc / y_test.numel()
        print(f'test accuracy: \t {100 * acc}%')


def get_data(train_pct=0.8):
    # obtain data
    data = np.load('data/pet_images/training_data.npy', allow_pickle=True)
    # separate input and labels
    X_data = torch.Tensor(list(data[:,0])).view(-1, 1, 50, 50)
    X_data = X_data / 255.0
    y_data = torch.Tensor(list(data[:,1]))
    
    # number of training data
    ntrain = int(train_pct * len(X_data))

    # split data
    X_train, y_train = X_data[:ntrain], y_data[:ntrain]    
    X_test, y_test = X_data[ntrain:], y_data[ntrain:]
    
    print(len(X_data), len(X_train), len(X_test))

    return X_train, y_train, X_test, y_test

# --------------------------------------------------------------------------------
# ConvClassifier
# --------------------------------------------------------------------------------
class ConvClassifier(nn.Module):

    def __init__(self):
        super().__init__()
        
        # convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5)
        
        # obtain number of neurons after convolutional layers
        img_size = CatsAndDogs.IMG_SIZE
        x = torch.randn(img_size, img_size).view(-1, 1, img_size, img_size)
        x = self.convolutional_layers(x)
        self.num_neurons = np.prod( np.shape(x)[1:] )
        
        # fully connected layers
        self.fc1 = nn.Linear(self.num_neurons, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.convolutional_layers(x)
        x = x.view(-1, self.num_neurons)
        x = self.fully_connected_layers(x)

        return x

    def fully_connected_layers(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)

        return x

    def convolutional_layers(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, (2,2))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, (2,2))
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, (2,2))

        return x


# --------------------------------------------------------------------------------
# CatsAndDogs
# --------------------------------------------------------------------------------
class CatsAndDogs():
    IMG_SIZE = 50
    CATS = 'data/pet_images/cats'
    DOGS = 'data/pet_images/dogs'
    LABELS = {CATS: 0, DOGS: 1}
    
    def __init__(self):
        self.training_data = []
        self.ncats = 0
        self.ndogs = 0

    def make_training_data(self):
        self.ncats = 0
        self.ndogs = 0

        for label in self.LABELS:
            for filename in tqdm( os.listdir(label) ):
                try:
                    # obtain image's path
                    path = os.path.join(label, filename)

                    # obtain the image
                    img = cv2.imread(path, flags=cv2.IMREAD_GRAYSCALE)
                    # resize image
                    img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                    
                    # add image and target to training_data
                    self.training_data.append([np.array(img), self.LABELS[label]])

                    if(label == self.CATS):
                        self.ncats += 1
                    elif(label == self.DOGS):
                        self.ndogs += 1

                except:
                    # we have an exception because reasons...
                    pass
            
        # inplace shuffling
        np.random.shuffle(self.training_data)

        # save data
        np.save('data/pet_images/training_data.npy', self.training_data)

        print(f'number of cats: {self.ncats}')
        print(f'number of dogs: {self.ndogs}')


# --------------------------------------------------------------------------------
# utils
# --------------------------------------------------------------------------------
def mkdir(path):
    if(os.path.exists(path) is False):
        os.mkdir(path)


# --------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    main()