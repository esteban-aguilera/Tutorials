import numpy as np
import os
import torch

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from torch import nn, optim


# --------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------
def main():
    test_feed_forward(train=False)


# --------------------------------------------------------------------------------
# neural networks
# --------------------------------------------------------------------------------
class FeedForwardNetwork(nn.Module):
    """Feed-forward neural network
    """

    def __init__(self):
        super().__init__()
        
        # fully connected layers.  nn.Linear(dim_input, dim_output)
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        # first layer
        x = nn.functional.relu( self.fc1(x) )

        # hidden layers
        x = nn.functional.relu( self.fc2(x) )
        x = nn.functional.relu( self.fc3(x) )
        
        # last layer
        x = nn.functional.log_softmax(self.fc4(x), dim=1)
        
        return x


# --------------------------------------------------------------------------------
# testers
# --------------------------------------------------------------------------------
def test_feed_forward(batch_size=16, train=False, path='models', filename='mnist.pth'):
    X_data, y_data = get_dataset()
    
    # reshape X_data
    X_data = np.reshape(X_data, (X_data.shape[0], -1))

    # split data
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

    # reshape data and transform it to a tensor
    X_train = torch.Tensor( np.reshape(X_train, (-1, batch_size, 28*28)) ).float()
    y_train = torch.Tensor( np.reshape(y_train, (-1, batch_size)) ).long()
    X_test = torch.Tensor( np.reshape(X_test, (-1, 1, 28*28)) ).float()
    y_test = torch.Tensor( y_test ).long()
    
    # create Network and load any existing previous model
    model = FeedForwardNetwork()

    # define optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    EPOCHS = 3

    # load previous model
    if(os.path.exists(f'{path}/{filename}')):
        model.load_state_dict( torch.load(f'{path}/{filename}') )
    else:
        train = True
    
    # train model
    if(train is True):
        for epoch in range(EPOCHS):
            acc = 0
            for X, y in zip(X_train, y_train):
                # zero the parameter gradients
                model.zero_grad()

                # forward + backward + optimize
                output = model(X)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                
                # calculate accuracy
                acc += (output.argmax(dim=1) == y).sum().item()
            
            acc = acc / (batch_size * len(X_train))

            print(f'\t epoch {epoch+1}: {loss.item()} \t acc: {100 * acc}%')

        # save trained model
        mkdir(path)
        torch.save(model.state_dict(), f'{path}/{filename}')
    
    # evaluate model
    acc = 0
    with torch.no_grad():  # don't calculate gradients
        for X, y in zip(X_test, y_test):
            # create prediction
            outputs = model(X)
            
            # test prediction
            acc += (outputs.argmax(dim=1).item() == y.item())
        
        print(acc, len(y_test))
        acc = acc / y_test.numel()
        print(f'test accuracy: \t {100 * acc}%')


# --------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------
def get_dataset():
    name = 'mnist_784'

    if(os.path.exists(f'data/{name}') is True):
        # load data
        X_data = np.load(f'data/{name}/X_data.npy')
        y_data = np.load(f'data/{name}/y_data.npy')
    else:
        print('fetching data')
        # get data
        data = fetch_openml(name, version=1)
            
        # transform data
        X_data = np.reshape(data['data'], (data['data'].shape[0], 28, 28))
        y_data = data['target'].astype(int)
        
        # create necessary directories
        mkdir('data/')
        mkdir(f'data/{name}')

        # save data
        np.save(f'data/{name}/X_data.npy', X_data)
        np.save(f'data/{name}/y_data.npy', y_data)
        
    # # encode y_data in a one-hot vector
    # one_hot_encoder = OneHotEncoder()
    # y_data = one_hot_encoder.fit_transform(y_data.reshape((-1,1)))
    # y_data = y_data.toarray()

    return X_data, y_data


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