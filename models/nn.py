import torch.nn as nn
import torch.nn.functional as F


class SimpleNet(nn.Module):
    def __init__(self, hidden_1, dropout, output, input):
        super(SimpleNet, self).__init__()

        self.fc1 = nn.Linear(input, hidden_1)
        self.fc2 = nn.Linear(hidden_1, output)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x


class Net(nn.Module):

    def __init__(self, hidden_1, hidden_2, dropout, output,
                 input
                 ):

        super(Net, self).__init__()
        self.fc1 = nn.Linear(input, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_2, output)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.out(x)
        return x


class Layer3Net(nn.Module):

    def __init__(self, hidden_1, hidden_2, hidden_3,
                 dropout, output, input
                 ):

        super(Layer3Net, self).__init__()
        self.fc1 = nn.Linear(input, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc3 = nn.Linear(hidden_2, hidden_3)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_3, output)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.drop(x)
        x = self.out(x)
        return x


def create_simple_net(hidden_1, dropout, output, input):
    net = SimpleNet(hidden_1, dropout, output, input)
    return net


def create_net(hidden_1=300, hidden_2=100, dropout=0.1, output=2,
               input=85):
    net = Net(hidden_1, hidden_2, dropout, output,
              input)
    return net


"""
def create_net(params):
    net = Net(hidden_1=params['hidden1'],
              hidden_2=params['hidden2'],
              dropout=params['dropout'],
              output=params['output'],
              input=params['input'])
    return net
"""


def create_layer3net(hidden_layer_1, hidden_layer_2, hidden_layer_3,
                     dropout_rate, n_classes, n_features):
    net = Layer3Net(hidden_layer_1, hidden_layer_2, hidden_layer_3,
                    dropout_rate, n_classes, n_features)
    return net
