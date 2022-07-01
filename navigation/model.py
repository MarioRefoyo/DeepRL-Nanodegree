import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class QNetworkCNN(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, input_size, action_size, seed, cnn1_filters=32, cnn2_filters=64, cnn3_filters=128, fc1_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetworkCNN, self).__init__()
        num_channels = input_size[0]
        height = input_size[1]
        width = input_size[2]

        # Conv layers
        self.seed = torch.manual_seed(seed)
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=cnn1_filters, kernel_size=(4, 4))
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=cnn1_filters, out_channels=cnn2_filters, kernel_size=(3, 3))
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv3 = nn.Conv2d(in_channels=cnn2_filters, out_channels=cnn3_filters, kernel_size=(3, 3))
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2))

        # Calculate the output sizes
        w1 = np.floor((width - 4)/1 + 1)
        w1 = np.floor(w1 / 2)
        print(w1)
        w2 = np.floor((w1 - 3) / 1 + 1)
        w2 = np.floor(w2 / 2)
        print(w2)
        w3 = np.floor((w2 - 3) / 1 + 1)
        w3 = np.floor(w3 / 2)
        print(w3)

        # FC layers
        self.fc1 = nn.Linear(in_features=int(w3)*int(w3)*cnn3_filters, out_features=fc1_units)
        self.fc2 = nn.Linear(fc1_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""

        # CNNs
        x = F.relu(self.conv1(state))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = F.relu(self.conv3(x))
        x = self.maxpool3(x)

        # FCs
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
