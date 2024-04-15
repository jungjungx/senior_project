import torch
import torch.nn as nn
import torch.nn.functional as F

class EMGModel(nn.Module):
    def __init__(self, in_features=1000, h1=64, h2=32, h3=16, out_features=2):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)  # New hidden layer
        self.out = nn.Linear(h3, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))  # Activation applied to the output of the new hidden layer
        x = self.out(x)
        return x
