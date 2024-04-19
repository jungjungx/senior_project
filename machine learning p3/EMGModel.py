import torch
import torch.nn as nn
import torch.nn.functional as F

class EMGModel(nn.Module):
    def __init__(self, input_size=250, hidden_size=126, hidden_size2=64, num_layers=2, num_classes=2):
        super(EMGModel, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_size2 = hidden_size2
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size2, num_layers, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size2, num_classes)
    
    def forward(self, x):
        # Initialize hidden states with zeros
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # Forward propagate through the first LSTM layer
        out, _ = self.lstm1(x, (h0, c0))
        
        # Forward propagate through the second LSTM layer
        h0_2 = torch.zeros(self.num_layers, batch_size, self.hidden_size2).to(x.device)
        c0_2 = torch.zeros(self.num_layers, batch_size, self.hidden_size2).to(x.device)
        out, _ = self.lstm2(out, (h0_2, c0_2))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out
