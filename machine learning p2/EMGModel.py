import torch
import torch.nn as nn
import torch.nn.functional as F

class EMGModel(nn.Module):
    def __init__(self, input_size=1000, hidden_size=512,hidden_size2=256,hidden_size3=128,hidden_size4=64, num_layers=2, num_classes=2):
        super(EMGModel, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_size2 = hidden_size2
        self.hidden_size3 = hidden_size3
        self.hidden_size4 = hidden_size4
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size2, num_layers, batch_first=True)
        self.lstm3 = nn.LSTM(hidden_size2, hidden_size3, num_layers, batch_first=True)  # Additional hidden layer
        self.lstm4 = nn.LSTM(hidden_size3, hidden_size4, num_layers, batch_first=True)  # Additional hidden layer
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size4, num_classes)
    
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
        
        # Forward propagate through the third LSTM layer
        h0_3 = torch.zeros(self.num_layers, batch_size, self.hidden_size3).to(x.device)
        c0_3 = torch.zeros(self.num_layers, batch_size, self.hidden_size3).to(x.device)
        out, _ = self.lstm3(out, (h0_3, c0_3))
        
        # Forward propagate through the fourth LSTM layer
        h0_4 = torch.zeros(self.num_layers, batch_size, self.hidden_size4).to(x.device)
        c0_4 = torch.zeros(self.num_layers, batch_size, self.hidden_size4).to(x.device)
        out, _ = self.lstm4(out, (h0_4, c0_4))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


