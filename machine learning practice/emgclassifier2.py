import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from psyonet_dataset import PhysioNetDataset
from psyonet_dataset import SinglePhysioNetDataset
from psyonet_dataset import MultiPhysioNetDataset

num_classes = 3
input_size = 6400

dat_dir = 'C:\\Users\\Jakeeer\\git\\senior_project\\machine learning practice\\database' #CHANGE
physionet_dataset = PhysioNetDataset(dat_dir,None,input_size) #contains all datasets

healthy = physionet_dataset[0]
myopathy = physionet_dataset[1]
neuropathy = physionet_dataset[2]
physionet_datasetv2 = MultiPhysioNetDataset(healthy['emg'],myopathy['emg'],neuropathy['emg'])

train_dataset = physionet_datasetv2
print(len(train_dataset))

for i in range(5):  # Print the first 5 samples
    sample = train_dataset[i]
    print(f"Sample {i}: {sample}")
    sample = train_dataset[i+6400]
    print(f"Sample {i}: {sample}")
    sample = train_dataset[i+6400*2-1]
    print(f"Sample {i}: {sample}")

class EMGNet(nn.Module):
    def __init__(self):
        super(EMGNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 125, 128)  # Assuming input signal length of 5000 and max pooling of 4
        self.fc2 = nn.Linear(128, num_classes)  # Adjust num_classes according to your problem

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, 2)
        x = x.view(-1, 32 * 125)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define model, loss function, and optimizer
model = EMGNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Assuming you have prepared your datasets and data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
#val_loader = DataLoader(val_dataset, batch_size=64)
#test_loader = DataLoader(test_dataset, batch_size=64)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, batch in enumerate(train_loader):
        inputs, labels = batch['emg'], batch['label']
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")

torch.save(model.state_dict(), f'C:\\Users\\Jakeeer\\git\\senior_project\\machine learning practice\\EMG_trainedv3.pth')

'''
# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in test_loader:
        inputs, labels = batch['emg'], batch['label']
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on test set: {100 * correct / total}%")
'''
