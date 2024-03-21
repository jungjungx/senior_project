# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from psyonet_dataset import PhysioNetDataset

#size of all datasets
input_size = 50000

# Define neural network model
class EMGClassifier(nn.Module):
    def __init__(self,input_size):
        super(EMGClassifier, self).__init__()
        # Define model architecture here
        self.fc1 = nn.Linear(input_size, 128).double()
        self.fc2 = nn.Linear(128, 10).double()

    def forward(self, x):
        #print(x)
        x = x.view(-1, self.fc1.in_features)  # Flatten input
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    # Define dataset directory and instantiate PhysioNetDataset
    dat_dir = 'C:\\Users\\Jakeeer\\Desktop\\Senior Project\\database' #CHANGE
    physionet_dataset = PhysioNetDataset(dat_dir,None,input_size)

    # Define batch size and DataLoader
    batch_size = 64
    dataloader = torch.utils.data.DataLoader(physionet_dataset, batch_size=batch_size, shuffle=True)

    # Instantiate the model, loss function, and optimizer
    model = EMGClassifier(input_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            #inputs, labels = data
            emg_signals, labels = data['emg'], data['label']

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass + backward pass + optimize
            outputs = model(emg_signals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # Print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    print('Finished Training')

    # Save the trained model
    torch.save(model.state_dict(), 'C:\\Users\\Jakeeer\\Desktop\\Senior Project\\EMG_trained.pth') #CHANGE
