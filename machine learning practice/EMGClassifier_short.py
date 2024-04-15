# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from psyonet_dataset import PhysioNetDataset
from psyonet_dataset import SinglePhysioNetDataset

#size of all datasets
input_size = 6400
num_channels = 3

# Define neural network model
class EMGClassifier(nn.Module):
    def __init__(self, input_size):
        super(EMGClassifier, self).__init__()
        # Define your model architecture here
        self.conv1 = nn.Conv1d(num_channels, 128, kernel_size=1).float()  # Change the number of input channels
        # Calculate the output size of the convolutional layer
        conv_output_size = self._get_conv_output_size(input_size)
        self.fc = nn.Linear(conv_output_size, 10).float()  # Adjusting the input size of the linear layer

    def forward(self, x):
        # Define the forward pass of your model
        x = x.float()  # Convert input to float
        x = x.unsqueeze(1)  # Add a channel dimension
        x = self.conv1(x)
        x = torch.flatten(x, 1)  # Flatten the output of the convolutional layer
        x = self.fc(x)
        return x
    
    def _get_conv_output_size(self, input_size):
        # Function to calculate the output size of the convolutional layer
        with torch.no_grad():
            input_tensor = torch.zeros(1, num_channels, input_size).float()  # Adjust the number of channels
            conv_output = self.conv1(input_tensor)
            conv_output_size = conv_output.size(2) 
        return conv_output_size

if __name__ == "__main__":
    # Define dataset directory and instantiate PhysioNetDataset
    dat_dir = 'C:\\Users\\Jakeeer\\git\\senior_project\\machine learning practice\\database' #CHANGE
    physionet_dataset = PhysioNetDataset(dat_dir,None,input_size) #contains all datasets

    sphysionet_datasets = [] #array of single datasets
    
    for idx in range(len(physionet_dataset)):
            sample = physionet_dataset[idx]
            #sphysionet_datasets[idx] = SinglePhysioNetDataset(sample['emg'],sample['label'])
            sphysionet_datasets.append(SinglePhysioNetDataset(sample['emg'], sample['label'],input_size))

    # Define batch size and DataLoader
    batch_size = 1

    models=[]
    # Iterate over each dataset within physionet_dataset
    for dataset in sphysionet_datasets:
        # Create DataLoader for the current dataset
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print("Length of DataLoader:", len(dataloader))
        # Instantiate the model, loss function, and optimizer
        model = EMGClassifier(input_size)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        # Training loop
        num_epochs = 5
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, data in enumerate(dataloader, 0):
                emg_signals, labels = data['emg'], data['label']
                print(emg_signals.shape)
                print(emg_signals)

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

        print('Finished Training for dataset ')
        models.append(model)

    # Save the trained model
    for i, model in enumerate(models):
        torch.save(model.state_dict(), f'C:\\Users\\Jakeeer\\git\\senior_project\\machine learning practice\\EMG_trained2_{i}.pth')

    print('All models trained and saved successfully.')
