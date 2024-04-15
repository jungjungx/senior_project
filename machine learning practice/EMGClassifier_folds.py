# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from psyonet_dataset import PhysioNetDataset
from sklearn.model_selection import KFold
import numpy as np

#size of all datasets
input_size = 50000

# Define neural network model
class EMGClassifier(nn.Module):
    def __init__(self, input_size):
        super(EMGClassifier, self).__init__()
        # Define model architecture here
        self.fc1 = nn.Linear(input_size, 128).double()
        self.fc2 = nn.Linear(128, 10).double()

    def forward(self, x):
        x = x.view(-1, self.fc1.in_features)  # Flatten input
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    # Define dataset directory and instantiate PhysioNetDataset
    dat_dir = 'C:\\Users\\Jakeeer\\git\\senior_project\\machine learning practice\\database' #CHANGE
    physionet_dataset = PhysioNetDataset(dat_dir, None, input_size)

    # Define batch size and DataLoader
    batch_size = 64
    dataloader = torch.utils.data.DataLoader(physionet_dataset, batch_size=batch_size, shuffle=True)

    # If dataset size is greater than 50000, perform cross-validation
    if len(physionet_dataset) >= 50000:
        print("folding...")
        # Define the number of folds for cross-validation
        num_folds = 10
        kf = KFold(n_splits=num_folds)
        fold_accuracies = []

        for fold, (train_indices, val_indices) in enumerate(kf.split(physionet_dataset)):
            print(f'Fold {fold + 1}/{num_folds}')
            train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
            val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

            train_loader = torch.utils.data.DataLoader(physionet_dataset, batch_size=batch_size, sampler=train_sampler)
            val_loader = torch.utils.data.DataLoader(physionet_dataset, batch_size=batch_size, sampler=val_sampler)

            # Instantiate the model, loss function, and optimizer
            model = EMGClassifier(input_size)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

            # Training loop
            num_epochs = 5
            for epoch in range(num_epochs):
                running_loss = 0.0
                for i, data in enumerate(train_loader, 0):
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

            # Evaluate the model on the validation set
            with torch.no_grad():
                model.eval()
                correct = 0
                total = 0
                for data in val_loader:
                    emg_signals, labels = data['emg'], data['label']
                    outputs = model(emg_signals)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                accuracy = correct / total
                fold_accuracies.append(accuracy)
                print(f'Validation Accuracy (Fold {fold + 1}): {accuracy}')

        # Calculate mean accuracy across all folds
        mean_accuracy = np.mean(fold_accuracies)
        print(f'Mean Accuracy: {mean_accuracy}')

    else:
        # If dataset size is less than 50000, train the model without cross-validation
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
    torch.save(model.state_dict(), 'C:\\Users\\Jakeeer\\git\\senior_project\\machine learning practice\\EMG_trained_fold.pth') #CHANGE
