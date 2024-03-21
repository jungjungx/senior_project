import torch
from torch.utils.data import DataLoader
from psyonet_dataset import PhysioNetDataset
from EMGClassifier import EMGClassifier

#size of all datasets
input_size = 50000

if __name__ == "__main__":
    # Load the trained model
    model = EMGClassifier(input_size)  # Create an instance of your model
    model.load_state_dict(torch.load('C:\\Users\\Jakeeer\\Desktop\\Senior Project\\EMG_trained.pth'))  #CHANGE

    # Set the model to evaluation mode
    model.eval()

    # Define the dataset for the test data
    test_dat_dir = 'C:\\Users\\Jakeeer\\Desktop\\Senior Project\\test_database'  # CHANGE - Path to directory containing test data files
    test_dataset = PhysioNetDataset(test_dat_dir,max_length=input_size)

    # Define batch size and DataLoader for test data
    batch_size = 1  # Set batch size to 1 for inference
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # Shuffle should be False for inference

    # Perform inference on test data
    predictions = []
    with torch.no_grad():  # Disable gradient tracking during inference
        for data in test_dataloader:
            emg_signals = data['emg']
            outputs = model(emg_signals)
            _, predicted = torch.max(outputs, 1)  # Get the predicted class with the highest probability
            predictions.append(predicted.item())  # Append the predicted class to the list of predictions

    # Print or process the predictions as needed
    print("Predictions:", predictions)

