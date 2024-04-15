import torch
from torch.utils.data import DataLoader
from psyonet_dataset import PhysioNetDataset
from EMGClassifier import EMGClassifier

# Size of all datasets
input_size = 5000  # Change if needed

def load_models(model_paths):
    models = []
    for path in model_paths:
        model = EMGClassifier(input_size)
        model.load_state_dict(torch.load(path))
        model.eval()
        models.append(model)
    return models

def ensemble_inference(models, dataloader):
    predictions = []
    with torch.no_grad():
        for data in dataloader:
            emg_signals = data['emg']
            model_predictions = []
            for model in models:
                outputs = model(emg_signals)
                _, predicted = torch.max(outputs, 1)
                model_predictions.append(predicted.item())
            predictions.append(model_predictions)
    return predictions

def aggregate_predictions(predictions):
    final_predictions = []
    for preds in predictions:
        # Example: simple voting
        final_prediction = max(set(preds), key=preds.count)
        final_predictions.append(final_prediction)
    return final_predictions

model1= 'C:\\Users\\Jakeeer\\git\\senior_project\\machine learning practice\\EMG_trained2_0.pth'
model2= 'C:\\Users\\Jakeeer\\git\\senior_project\\machine learning practice\\EMG_trained2_1.pth'
model3= 'C:\\Users\\Jakeeer\\git\\senior_project\\machine learning practice\\EMG_trained2_2.pth'

if __name__ == "__main__":
    # Load multiple trained models
    model_paths = [model1, model2, model3]  # Update with your model paths
    models = load_models(model_paths)

    # Define the dataset for the test data
    test_dat_dir = 'C:\\Users\\Jakeeer\\git\\senior_project\\machine learning practice\\test_database'  # CHANGE 
    test_dataset = PhysioNetDataset(test_dat_dir, max_length=input_size)

    # Define batch size and DataLoader for test data
    batch_size = 1  # Set batch size to 1 for inference
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # Shuffle should be False for inference

    # Perform inference with ensemble of models
    predictions = ensemble_inference(models, test_dataloader)

    # Aggregate predictions
    final_predictions = aggregate_predictions(predictions)

    # Print or process the final predictions as needed
    print("Final Predictions:", final_predictions)
