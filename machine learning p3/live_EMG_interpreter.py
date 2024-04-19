import torch
import pandas as pd
import numpy as np
from EMGModel import EMGModel
from live_csvcompiler import convert_csv
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

#model directory
model_file = 'C:\\Users\\Jakeeer\\git\\senior_project\\machine learning p3\\EMG_trained.pth'
#for live csv compiler
input_file = "C:\\Users\\Jakeeer\\git\\senior_project\\machine learning p3\\VoltageTest.csv"
output_file = "C:\\Users\\Jakeeer\\git\\senior_project\\machine learning p3\\VoltageTest_compiled.csv"

def live_interpret():
    # Preprocess data as needed
    convert_csv(input_file,output_file)

    # Read data from preprocessed data
    data = pd.read_csv(output_file,dtype=float,header=None)
    #print(data.head())

    # Convert data to NumPy array
    X = data.values

    # Convert NumPy array to PyTorch tensor
    X_test = torch.FloatTensor(X)
    X_test = X_test.unsqueeze(-1)  # Add an additional dimension at the end
    X_test = X_test.permute(0, 2, 1)  # Permute dimensions to match expected shape

    model = EMGModel()  # Create an instance of your model
    model.load_state_dict(torch.load(model_file))  #CHANGE

    # Perform inference with your model
    # Assuming model is already loaded and ready for inference
    with torch.no_grad():
        predictions = model(X_test)

    # Process predictions as needed
    # For example, print the predictions or perform further analysis
    print(predictions)

    rest_val, contraction_val = predictions[0]

    if rest_val > contraction_val:
        print("Resting EMG signal")
    else:
        print("Contracting EMG Signal")

if __name__ == "__main__":
    live_interpret()
