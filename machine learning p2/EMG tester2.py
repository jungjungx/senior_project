import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from EMGModel import EMGModel

df = pd.read_csv("C:\\Users\\Jakeeer\\git\\senior_project\\machine learning p2\\emg_compiled_test.csv")

# Convert labels to integers
label_mapping = {'rest': 0,'contraction':1}  # Define a mapping from labels to integers
df['Label'] = df['Label'].map(label_mapping)   # Convert labels using the mapping

# Assuming the last column in your CSV file contains the labels
# Extract features (X) by dropping the label column
X = df.drop(df.columns[-1], axis=1)

# Extract labels (y) from the last column
y = df[df.columns[-1]]

#print(X.head(20))
#print(y.head(20))

# Convert these to numpy arrays
X = X.values
y = y.values


#Testing data
X_test_tensor = torch.FloatTensor(X) #change
X_test_tensor = X_test_tensor.unsqueeze(-1)  # Add an additional dimension at the end
X_test_tensor = X_test_tensor.permute(0, 2, 1)  # Permute dimensions to match expected shape

y_test_tensor = torch.LongTensor(y) #change

model = EMGModel()  # Create an instance of your model
model.load_state_dict(torch.load('C:\\Users\\Jakeeer\\git\\senior_project\\machine learning p2\\EMG_trained.pth'))  #CHANGE
criterion = nn.CrossEntropyLoss()

# Evaluate Model on Test Data Set (validate model on test set)
with torch.no_grad():
    correct = 0
    for i, data in enumerate(X_test_tensor):
        # Add a batch dimension for each sample
        data = data.unsqueeze(0)
        
        # Forward pass through the model
        y_val = model(data)

        # Get the predicted label
        predicted_label = y_val.argmax().item()

        # Compare with the actual label
        actual_label = y_test_tensor[i].item()

        # Print predictions and actual labels
        print(f"Sample {i+1}: {predicted_label}, {actual_label}")

        # Print tensor values
        print(y_val)

        # Update correct predictions count
        if predicted_label == actual_label:
            correct += 1

# Calculate accuracy
accuracy = correct / len(X_test_tensor) * 100
print(f"Accuracy: {accuracy:.2f}%")