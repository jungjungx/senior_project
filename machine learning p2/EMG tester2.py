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
y_test_tensor = torch.LongTensor(y) #change

model = EMGModel()  # Create an instance of your model
model.load_state_dict(torch.load('C:\\Users\\Jakeeer\\git\\senior_project\\machine learning p2\\EMG_trained.pth'))  #CHANGE
criterion = nn.CrossEntropyLoss()

# Evaluate Model on Test Data Set (validate model on test set)
with torch.no_grad():  # Basically turn off back propogation
  y_eval = model.forward(X_test_tensor) # X_test are features from our test set, y_eval will be predictions
  loss = criterion(y_eval, y_test_tensor) # Find the loss or error

correct = 0
with torch.no_grad():
  for i, data in enumerate(X_test_tensor):
    y_val = model.forward(data)

    if y[i] == 0:
      x = "rest"
    elif y[i] == 1:
      x = 'contraction'
    else:
      x = '?'


    # Will tell us what type of signal our network thinks it is
    print(f'{i+1}.)  {str(y_val)} \t {y[i]} \t {y_val.argmax().item()}')

    # Correct or not
    if y_val.argmax().item() == y[i]:
      correct +=1

print(f'We got {correct} correct!')