import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from EMGModel import EMGModel

df = pd.read_csv("C:\\Users\\Jakeeer\\git\\senior_project\\machine learning p3\\emg_compiled.csv")

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

# Train Test Split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train = torch.FloatTensor(X) #CHANGE TO SPLIT/NOT SPLIT
y_train = torch.LongTensor(y) #CHANGE TO SPLIT/NOT SPLIT
X_train = X_train.unsqueeze(-1)  # Add an additional dimension at the end
X_train = X_train.permute(0, 2, 1)  # Permute dimensions to match expected shape
#print(X_train.size())

model = EMGModel()
# Set the criterion of model to measure the error, how far off the predictions are from the data
criterion = nn.CrossEntropyLoss()

# Choose Adam Optimizer, lr = learning rate (if error doesn't go down after a bunch of iterations (epochs), lower our learning rate)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.0000001, momentum=0.82, weight_decay=1e-4)


# Train our model!
# Epochs? (one run thru all the training data in our network)
epochs = 5000
losses = []
for epoch in range(epochs):
      # Forward pass: compute predicted outputs by passing inputs to the model
    outputs = model(X_train)
    
    # Calculate the loss
    loss = criterion(outputs, y_train)
    
    # Backward pass: compute gradient of the loss with respect to model parameters
    optimizer.zero_grad()
    loss.backward()
    
    # Perform a single optimization step (parameter update)
    optimizer.step()
    
    # Append the loss value to the list for visualization
    losses.append(loss.item())
    
    # Print the loss every 100 epochs
    if epoch % 10 == 0:
      print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Graph it out!
# Plot the loss values
plt.plot(range(epochs), losses)
plt.ylabel("Loss/Error")
plt.xlabel("Epoch")
plt.title("Training Loss")
plt.show()

#TESTING PORTION OF CODE

'''
#Testing data
X_test_tensor = torch.FloatTensor(X_test) #change
y_test_tensor = torch.LongTensor(y_test) #change
X_test_tensor = X_test_tensor.unsqueeze(-1)  # Add an additional dimension at the end
X_test_tensor = X_test_tensor.permute(0, 2, 1)  # Permute dimensions to match expected shape

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
'''


# Save the trained model
torch.save(model.state_dict(), 'C:\\Users\\Jakeeer\\git\\senior_project\\machine learning p3\\EMG_trained.pth') #CHANGE