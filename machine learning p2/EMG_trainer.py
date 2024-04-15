import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from EMGModel import EMGModel

df = pd.read_csv("C:\\Users\\Jakeeer\\git\\senior_project\\machine learning p2\\emg_compiled.csv")

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

model = EMGModel()
# Set the criterion of model to measure the error, how far off the predictions are from the data
criterion = nn.CrossEntropyLoss()

# Define hyperparameters
lr = 0.00000001  # Learning rate
alpha = 0.9  # Smoothing constant
eps = 1e-8  # Epsilon
weight_decay = 1e-4  # Weight decay (L2 regularization coefficient)
momentum = 0.9  # Momentum
centered = True  # Use centered RMSprop
# Choose Adam Optimizer, lr = learning rate (if error doesn't go down after a bunch of iterations (epochs), lower our learning rate)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00000001)
#optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=momentum, centered=centered)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.0000001, momentum=0.82, weight_decay=1e-4)


# Train our model!
# Epochs? (one run thru all the training data in our network)
epochs = 1500
losses = []
for i in range(epochs):
  # Go forward and get a prediction
  y_pred = model.forward(X_train) # Get predicted results

  # Measure the loss/error, gonna be high at first
  loss = criterion(y_pred, y_train) # predicted values vs the y_train

  # Keep Track of our losses
  losses.append(loss.detach().numpy())

  # print every 10 epoch
  if i % 10 == 0:
    print(f'Epoch: {i} and loss: {loss}')

  # Do some back propagation: take the error rate of forward propagation and feed it back
  # thru the network to fine tune the weights
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

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

# Evaluate Model on Test Data Set (validate model on test set)
with torch.no_grad():  # Basically turn off back propogation
  y_eval = model.forward(X_test_tensor) # X_test are features from our test set, y_eval will be predictions
  loss = criterion(y_eval, y_test_tensor) # Find the loss or error

correct = 0
with torch.no_grad():
  for i, data in enumerate(X_test_tensor):
    y_val = model.forward(data)

    if y_test[i] == 0:
      x = "rest"
    elif y_test[i] == 1:
      x = 'contraction'
    else:
      x = '?'


    # Will tell us what type of signal our network thinks it is
    print(f'{i+1}.)  {str(y_val)} \t {y_test[i]} \t {y_val.argmax().item()}')

    # Correct or not
    if y_val.argmax().item() == y_test[i]:
      correct +=1

print(f'We got {correct} correct!')
'''


# Save the trained model
torch.save(model.state_dict(), 'C:\\Users\\Jakeeer\\git\\senior_project\\machine learning p2\\EMG_trained.pth') #CHANGE