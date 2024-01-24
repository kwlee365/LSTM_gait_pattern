#!/usr/bin/python3

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset 
from torch.utils.data import DataLoader 
from utils import sequence_data_generator as SeqDataGenerator
from gait_lstm import GaitLSTM

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: ", device)

## Data processing
FEATURE = ['right_back_ang_x', 'right_back_ang_y', 'right_back_ang_z',
           'right_back_acc_x', 'right_back_acc_y', 'right_back_acc_z',
           'left_back_ang_x', 'left_back_ang_y', 'left_back_ang_z',
           'left_back_acc_x', 'left_back_acc_y', 'left_back_acc_z']

LABEL = ['Label']

TRAINSET = ['data/back/KBack.csv',
            'data/back/DBack.csv',
            'data/back/JBack.csv',
            'data/back/CBack.csv']
TESTSET  = 'data/back/MBack.csv'

WINDOW = 100

## Network parameters
input_dim = len(FEATURE)
hidden_dim = 256
sequence_length = WINDOW   # 5 sec
num_classes = 100     
num_layers = 3

learning_rate = 0.001
momentum = 0.9
num_epochs = 1000
batch_size = 256

# Datasets

X_train1, Y_train1 = SeqDataGenerator.dataset_generator(TRAINSET[0], FEATURE, LABEL, WINDOW)
X_train2, Y_train2 = SeqDataGenerator.dataset_generator(TRAINSET[1], FEATURE, LABEL, WINDOW)
X_train3, Y_train3 = SeqDataGenerator.dataset_generator(TRAINSET[2], FEATURE, LABEL, WINDOW)
X_train4, Y_train4 = SeqDataGenerator.dataset_generator(TRAINSET[3], FEATURE, LABEL, WINDOW)
X_train = np.concatenate((X_train1, X_train2, X_train3, X_train4), axis = 0)
Y_train = np.concatenate((Y_train1, Y_train2, Y_train3, Y_train4), axis = 0)

X_test, Y_test = SeqDataGenerator.dataset_generator(TESTSET, FEATURE, LABEL, WINDOW)

X_train_tensor = torch.FloatTensor(X_train)
X_test_tensor = torch.FloatTensor(X_test)
Y_train_tensor = torch.LongTensor(Y_train).squeeze(dim=-1)
Y_test_tensor = torch.LongTensor(Y_test).squeeze(dim=-1)

# Y_train_tensor = F.one_hot(Y_train_tensor.to(torch.int64), num_classes=100).squeeze(dim=1)
# Y_test_tensor = F.one_hot(Y_test_tensor.to(torch.int64), num_classes=100).squeeze(dim=1)

train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
train_loader = DataLoader(train_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        drop_last=False)

test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
test_loader = DataLoader(test_dataset,
                        batch_size=1,
                        shuffle=False,
                        drop_last=False)

# LSTM model
model = GaitLSTM(input_dim, hidden_dim, num_layers, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
train_hist = np.zeros(num_epochs)
mse_hist = np.zeros(num_epochs)

best_loss = 10 ** 9 
patience_limit = 10 
patience_check = 0 

print("-------------------------------------------------")
print("----------------------Train----------------------")
print("-------------------------------------------------")
print("Remember! This Network is 'back_model.pt'.!!")
for epoch in range(num_epochs):
        avg_cost = 0
        val_loss = 0
        # torch.autograd.set_detect_anomaly(True)
        
        for i, (x, labels) in enumerate(train_loader):
                x = x.reshape(-1, sequence_length, input_dim).to(device)
                labels= labels.type(torch.LongTensor).to(device)
                # Forward pass
                outputs = model(x)
                
                loss = criterion(outputs, labels)
                
                # backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                avg_cost += loss/total_step
                val_loss += loss.item()
                
                if (i+1) % 100 == 0: 
                        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.8f}'
                              .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
                     
                
        if val_loss > best_loss: 
                patience_check += 1

                if patience_check >= patience_limit:
                        print("-------------Early Stopping is activated-------------") 
                        break
        else:
                best_loss = val_loss
                patience_check = 0
                        
        train_hist[epoch] = avg_cost
        
fig = plt.figure(figsize=(10, 4))
plt.plot(train_hist, label="Training loss")
plt.legend()
plt.show()

# Test the model
model.eval()
total_test = len(test_loader)
percent_gait_label_hist = []
percent_gait_output_hist = []

print("------------------------------------------------")
print("----------------------Test----------------------")
print("------------------------------------------------")
with torch.no_grad():
        correct = 0
        total = 0
        
        for x, labels in test_loader:
                x = x.reshape(-1, sequence_length, input_dim).to(device)
                labels= labels.float().to(device)
                
                outputs = model(x)
                total += labels.size(0)
                
                label = labels.cpu().numpy()   
                softmax = nn.Softmax(dim=0)
                output = softmax(outputs.squeeze(dim=0))
                output = torch.argmax(output).cpu().numpy()
                
                percent_gait_label_hist.append(label)
                percent_gait_output_hist.append(output)

fig = plt.figure(figsize=(10, 4))
plt.plot(percent_gait_label_hist,  label="Ground truth % gait")
plt.plot(percent_gait_output_hist, label="lstm output % gait")
plt.legend()
plt.show()
                
#Save the model checkpoint
# torch.save(model, 'model.pt')
torch.save(model.state_dict(), 'back_model.pt')
        
