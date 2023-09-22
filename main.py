#!/usr/bin/python3

import torch
import numpy as np
import matplotlib.pyplot as plt

import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from utils import sequence_data_generator as SeqDataGenerator
from gait_lstm import GaitLSTM

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: ", device)

## Data processing
FEATURE = ['right_shoulder_ang_x', 'right_shoulder_ang_y', 'right_shoulder_ang_z',
           'right_shoulder_acc_x', 'right_shoulder_acc_y', 'right_shoulder_acc_z',
           'left_shoulder_ang_x', 'left_shoulder_ang_y', 'left_shoulder_ang_z',
           'left_shoulder_acc_x', 'left_shoulder_acc_y', 'left_shoulder_acc_z']

LABEL = ['right_q', 'left_q']

TRAINSET = 'data/walking_data_train.csv'
TESTSET  = 'data/walking_data_test.csv'

WINDOW = 500

## Network parameters
input_dim = len(FEATURE)
hidden_dim = 64
sequence_length = WINDOW   # 5 sec
num_classes = 2       
num_layers = 2

learning_rate = 0.0005
num_epochs = 200
batch_size = 256

# Datasets
train_loader, test_loader = SeqDataGenerator.tensor_dataset_generator(TRAINSET, TESTSET, FEATURE, LABEL, WINDOW, batch_size)

# LSTM model
model = GaitLSTM(input_dim, hidden_dim, num_layers, num_classes).to(device)

# Loss and optimizer
criterion = nn.MSELoss() 
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
train_hist = np.zeros(num_epochs)

best_loss = 10 ** 9 
patience_limit = 10 
patience_check = 0 

print("-------------------------------------------------")
print("----------------------Train----------------------")
print("-------------------------------------------------")
for epoch in range(num_epochs):
        avg_cost = 0
        val_loss = 0
        # torch.autograd.set_detect_anomaly(True)
        
        for i, (x, labels) in enumerate(train_loader):
                x = x.reshape(-1, sequence_length, input_dim).to(device)
                labels= labels.float().to(device)
                # Forward pass
                outputs = model(x)
                
                loss = criterion(outputs, labels)
                
                # Backward and optimize_
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
right_q_label_hist = []
left_q_label_hist = []
right_q_output_hist = []
left_q_output_hist = []

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
                
                right_q_label_hist.append(labels.cpu().numpy()[0][0])
                right_q_output_hist.append(outputs.cpu().numpy()[0][0])
                
                left_q_label_hist.append(labels.cpu().numpy()[0][1])
                left_q_output_hist.append(outputs.cpu().numpy()[0][1])

fig = plt.figure(figsize=(10, 4))
plt.plot(right_q_label_hist,  label="Ground truth Joint pos")
plt.plot(right_q_output_hist, label="lstm output Joint pos")
plt.legend()
plt.show()

fig = plt.figure(figsize=(10, 4))
plt.plot(left_q_label_hist,  label="Ground truth Joint pos")
plt.plot(left_q_output_hist, label="lstm output Joint pos")
plt.legend()
plt.show()
                
#Save the model checkpoint
torch.save(model, 'model_qpos.pt')
torch.save(model.state_dict(), 'model_qpos_state_dict.pt')
        
