#!/usr/bin/python3

import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import sequence_data_generator as SeqDataGenerator

import torch.nn.functional as F
from torch.utils.data import TensorDataset 
from torch.utils.data import DataLoader 
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: ", device)

## Data processing
FEATURE = ['right_acc_x', 'right_acc_y', 'right_acc_z',
            'right_ang_x', 'right_ang_y', 'right_ang_z',
            'left_acc_x',  'left_acc_y',  'left_acc_z',
            'left_ang_x',  'left_ang_y',  'left_ang_z']
LABEL = ['label']

TRAINSET = ['data/train_walking_data_exp1.csv', 
            'data/train_walking_data_exp2.csv', 
            'data/train_walking_data_exp3.csv']
TESTSET  = 'data/test_walking_data.csv'

WINDOW = 500

X_train1, Y_train1 = SeqDataGenerator.dataset_generator(TRAINSET[0], FEATURE, LABEL, WINDOW)
X_train2, Y_train2 = SeqDataGenerator.dataset_generator(TRAINSET[1], FEATURE, LABEL, WINDOW)
X_train3, Y_train3 = SeqDataGenerator.dataset_generator(TRAINSET[2], FEATURE, LABEL, WINDOW)

X_train = np.concatenate((X_train1, X_train2, X_train3), axis = 0)
Y_train = np.concatenate((Y_train1, Y_train2, Y_train3), axis = 0)
X_test, Y_test = SeqDataGenerator.dataset_generator(TESTSET, FEATURE, LABEL, WINDOW)

X_train_tensor = torch.FloatTensor(X_train)
X_test_tensor = torch.FloatTensor(X_test)
Y_train_tensor = torch.FloatTensor(Y_train)
Y_test_tensor = torch.FloatTensor(Y_test)

## Network
input_dim = len(FEATURE)
hidden_dim = 64
sequence_length = WINDOW   # 5 sec
num_classes = 100        
num_layers = 2

learning_rate = 0.0005
num_epochs = 200
batch_size = 256

Y_train_label = F.one_hot(Y_train_tensor.to(torch.int64), num_classes=num_classes).reshape(-1, num_classes)
Y_test_label = F.one_hot(Y_test_tensor.to(torch.int64), num_classes=num_classes).reshape(-1, num_classes)

train_dataset = TensorDataset(X_train_tensor, Y_train_label)
train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          drop_last=False)

test_dataset = TensorDataset(X_test_tensor, Y_test_label)
test_loader = DataLoader(test_dataset,
                         batch_size=1,
                         shuffle=False,
                         drop_last=False)
                         
class GaitLSTM(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
                super(GaitLSTM, self).__init__()
                self.input_dim  = input_dim
                self.hidden_dim = hidden_dim
                self.num_classes = num_classes
                self.num_layers = num_layers
                
                self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers = num_layers, dropout = 0.2, batch_first = True)
                self.tanh = nn.Tanh()
                self.fc = nn.Linear(hidden_dim, num_classes)
                                                
        def forward(self, x):
                # Set initial hidden and cell states
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device) 
                c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)

                # Decode the hidden state of the last time step.
                out, _ = self.lstm(x, (h0, c0))
                self.tanh = nn.Tanh()
                out = self.fc(out[:, -1 ,:])
                
                return out
        
model = GaitLSTM(input_dim, hidden_dim, num_layers, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
train_hist = np.zeros(num_epochs)

best_loss = 10 ** 9 
patience_limit = 3 
patience_check = 0 

print("-------------------------------------------------")
print("----------------------Train----------------------")
print("-------------------------------------------------")
for epoch in range(num_epochs):
        avg_cost = 0
        val_loss = 0
        
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
                        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
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
label_hist = []
output_hist = []

print("------------------------------------------------")
print("----------------------Test----------------------")
print("------------------------------------------------")
with torch.no_grad():
        correct = 0
        total = 0
        
        for x, labels in test_loader:
                x = x.reshape(-1, sequence_length, input_dim).to(device)
                labels= labels.reshape(-1, num_classes).float().to(device)
                
                outputs = model(x)
                total += labels.size(0)
                
                print("labels: ", torch.argmax(labels, dim=1), "output: ", torch.argmax(outputs, dim=1))
                
                correct += (torch.argmax(outputs, dim=1) == torch.argmax(labels, dim=1)).sum().item()
                
                if(i==1):
                        print(torch.argmax(labels, dim=1).item())
                
                label_hist.append(torch.argmax(labels, dim=1).item())
                output_hist.append(torch.argmax(outputs, dim=1).item())
                
        print('total: ', total)
        print('correct: ', correct)
        print('Test accuracy of the model on the 10000 test: {} %'.format(100 * correct / total))

fig = plt.figure(figsize=(10, 4))
plt.plot(label_hist,  label="label % gait")
plt.plot(output_hist, label="lstm output")
plt.legend()
plt.show()
                
#Save the model checkpoint
# torch.save(model.state_dict(), 'model.ckpt')
        
