import torch
from torch import nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gait_lstm import GaitLSTM

device = "cuda" if torch.cuda.is_available() else "cpu"

## Data processing
FEATURE = ['right_shoulder_ang_x', 'right_shoulder_ang_y', 'right_shoulder_ang_z',
           'right_shoulder_acc_x', 'right_shoulder_acc_y', 'right_shoulder_acc_z',
           'left_shoulder_ang_x', 'left_shoulder_ang_y', 'left_shoulder_ang_z',
           'left_shoulder_acc_x', 'left_shoulder_acc_y', 'left_shoulder_acc_z']

LABEL = ['right_q', 'left_q']

TRAINSET = 'data/walking_data_train.csv'
TESTSET  = 'data/walking_data_test.csv'

WINDOW = 500

csv_reader = pd.read_csv(TRAINSET, encoding='utf-8')
feature = pd.DataFrame(csv_reader, columns=FEATURE).to_numpy().astype(float)
label = pd.DataFrame(csv_reader, columns=LABEL).to_numpy().astype(float)

## Network
input_dim = len(FEATURE)
hidden_dim = 64
sequence_length = WINDOW   # 5 sec
num_classes = 2        
num_layers = 2

learning_rate = 0.0005
num_epochs = 100
batch_size = 256
        
model = GaitLSTM(input_dim, hidden_dim, num_layers, num_classes).to(device)

## Load model
model_state_dict = torch.load('model_qpos_state_dict.pt', map_location=device)
model.load_state_dict(model_state_dict)
model.to(device)

imu_data = np.zeros((1, 500, 12))

left_q = []
left_q_label = []
right_q = []
right_q_label = []

sim_tick = feature.shape[0]

with torch.no_grad():
    model.eval()
    
    for tick in range(sim_tick - WINDOW):
        
        imu_data = feature[tick:tick + WINDOW]
        imu_data = np.array(imu_data).astype(float)
        imu_data = torch.FloatTensor(imu_data).to(device)
        
        imu_data = imu_data.reshape(-1, imu_data.shape[0], imu_data.shape[1]).to(device)

        q_pred = model(imu_data)
        

        right_q.append(q_pred.cpu().numpy()[0][0])
        right_q_label.append(label[tick + WINDOW][0]) 
        left_q.append(q_pred.cpu().numpy()[0][1])
        left_q_label.append(label[tick + WINDOW][1])
        
fig = plt.figure(figsize=(10, 4))
plt.plot(right_q,  label="lstm output")
plt.plot(right_q_label, label="lstm label")
plt.legend()
plt.show()

fig = plt.figure(figsize=(10, 4))
plt.plot(left_q,  label="lstm output")
plt.plot(left_q_label, label="lstm label")
plt.legend()
plt.show()