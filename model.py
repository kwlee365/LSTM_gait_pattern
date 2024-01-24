import torch
from torch import nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gait_lstm import GaitLSTM
from data import MSDATA
import math

device = "cuda" if torch.cuda.is_available() else "cpu"

dsaver = MSDATA('save', 'shoulder_result_correction.txt')

## Data processing
FEATURE = ['right_shoulder_ang_x', 'right_shoulder_ang_y', 'right_shoulder_ang_z',
           'right_shoulder_acc_x', 'right_shoulder_acc_y', 'right_shoulder_acc_z',
           'left_shoulder_ang_x', 'left_shoulder_ang_y', 'left_shoulder_ang_z',
           'left_shoulder_acc_x', 'left_shoulder_acc_y', 'left_shoulder_acc_z']

# LABEL = ['right_q', 'left_q']
LABEL = ['Label']

TESTSET  = 'data/shoulder/MShoulder.csv'

WINDOW = 100

csv_reader = pd.read_csv(TESTSET, encoding='utf-8')
feature = pd.DataFrame(csv_reader, columns=FEATURE).to_numpy().astype(float)
label = pd.DataFrame(csv_reader, columns=LABEL).to_numpy().astype(float)

## Network parameters
input_dim = len(FEATURE)
hidden_dim = 256
sequence_length = WINDOW   # 1 sec
num_classes = 100     
num_layers = 3

learning_rate = 0.001
momentum = 0.9
num_epochs = 1000
batch_size = 256
        
model = GaitLSTM(input_dim, hidden_dim, num_layers, num_classes).to(device)

## Load model
model_state_dict = torch.load('shoulder_model.pt', map_location=device)
model.load_state_dict(model_state_dict)
model.to(device)

imu_data = np.zeros((1, sequence_length, 12))

percent_gait = []
percent_gait_label = []
error_list = []
square_error_list = []
square_error = 0

class_total = 0
class_correct = 0

sim_tick = feature.shape[0]
# sim_tick = 500
total_tick = sim_tick-WINDOW

with torch.no_grad():
    model.eval()
    
    for tick in range(total_tick):
        # print(tick)
        
        imu_data = feature[tick:tick + WINDOW]
        imu_data = np.array(imu_data).astype(float)
        imu_data = torch.FloatTensor(imu_data).to(device)
        
        imu_data = imu_data.reshape(-1, imu_data.shape[0], imu_data.shape[1]).to(device)

        outputs = model(imu_data)
        softmax = nn.Softmax(dim=0)
        output = softmax(outputs.squeeze(dim=0))
        output = torch.argmax(output).cpu().numpy()
        
        error = (output - label[tick + WINDOW][0])
        if(error > 50.0):
            error = abs(100 - error)
        elif(error < -50.0):
            error = abs(100 + error)
            
        error_list.append(error)
        square_error_list.append(error ** 2)
            
        square_error += error ** 2
        
        if (output == label[tick + WINDOW][0]):
            class_correct += 1
        class_total += 1
            
        percent_gait.append(output)
        percent_gait_label.append(label[tick + WINDOW][0])
        
        data = '%lf\t%lf\t%lf\t\n'%(output,
                                    label[tick + WINDOW][0],
                                    error)
        
        dsaver.save(data)
        
mse = square_error / total_tick
rmse = np.sqrt(mse)

# print('percent_gait')
# print(percent_gait)
# print('percent_gait_label')
# print(percent_gait_label)


# print('error list')
# print(error_list)
# print('square error list')
# print(square_error_list)

print('shoulder')
print(rmse)
# print(class_correct / class_total * 100)
        
fig = plt.figure(figsize=(10, 4))
plt.plot(percent_gait,  label="lstm output")
plt.plot(percent_gait_label, label="lstm label")
plt.legend()
plt.show()