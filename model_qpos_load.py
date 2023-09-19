import torch
from torch import nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data import MSDATA

device = "cuda" if torch.cuda.is_available() else "cpu"

## Data processing
FEATURE = ['right_acc_x', 'right_acc_y', 'right_acc_z',
            'right_ang_x', 'right_ang_y', 'right_ang_z',
            'left_acc_x',  'left_acc_y',  'left_acc_z',
            'left_ang_x',  'left_ang_y',  'left_ang_z']
LABEL = ['q']

TRAINSET = ['data/train_walking_data_exp1.csv', 
            'data/train_walking_data_exp2.csv', 
            'data/train_walking_data_exp3.csv']
TESTSET  = 'data/test_walking_data.csv'

WINDOW = 500

csv_reader = pd.read_csv(TRAINSET[0], encoding='utf-8')
    
feature = pd.DataFrame(csv_reader, columns=FEATURE).to_numpy().astype(float)
label = pd.DataFrame(csv_reader, columns=LABEL).to_numpy().astype(float)

## Network
input_dim = len(FEATURE)
hidden_dim = 64
sequence_length = WINDOW   # 5 sec
num_classes = 1        
num_layers = 2

learning_rate = 0.0005
num_epochs = 100
batch_size = 256

class GaitLSTM(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
                super(GaitLSTM, self).__init__()
                self.input_dim  = input_dim
                self.hidden_dim = hidden_dim
                self.num_classes = num_classes
                self.num_layers = num_layers
                
                self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers = num_layers, dropout=0.2, batch_first = True)
                self.tanh = nn.Tanh()
                self.fc = nn.Linear(hidden_dim, num_classes)
                # Do I have to consider batchnormalization? #
                                                
        def forward(self, x):
                # Set initial hidden and cell states
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device) 
                c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)

                # Decode the hidden state of the last time step.
                out, _ = self.lstm(x, (h0, c0))
                out = self.tanh(out[:, -1 ,:])
                out = self.fc(out)
                # out = F.softmax(out, dim=1)   
                
                return out
        
model = GaitLSTM(input_dim, hidden_dim, num_layers, num_classes).to(device)

## Load model
model_state_dict = torch.load('model_qpos.ckpt', map_location=device)
model.load_state_dict(model_state_dict)

imu_data = np.zeros((1, 500, 12))

q = []
q_label = []

sim_tick = feature.shape[0]

dsaver = MSDATA('save', 'plotData.txt')


with torch.no_grad():
    model.eval()
    
    for tick in range(sim_tick - WINDOW):
        
        imu_data = feature[tick:tick + WINDOW]
        imu_data = np.array(imu_data).astype(float)
        imu_data = torch.FloatTensor(imu_data).to(device)
        
        imu_data = imu_data.reshape(-1, imu_data.shape[0], imu_data.shape[1]).to(device)

        q_pred = model(imu_data)
        
        q.append(q_pred.item())
        q_label.append(label[tick + WINDOW])
        
        data = '%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n'%(q_pred.item(),
                                                 label[tick + WINDOW],
                                                 feature[tick + WINDOW, 0],
                                                 feature[tick + WINDOW, 1],
                                                 feature[tick + WINDOW, 2],
                                                 feature[tick + WINDOW, 3],
                                                 feature[tick + WINDOW, 4],
                                                 feature[tick + WINDOW, 5])
        dsaver.save(data)
        
fig = plt.figure(figsize=(10, 4))
plt.plot(q,  label="lstm output")
plt.plot(q_label, label="lstm label")
plt.legend()
plt.show()
