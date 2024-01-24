import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GaitLSTM(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
                super(GaitLSTM, self).__init__()
                self.input_dim  = input_dim
                self.hidden_dim = hidden_dim
                self.num_classes = num_classes
                self.num_layers = num_layers
                
                self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers = num_layers, dropout=0.1, batch_first = True)
                self.tanh = nn.Tanh()
                self.fc = nn.Linear(hidden_dim, num_classes)
          
        def forward(self, x):
                # Set initial hidden and cell states
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device) 
                c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)

                # Decode the hidden state of the last time step.
                out, _ = self.lstm(x, (h0, c0))
                out = self.tanh(out[:, -1 ,:])
                out = self.fc(out)
                
                return out