#In[]
from utils import tensorflow_lstm_model as GaitLSTM 
from utils import sequence_data_generator as SeqDataGenerator 
import tensorflow as tf

damn = [0,1,2,3,4,5,6,7,8,9]
damn_label = tf.one_hot(damn, 10)
print("raw damn data")
print(damn)
print(damn_label)

damn, damn_label = SeqDataGenerator.make_sequene_dataset(damn, damn_label, 3)

print("processed damn data")
print(damn)
print(damn_label)

print("processed damn data size")
print(damn.shape)
print(damn_label.shape)
# In[]
import numpy as np

data = [i for i in range(500)]
print(data)

num = len(data) - 1
print(num)

for i in range(num):
    data[i] = data[i+1]

data[num] = 10
print(data)

concat_data = np.stack((np.array(data), np.array(data)), axis = 1)
concat_Data = np.reshape(concat_data, [1, concat_data.shape[0], concat_data.shape[1]])
# tensor = tf.convert_to_tensor(concat_data)

print(concat_Data.shape)

model = tf.keras.models.load_model('/home/kwan/LSTM_gait_pattern/LSTM/models/GaitLSTMModel.h5')
model.predict(concat_Data)

# %%
