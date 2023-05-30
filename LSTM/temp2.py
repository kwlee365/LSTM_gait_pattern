# In[] Start

from utils import sequence_data_generator as SeqDataGenerator
from utils import tensorflow_lstm_model as GaitLSTM
from tensorflow.python.client import device_lib
from sklearn.preprocessing import MinMaxScaler
import logging, os, time
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tf.get_logger().setLevel(logging.INFO)

DATA = ['Time', 'q', 'qdot',
        'pelv_gyro_x', 'pelv_gyro_y', 'pelv_gyro_z',
        'pelv_acc_x', 'pelv_acc_y', 'pelv_acc_z',
        'leg_gyro_x', 'leg_gyro_y', 'leg_gyro_z',
        'leg_acc_x', 'leg_acc_y', 'leg_acc_z']

FEATURE = ['leg_gyro_x', 'leg_gyro_y', 'leg_gyro_z',
        'leg_acc_x', 'leg_acc_y', 'leg_acc_z']

LABEL = ['Label']

csv_reader = pd.read_csv('datasets/walking_data.csv', encoding='utf-8')

plt.title('Gait pattern segmentation')
plt.xlabel('time')
plt.grid()
plt.xlim([0, 1000])
plt.plot(csv_reader['leg_gyro_z'])
plt.legend(('leg_gyro_z', '% gait'), loc='best')
plt.show()

plt.title('Gait pattern segmentation')
plt.xlabel('time')
plt.grid()
plt.xlim([0, 1000])
plt.plot(csv_reader['leg_gyro_z'])
plt.plot(csv_reader['Label']/100)
plt.legend(('leg_gyro_z', '% gait'), loc='best')
plt.show()

# %%
