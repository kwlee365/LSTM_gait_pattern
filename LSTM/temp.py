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
# %%
