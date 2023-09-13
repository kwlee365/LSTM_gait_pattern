def dataset_generator(file_name, FEATURE, LABEL, WINDOW):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    csv_reader = pd.read_csv(file_name, encoding='utf-8')
    
    feature = pd.DataFrame(csv_reader, columns=FEATURE).to_numpy().astype(float)
    label = pd.DataFrame(csv_reader, columns=LABEL).to_numpy().astype(float)

    feature, label = make_sequene_dataset(feature, label, WINDOW)
    
    return feature, label
    
def make_sequene_dataset(feature, label, window_size):
    
    import numpy as np
    
    feature_list = []      
    label_list = []       
    
    for i in range(len(feature)-window_size):

        feature_list.append(feature[i:i+window_size])
        label_list.append(label[i+window_size-1])

    return np.array(feature_list).astype(float), np.array(label_list).astype(float)
