def tensor_dataset_generator(train_file_name, test_file_name, FEATURE, LABEL, WINDOW, batch_size):
    import numpy as np
    import torch
    from torch.utils.data import TensorDataset 
    from torch.utils.data import DataLoader 
    
    X_train, Y_train = dataset_generator(train_file_name, FEATURE, LABEL, WINDOW)
    X_test, Y_test = dataset_generator(test_file_name, FEATURE, LABEL, WINDOW)
    
    X_train_tensor = torch.FloatTensor(X_train)
    X_test_tensor = torch.FloatTensor(X_test)
    Y_train_tensor = torch.FloatTensor(Y_train)
    Y_test_tensor = torch.FloatTensor(Y_test)

    # print("check train nan data", np.isnan(X_train_tensor).any(), # Check NaN 
    #     "check train inf data", np.isinf(X_train_tensor).any())
    # print("check testnan data", np.isnan(X_test_tensor).any(), 
    #     "check test inf data", np.isinf(X_test_tensor).any())

    # mean = X_train_tensor.mean()  # Train data normalization
    # std = X_train_tensor.std()
    # X_train_tensor = (X_train_tensor - mean) / std

    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    train_loader = DataLoader(train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            drop_last=False)

    test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
    test_loader = DataLoader(test_dataset,
                            batch_size=1,
                            shuffle=False,
                            drop_last=False)

    return train_loader, test_loader
    
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
