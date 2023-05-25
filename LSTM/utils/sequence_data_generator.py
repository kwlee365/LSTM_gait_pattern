import numpy as np

def make_train_and_test_dataset(X, Y, a):

    split = int(len(X)* (1.0 - a))

    X_train = X[0:split]
    Y_train = Y[0:split]

    X_test = X[split:]
    Y_test = Y[split:]

    return X_train, Y_train, X_test, Y_test

def make_sequene_dataset(feature, label, window_size):
    
    import numpy as np
    
    feature_list = []      
    label_list = []       
    
    for i in range(len(feature)-window_size):

        feature_list.append(feature[i:i+window_size])
        label_list.append(label[i+window_size-1])

    return np.array(feature_list).astype(float), np.array(label_list).astype(float)
