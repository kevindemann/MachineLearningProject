import numpy as np
from sklearn.preprocessing import OneHotEncoder

def load_data(filepath):
    """
    Loads data from file and returns it as a NumPy array.
    """
    data = np.loadtxt(filepath)
    return data

def split_data(data, data_labels, num_classes, samples_per_class, train_samples_per_class):
    """
    Divides train set and test set.
    """
    train_data = []
    test_data = []
    
  
    
    for i in range(num_classes):
        class_data = data[i * samples_per_class:(i + 1) * samples_per_class]
        
        train_data.append(class_data[:train_samples_per_class])
        test_data.append(class_data[train_samples_per_class:])
     

    return np.vstack(train_data), np.vstack(test_data)

def create_labels(num_classes, train_samples_per_class, test_samples_per_class):
    """
    Creates labels for training and testing.
    """
    train_labels = np.repeat(np.arange(num_classes), train_samples_per_class)
    test_labels = np.repeat(np.arange(num_classes), test_samples_per_class)
    return train_labels, test_labels

def one_hot_encode(labels):
    """
    Returns the labels in one-hot encoding.
    """
    encoder = OneHotEncoder(sparse_output=False)
    return encoder.fit_transform(labels.reshape(-1, 1))


