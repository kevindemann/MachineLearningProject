from jax import random
import jax.numpy as jnp
import numpy as np
import pandas as pd

def schuffel_data(x, y, seed):
    key = random.PRNGKey(seed) 
    permutation = random.permutation(key, x.shape[0])
    return x[permutation], y[permutation]

def batch_generator(x_input, y_target, batch_size, schuffel = True, seed = 42):
    
    n_batches = int(len(y_target)/batch_size)
        
    #x_batches = jnp.zeros(shape=(n_batches, batch_size, x_input.shape[1]))
    #y_batches = jnp.zeros(shape=(n_batches, batch_size))
    
    if schuffel:     
        x_input, y_target = schuffel_data(x_input, y_target, seed)
    
    for i in range(n_batches):
        
        #x_batches = x_batches.at[i].set(x_input[i*batch_size:(i+1)*batch_size,:])
        #y_batches = y_batches.at[i].set(y_target[i*batch_size:(i+1)*batch_size])

        x_batch = jnp.array(x_input[i*batch_size:(i+1)*batch_size,:])
        y_batch = jnp.array(y_target[i*batch_size:(i+1)*batch_size])
    
        yield x_batch, y_batch
        
def standardize_data(x):

    
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    x_std = (x - mean)/(std + 0.0001)

    
    return x_std, mean, std

def data_split(x, y, split_coeff = 0.8):
    n_rows = x.shape[0]
    
    sc_size = int(split_coeff * n_rows)
    
    
    sc_x = x[:sc_size]
    sc_y = y[:sc_size]
    
    sc_inv_x = x[sc_size:]
    sc_inv_y = y[sc_size:]
    
    return sc_x, sc_y, sc_inv_x, sc_inv_y

def prepare_data(x_input, y_target, standerdize = True):


    x_train, y_train, x_test, y_test = data_split(x_input, y_target, 0.8)


    x_train_n, y_train_n, x_train_v, y_train_v = data_split(x_train, y_train, 0.8)


    if standerdize:
        
        #main traing data
        x_std_train, mean_train, std_train = standardize_data(x_train)
        x_train  = x_std_train
        x_test = (x_test - mean_train)/std_train
                
        #Validation
        x_std_train_n, mean_train_n, std_train_n = standardize_data(x_train_n)
        x_train_n = x_std_train_n
        x_train_v = (x_train_v - mean_train_n)/std_train_n

    return x_train, y_train, x_train_n, y_train_n, x_train_v, y_train_v, x_test, y_test
  
def mnist_data_load(percent = 0.1):
    

    
    train_df = pd.read_csv('../data/mnist_train.csv')
    test_df = pd.read_csv('../data/mnist_test.csv')
    
    all_data = pd.concat([train_df, test_df], ignore_index=True)
    
    n_points = len(all_data)
    
    cut_off_val = int(percent * n_points)
    
    x_df = all_data.drop('label', axis=1)
    y_df = all_data['label']
    
    x_input = x_df.to_numpy()
    y_target = y_df.to_numpy()
    
    
    
    return x_input[:cut_off_val], y_target[:cut_off_val]