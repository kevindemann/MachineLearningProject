import jax.numpy as jnp
from jax import random, nn
from jax import value_and_grad

import numpy as np 

from data_utilities import batch_generator
from optimizers  import gd_parameter_update


class ANN(object):
    def __init__(self, architecture):
        self.architecture = architecture
        self.params = None 
        
    def neuron_initialization(self, seed = 42, normalizer = 'He'):
    
        self.params = {}
        
        key = random.PRNGKey(seed)
        for i in range(len(self.architecture)-1):
            
            inputs = self.architecture[i]
            oputs = self.architecture[i+1]
            
            initializer = nn.initializers.he_normal() #type of nomrmalization to be specificietd so that we can 
            #initializer(subkey, (inputs, oputs), jnp.float32) 
            
            key, subkey = random.split(key)
            self.params[f'w_{i}'] = initializer(subkey, (inputs, oputs), jnp.float32)  #random.uniform(subkey, shape=(inputs, oputs), minval=-1, maxval= 1)     #Weights from neuron to neuron 
            key, subkey = random.split(key)
            self.params[f'b_{i}'] = initializer(subkey, (1, oputs), jnp.float32)  #random.uniform(subkey, shape=(1, oputs),  minval=-1, maxval= 1)           #Bais vecor for each layer
            
    def forward_propagation(self, x_input):
    
        # could this be removed if we utilize back propagation using jax??
        a = x_input
        
        n_layers = int(len(self.params)/2)
        
        for i in range(n_layers):
            w = self.params[f'w_{i}']
            b = self.params[f'b_{i}']
            a_input = a
            
            z = a_input @ w + b
        
            if i < n_layers - 1:
                a = nn.relu(z)           #general simple activation function is used
            else: #problem specific case, for classification we do softmax
                a = nn.softmax(z)
                
            
        return a
    
    def cross_entropy_loss(self, x_input, y_labels, lamba_lasso = 0, lambda_ridge = 0):
        y_probs = self.forward_propagation(self.params, x_input)
        log_probs = jnp.log(y_probs) 
        one_hot_labels = nn.one_hot(y_labels, y_probs.shape[-1])  # Convert to one-hot encoding, and use the y_probes dims as the num of classes
        l = -jnp.mean(jnp.sum(one_hot_labels * log_probs, axis=1))
        
        if lamba_lasso != 0:
            l+= lamba_lasso*jnp.sum(jnp.array([jnp.sum(jnp.abs(self.params[f'w_{i}'])) for i in range(int(len(self.params) / 2))]))
        if lambda_ridge != 0:
            l+= lambda_ridge*jnp.sum(jnp.array([jnp.sum(self.params[f'w_{i}'] ** 2) for i in range(int(len(self.params) / 2))]))
            
        
        return l
    
    
    

class Classifier(object):
    def __init__(self, x_input, y_input, hidden_architecture):
        
        self.x_input, self.y_input = x_input, y_input
        
        self.architecture = np.concatenate([x_input.shape[1], np.concatenate([hidden_architecture, len(set(y_input))])])
        
        self.model = ANN(self.architecture)
        
    
    def hyperparameter_settings(self):
        pass 
    
    
    def data_prep(self):
        pass
    
    
    def train(self, loss, x_input, y_target, batch_size = 25, epochs = 200, alpha = 0.01):
        history = {'loss_v_epoch': [], 'accuracy_v_epoch': []}
        for i in range(epochs): 
            j = 0
            for x_i_batch, y_i_batch in batch_generator(x_input, y_target, batch_size, schuffel = True, seed = 42): #we go over each batch
                loss_i, param_grad = value_and_grad(loss, argnums=0)(self.params, x_i_batch, y_i_batch)
                self.params = gd_parameter_update(param_grad, self.params, alpha)
                self.model.params = self.params #updating the paremeters in the base model (inheritence should be used here to simplifiy)

            history['loss_v_epoch'].append(loss_i)

            print(f'Epoch {i} -> loss: {loss_i}')

    
    def predict(self, x):
        y_probs = self.model.forward_propagation(x)
        return np.argmax(y_probs, axis=1)

    def accuracy(self, x, y):
        y_pred = self.predict(self.params, x)
        return np.mean(y == y_pred)
    
    
    def hyperparameter_tuning(self):
        pass
    
        
    def validation_testing_evalutaion(self): 
        pass
    
    def testing_evalutaion(self):
        pass    