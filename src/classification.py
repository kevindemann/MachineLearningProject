import jax.numpy as jnp
from jax import random, nn
from jax import value_and_grad
from sklearn.model_selection import KFold
import itertools

import numpy as np 

from data_utilities import batch_generator, prepare_data, schuffel_data
from optimizers  import gd_parameter_update, adam_parameter_update


class ANN(object):
    def __init__(self, architecture):
        self.architecture = architecture
        #self.params = None 
        
    def neuron_initialization(self, seed = 42, normalizer = 'He'):
    
        params = {}
        
        key = random.PRNGKey(seed)
        for i in range(len(self.architecture)-1):
            
            inputs = self.architecture[i]
            oputs = self.architecture[i+1]
            
            if normalizer == 'He':
                
                initializer = nn.initializers.he_normal() #type of nomrmalization to be specificietd so that we can 
            #initializer(subkey, (inputs, oputs), jnp.float32) 
            else:
                raise ValueError(f"initializers: '{normalizer}'. Not supported")
            
            key, subkey = random.split(key)
            params[f'w_{i}'] = initializer(subkey, (inputs, oputs), jnp.float32)  #random.uniform(subkey, shape=(inputs, oputs), minval=-1, maxval= 1)     #Weights from neuron to neuron 
            key, subkey = random.split(key)
            params[f'b_{i}'] = initializer(subkey, (1, oputs), jnp.float32)  #random.uniform(subkey, shape=(1, oputs),  minval=-1, maxval= 1)           #Bais vecor for each layer
        
        return params
    
    def forward_propagation(self, params, x_input):
    
        # could this be removed if we utilize back propagation using jax??
        a = x_input
        
        n_layers = int(len(params)/2)
        
        for i in range(n_layers):
            w = params[f'w_{i}']
            b = params[f'b_{i}']
            a_input = a
            
            z = a_input @ w + b
        
            if i < n_layers - 1:
                a = nn.relu(z)           #general simple activation function is used
            else: #problem specific case, for classification we do softmax
                a = nn.softmax(z)
                
            
        return a
    
    def cross_entropy_loss(self, params,  x_input, y_labels, lamba_lasso = 0, lambda_ridge = 0):
        y_probs = self.forward_propagation(params, x_input)
        log_probs = jnp.log(y_probs) 
        one_hot_labels = nn.one_hot(y_labels, y_probs.shape[-1])  # Convert to one-hot encoding, and use the y_probes dims as the num of classes
        l = -jnp.mean(jnp.sum(one_hot_labels * log_probs, axis=1))
        
        if lamba_lasso != 0:
            l+= lamba_lasso*jnp.sum(jnp.array([jnp.sum(jnp.abs(params[f'w_{i}'])) for i in range(int(len(params) / 2))]))
        if lambda_ridge != 0:
            l+= lambda_ridge*jnp.sum(jnp.array([jnp.sum(params[f'w_{i}'] ** 2) for i in range(int(len(params) / 2))]))
            
        
        return l
    
    
    

class Classifier(object):
    def __init__(self, x_train, y_train, x_test, y_test, hidden_architecture, seed = 42):
        
        #self.x_input, self.y_input = x_input, y_input
       
        self.architecture = [x_train.shape[1]] + hidden_architecture + [len(set(y_train))]
        
        self.nn = ANN(self.architecture)
        
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

        #x_train_n, y_train_n = None, None
        #x_train_v, y_train_v = None, None
        self.seed = seed

    
    def data_prep(self, test_split_coef = 0.8, schuffel = True, standardization = False):
        #split into test/train/val
        if schuffel:
            self.x_input, self.y_input = schuffel_data(self.x_input, self.y_input, self.seed)
            
        self.x_train, self.y_train, self.x_test, self.y_test = prepare_data(self.x_input, self.y_input, test_split_coef, standardization)
        
    
    def data_schuffel(self):
        self.x_train, self.y_train = schuffel_data(self.x_train, self.y_train, self.seed)


    def train(self, data = None, params = None, loss = 'cross_entropy', loss_params = {'lambda_lasso': 0, 'lambda_ridge': 0}, 
              optimizer = 'gd', optimizer_params = {'alpha': 0.001}, batch_size = 50, epochs = 200, epoch_rate = 10, seed = 42):
        
        history = {'loss_train': np.zeros(epochs), 'accuracy_train': np.zeros(epochs), 'loss_test': np.zeros(epochs), 'accuracy_test': np.zeros(epochs)}
        
        if not params:
            params = self.nn.neuron_initialization(seed)
            
        if not data: 
            x = self.x_train
            y = self.y_train     
            
            x_test = self.x_test
            y_test = self.y_test  
        else:
            x, y, x_test ,y_test = data
            
            
        if optimizer == "adam":
            m = {key: jnp.zeros_like(value) for key, value in params.items()}
            v = {key: jnp.zeros_like(value) for key, value in params.items()}
            t = 1
            
            
            
        for i in range(1, epochs+1): 
            
            
            for x_i_batch, y_i_batch in batch_generator(x, y, batch_size, schuffel = True, seed = seed): #we go over each batch
            
                if loss == 'cross_entropy':
                    loss_i, param_grad = value_and_grad(self.nn.cross_entropy_loss, argnums=0)(params, x_i_batch, y_i_batch, loss_params['lambda_lasso'], loss_params['lambda_ridge'])
                else:
                    raise ValueError(f"Loss: '{loss}'. Not supported")
                
                if optimizer == 'gd':    
                    params = gd_parameter_update(param_grad, params, optimizer_params['alpha'])
                elif optimizer == 'adam':
                    params, m, v = adam_parameter_update(param_grad, params, m, v, t, optimizer_params['alpha']) 
                                                        # optimizer_params['b_1'], optimizer_params['b_2'], optimizer_params['eps']) in case more are added
                    t += 1
                else:
                    raise ValueError(f"Optimizer: '{optimizer}'. Not supported")
                
                #self.nn.params = self.params #updating the paremeters in the base model (inheritence should be used here to simplifiy)
            history['loss_train'][i-1] = loss_i
            history['accuracy_train'][i-1] = self.accuracy(params, x, y)
            
            if loss == 'cross_entropy':
                test_loss_i = self.nn.cross_entropy_loss( params,  x_test, y_test, loss_params['lambda_lasso'], loss_params['lambda_ridge'])
                history['loss_test'][i-1] = test_loss_i
            history['accuracy_test'][i-1] = self.accuracy(params, x_test, y_test)
            
            
            if i % epoch_rate == 0:
                
                print(f'Epoch {i}')
                print(f'Train Loss: {loss_i:.4f} |Train Accuracy: {self.accuracy(params, x, y):.4f}')
                print(f'Test Loss: {test_loss_i:.4f} |Test Accuracy: {self.accuracy(params,x_test, y_test):.4f}')
                print('--------------------------------------------------------------------')
            
        return params, history 
        
  
    def predict(self, params, x):
        y_probs = self.nn.forward_propagation(params, x)
        return np.argmax(y_probs, axis=1)

    def accuracy(self, params, x, y):
        y_pred = self.predict(params, x)
        return np.mean(y == y_pred)
    
    
    def hyperparameter_search(self, k_folds, loss = 'cross_entropy', optimizer = 'gd',
                              search_params = {'alpha':[0.001],'lasso':[0, 0.001],'ridge':[0.01] , 'batch':[128, 256]}, epochs = 200):
        
        alphas = search_params['alpha']
        lasso_values = search_params['lasso']
        ridge_values = search_params['ridge']
        batch_sizes = search_params['batch']

        best_acc = 0
        best_params = {}

        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

        for alpha, lambda_lasso, lambda_ridge, batch_size in itertools.product(alphas, lasso_values, ridge_values, batch_sizes):
            print(f"\nTesting con: alpha={alpha}, Lasso={lambda_lasso}, Ridge={lambda_ridge}, batch_size={batch_size}")

            fold_accuracies = []
            
            loss_params = {'lambda_lasso': lambda_lasso, 'lambda_ridge': lambda_ridge}

            optimizer_params = {'alpha': alpha}
            
            for train_index, val_index in kf.split(self.x_train):
                train_data, val_data = self.x_train[train_index], self.x_train[val_index]
                train_labels, val_labels = self.y_train[train_index], self.y_train[val_index]
                
                val_train_data = (train_data, train_labels, val_data, val_labels)
       
                trained_params, _ = self.train(
                                                    data=val_train_data, params = None, loss = loss, loss_params= loss_params, optimizer = 'adam',
                                                    optimizer_params = optimizer_params, batch_size= batch_size, epochs=epochs
                                                    ) 
         
                val_acc = self.accuracy(trained_params, val_data, val_labels)
                fold_accuracies.append(val_acc)

            avg_acc = jnp.mean(jnp.array(fold_accuracies))
            print(f"Average Validation Accuracy: {avg_acc:.4f}")

            if avg_acc > best_acc:
                best_acc = avg_acc
                best_params = {'alpha': alpha, 'lambda_lasso': lambda_lasso, 'lambda_ridge': lambda_ridge, 'batch_size': batch_size}

        print("\nBest combination:")
        print(f"alpha={best_params['alpha']}, lambda_lasso={best_params['lambda_lasso']}, lambda_ridge={best_params['lambda_ridge']}, batch_size={best_params['batch_size']}")
        print(f"Best Average Accuracy: {best_acc:.4f}")

        best_optimization_params = {'alpha': best_params['alpha']}
        best_loss_params = {'lamba_lasso': best_params['lambda_lasso'], 'lambda_ridge': best_params['lambda_ridge']}
        best_batch_size = best_params['batch_size']
        return best_optimization_params, best_loss_params, best_batch_size


    