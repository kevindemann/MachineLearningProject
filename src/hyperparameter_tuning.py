
import numpy as np
from sklearn.linear_model import Ridge, Lasso 

#basic for linear regression, everything else will follow
def choose_hyper_param(train, predict, X_train_n, y_train_n, X_train_v, y_train_v, is_ridge: bool):
    mse_arr = []
    lam_arr = []

    for pow_lam in range(-4, 3):
        lam = 10 ** pow_lam

        if is_ridge:
            W1, b1, W2, b2 = train() #L1
        else:
            #then lasso is done
            W1, b1, W2, b2 = train() #L1
            
        y_pred_v = predict(X_train_v, W1, b1, W2, b2)
        mse = (np.dot(y_pred_v - y_train_v, y_pred_v - y_train_v))/(2*len(y_train_v))
        
        mse_arr.append(mse) 
        lam_arr.append(lam)


    # get the index of the lambda value that has the minimal use
    lambda_idx_min = np.argmin(np.array(mse_arr))

    # return the optimal lambda value
    return lam_arr[lambda_idx_min]