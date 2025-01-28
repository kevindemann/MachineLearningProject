def gd_parameter_update(param_grad, params, alpha):
   
    #updating parameters
    updated_params = {}
    for param in params.keys():
        updated_params[param] = params[param] - alpha*param_grad[param]
    
    return updated_params