import jax.numpy as jnp

def gd_parameter_update(param_grad, params, alpha):
   
    #updating parameters
    updated_params = {}
    for param in params.keys():
        updated_params[param] = params[param] - alpha*param_grad[param]
    
    return updated_params

def adam_parameter_update(param_grad, params, m, v, t, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    
    updated_params = {}
    new_m = {}
    new_v = {}

    for param in params.keys():
        new_m[param] = beta1 * m[param] + (1 - beta1) * param_grad[param]
        
        new_v[param] = beta2 * v[param] + (1 - beta2) * jnp.square(param_grad[param])
        
        m_hat = new_m[param] / (1 - beta1 ** t)
        v_hat = new_v[param] / (1 - beta2 ** t)
        
        updated_params[param] = params[param] - alpha * m_hat / (jnp.sqrt(v_hat) + eps)

    return updated_params, new_m, new_v
