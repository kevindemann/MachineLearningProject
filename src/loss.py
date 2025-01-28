
def cross_entropy_loss(params, x_input, y_labels, lamba_lasso = 0, lambda_ridge = 0):
    y_probs = forward_propagation(params, x_input)
    log_probs = jnp.log(y_probs) 
    one_hot_labels = nn.one_hot(y_labels, y_probs.shape[-1])  # Convert to one-hot encoding, and use the y_probes dims as the num of classes
    l = -jnp.mean(jnp.sum(one_hot_labels * log_probs, axis=1))
    
    if lamba_lasso != 0:
        l+= lamba_lasso*jnp.sum(jnp.array([jnp.sum(jnp.abs(params[f'w_{i}'])) for i in range(int(len(params) / 2))]))
    if lambda_ridge != 0:
        l+= lambda_ridge*jnp.sum(jnp.array([jnp.sum(params[f'w_{i}'] ** 2) for i in range(int(len(params) / 2))]))
        
    
    return l