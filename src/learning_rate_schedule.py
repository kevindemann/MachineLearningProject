import numpy as np
import mlp

def update_weights_adam(W1, b1, W2, b2, dW1, db1, dW2, db2, m, v, beta1, beta2, epsilon, t, learning_rate):
    # Momentum update (m)
    m["W1"] = beta1 * m["W1"] + (1 - beta1) * dW1
    m["b1"] = beta1 * m["b1"] + (1 - beta1) * db1
    m["W2"] = beta1 * m["W2"] + (1 - beta1) * dW2
    m["b2"] = beta1 * m["b2"] + (1 - beta1) * db2

    # Variance update (v)
    v["W1"] = beta2 * v["W1"] + (1 - beta2) * (dW1 ** 2)
    v["b1"] = beta2 * v["b1"] + (1 - beta2) * (db1 ** 2)
    v["W2"] = beta2 * v["W2"] + (1 - beta2) * (dW2 ** 2)
    v["b2"] = beta2 * v["b2"] + (1 - beta2) * (db2 ** 2)

    # Bias correction
    m_corrected = {key: value / (1 - beta1 ** t) for key, value in m.items()}
    v_corrected = {key: value / (1 - beta2 ** t) for key, value in v.items()}

    # Update weights
    W1 -= learning_rate * m_corrected["W1"] / (np.sqrt(v_corrected["W1"]) + epsilon)
    b1 -= learning_rate * m_corrected["b1"] / (np.sqrt(v_corrected["b1"]) + epsilon)
    W2 -= learning_rate * m_corrected["W2"] / (np.sqrt(v_corrected["W2"]) + epsilon)
    b2 -= learning_rate * m_corrected["b2"] / (np.sqrt(v_corrected["b2"]) + epsilon)

    return W1, b1, W2, b2


def train_with_adam_early_stopping(
    X_train, y_train, X_val, y_val, input_size, hidden_size, output_size,
    learning_rate, epochs, beta1=0.9, beta2=0.999, epsilon=1e-8,
    patience=10, lambda_l2=0.01
):
    W1, b1, W2, b2 = mlp.initialize_weights(input_size, hidden_size, output_size)

    # Inizializing parameters
    m = {"W1": np.zeros_like(W1), "b1": np.zeros_like(b1), "W2": np.zeros_like(W2), "b2": np.zeros_like(b2)}
    v = {"W1": np.zeros_like(W1), "b1": np.zeros_like(b1), "W2": np.zeros_like(W2), "b2": np.zeros_like(b2)}

    validation_accuracies = []
    best_accuracy = -np.inf
    best_weights = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        # Forward propagation
        A2, cache = mlp.forward_propagation(X_train, W1, b1, W2, b2)
        
        # Loss with L2 regularization
        loss = mlp.cross_entropy_loss_with_l2(y_train, A2, W1, W2, lambda_l2)
        
        # Backpropagation
        dW1, db1, dW2, db2 = mlp.backward_propagation(X_train, y_train, cache, W1, W2)

        # Adam weights update
        W1, b1, W2, b2 = update_weights_adam(W1, b1, W2, b2, dW1, db1, dW2, db2, m, v, beta1, beta2, epsilon, epoch, learning_rate)

        val_predictions = mlp.predict(X_val, W1, b1, W2, b2)
        val_accuracy = mlp.accuracy(y_val, val_predictions)
        validation_accuracies.append(val_accuracy)

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_weights = (W1.copy(), b1.copy(), W2.copy(), b2.copy())
            epochs_without_improvement = 0 
        else:
            epochs_without_improvement += 1

        if epoch % 100 == 0 or epoch == 1:
            print(f"Epoch {epoch}, Loss: {loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # Patience check
        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}. Best Validation Accuracy: {best_accuracy:.4f}")
            break

    # Use the best weights
    if best_weights:
        W1, b1, W2, b2 = best_weights

    return W1, b1, W2, b2, validation_accuracies

