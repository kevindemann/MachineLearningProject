import preprocessing
import mlp
import learning_rate_schedule
import time
import matplotlib.pyplot as plt

dataset_path = "small/mfeat-pix"
data = preprocessing.load_data(dataset_path)

num_classes = 10
samples_per_class = 200
train_samples_per_class = 100
test_samples_per_class = 100

train_data, test_data = preprocessing.split_data(data, num_classes, samples_per_class, train_samples_per_class)

train_labels, test_labels = preprocessing.create_labels(num_classes, train_samples_per_class, test_samples_per_class)

train_labels_one_hot = preprocessing.one_hot_encode(train_labels)
test_labels_one_hot = preprocessing.one_hot_encode(test_labels)

input_size = train_data.shape[1]
hidden_size = 64
output_size = num_classes
initial_learning_rate = 0.01
epochs = 2000

# Training with standard learning rate
print("Beginning standard training...")
start_time = time.time()
W1, b1, W2, b2, validation_accuracies = mlp.train_with_validation(
    train_data, train_labels_one_hot, test_data, test_labels_one_hot,
    input_size, hidden_size, output_size, initial_learning_rate, epochs
)
end_time = time.time()
print(f"Standard training completed in {end_time - start_time:.2f} seconds.")

# Prediction
train_predictions = mlp.predict(train_data, W1, b1, W2, b2)
train_accuracy = mlp.accuracy(train_labels_one_hot, train_predictions)
print(f"Accuracy on train set (Standard Training): {train_accuracy * 100:.2f}%")

test_predictions = mlp.predict(test_data, W1, b1, W2, b2)
test_accuracy = mlp.accuracy(test_labels_one_hot, test_predictions)
print(f"Accuracy on test set (Standard Training): {test_accuracy * 100:.2f}%")

print("Beginning training with Adam optimizer and Early Stopping...")
W1, b1, W2, b2, validation_accuracies = learning_rate_schedule.train_with_adam_early_stopping(
    train_data, train_labels_one_hot, test_data, test_labels_one_hot,
    input_size, hidden_size, output_size, initial_learning_rate, epochs, patience=500, lambda_l2=0.01
)

# Accuracy on validation set
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(validation_accuracies) + 1), validation_accuracies, label="Validation Accuracy (Adam + Early Stopping)")
plt.title("Validation Accuracy vs. Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()
