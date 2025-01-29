import augmentation
import preprocessing
import mlp
import learning_rate_schedule
import util

import matplotlib.pyplot as plt
import numpy as np
import time

dataset_path = "small/mfeat-pix"

np.random.seed(42)

# Augmentation parameters. All params are ratios except for rotation
augment_noise = 0.2
augment_scale = 0.8
augment_rotate = np.radians(15)
augment_translate = 0.1

img_shape = (16, 15)

data = preprocessing.load_data(dataset_path)

num_classes = 10
samples_per_class = 200
train_samples_per_class = 100
test_samples_per_class = 100

train_data, test_data = preprocessing.split_data(data, num_classes, samples_per_class, train_samples_per_class)

train_data = augmentation.augment_data(train_data, augment_noise, augment_scale, augment_rotate, augment_translate, img_shape)
util.show_dataset(train_data, title= "Augmented training data", img_shape=img_shape)
util.show_dataset(test_data, title= "Testing data", img_shape=img_shape)

train_labels, test_labels = preprocessing.create_labels(num_classes, train_samples_per_class, test_samples_per_class)

train_labels_one_hot = preprocessing.one_hot_encode(train_labels)
test_labels_one_hot = preprocessing.one_hot_encode(test_labels)

input_size = train_data.shape[1]
hidden_size = 128
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

# Prediction for Adam optimizer
train_predictions_adam = learning_rate_schedule.predict(train_data, W1, b1, W2, b2)
train_accuracy_adam = learning_rate_schedule.accuracy(train_labels_one_hot, train_predictions_adam)
print(f"Accuracy on train set (Adam + Early Stopping): {train_accuracy_adam * 100:.2f}%")

test_predictions_adam = learning_rate_schedule.predict(test_data, W1, b1, W2, b2)
test_accuracy_adam = learning_rate_schedule.accuracy(test_labels_one_hot, test_predictions_adam)
print(f"Accuracy on test set (Adam + Early Stopping): {test_accuracy_adam * 100:.2f}%")
