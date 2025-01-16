import preprocessing
import mlp
import time

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
learning_rate = 0.01
epochs = 1000

# Training
print("Beginning training...")
start_time = time.time()
W1, b1, W2, b2 = mlp.train(train_data, train_labels_one_hot, input_size, hidden_size, output_size, learning_rate, epochs)
end_time = time.time()
print(f"Train completed in {end_time - start_time:.2f} seconds.")

# Prediction
train_predictions = mlp.predict(train_data, W1, b1, W2, b2)
train_accuracy = mlp.accuracy(train_labels_one_hot, train_predictions)
print(f"Accuracy on train set: {train_accuracy * 100:.2f}%")

# Test
test_predictions = mlp.predict(test_data, W1, b1, W2, b2)
test_accuracy = mlp.accuracy(test_labels_one_hot, test_predictions)
print(f"Accuracy on test set: {test_accuracy * 100:.2f}%")
