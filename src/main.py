import preprocessing

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

