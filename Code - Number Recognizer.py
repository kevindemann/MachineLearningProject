import matplotlib.pyplot as plt
import numpy as np
import idx2numpy

training_images_data ='Data/train-images.idx3-ubyte'
training_labels_data = 'Data/train-labels.idx1-ubyte'
test_images_data = 'Data/t10k-images.idx3-ubyte'
test_labels_data = 'Data/t10k-labels.idx1-ubyte'


def read(x): return idx2numpy.convert_from_file(x)

training_images = read(training_images_data)
training_labels = read(training_labels_data)
test_images = read(test_images_data)
test_labels = read(test_labels_data)

plt.imshow(training_images[1], cmap=plt.cm.binary)
