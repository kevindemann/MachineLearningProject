import numpy as np

from matplotlib import pyplot as plt

with open("mfeat-pix") as datafile:
    raw_data = datafile.read()

lines = raw_data.split('\n')
data = [line.split() for line in lines]

numbers = [
    int(x)
    for line in data
    for x in line
]

arr = np.array(numbers).reshape((2000, 16,15))

# Show first data set
plt.imshow(arr[0] * 255 /6, cmap="gray")
plt.title("Multiple features 16x15")
plt.show()