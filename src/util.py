import numpy as np
from matplotlib import pyplot as plt


# Show first 10 samples of every number
def show_dataset(dataset, title="", img_shape=None):
    f, axarr = plt.subplots(10, 10)

    f.subplots_adjust(wspace=0.01, hspace=0.01)
    f.suptitle(title)

    # Show first data set
    for i in range(10):
        for j in range(10):
            img = dataset[len(dataset) // 10 * i + j]
            if img_shape is not None:
                img = img.reshape(img_shape)

            axarr[i][j].imshow(
                img,
                cmap="gray",
                interpolation="nearest",
            )

            axarr[i][j].axis("off")
