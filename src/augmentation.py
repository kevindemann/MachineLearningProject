import skimage.transform
import preprocessing
import numpy as np
from matplotlib import pyplot as plt
import skimage


def augment_data(inputs, img_shape, rotate_rad, num_versions=2):
    """
    Augments the input data by creating multiple rotated versions per image.
    """
    minval = np.min(inputs)
    maxval = np.max(inputs)

    augmented_data = []
    for i in range(len(inputs)):
        original = inputs[i].reshape(img_shape)
        augmented_data.append(original.flatten())  # Keep the original image

        for _ in range(num_versions):
            rotate = (np.random.random() - 0.5) * 2 * rotate_rad
            tform = skimage.transform.AffineTransform(rotation=rotate)
            transformed = skimage.transform.warp(original, inverse_map=tform.inverse)
            augmented_data.append(np.clip(transformed, minval, maxval).flatten())

    return np.vstack(augmented_data)


# Test for augmenting data
if __name__ == "__main__":
    import preprocessing

    np.random.seed(42)

    dataset_path = "data/mfeat-pix"

    # Augmentation parameters. All params are ratios except for rotation
    augment_rotate = np.radians(12)
    img_shape = (16, 15)

    data = preprocessing.load_data(dataset_path)

    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(data[i].reshape(img_shape), cmap="gray")
        ax.axis("off")
    plt.suptitle("Original Data")
    plt.show()

    augmented = augment_data(data, img_shape, augment_rotate, num_versions=3)

    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(augmented[i].reshape(img_shape), cmap="gray")
        ax.axis("off")
    plt.suptitle("Augmented Data")
    plt.show()
