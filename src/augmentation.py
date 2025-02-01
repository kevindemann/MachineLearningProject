import skimage.transform
import numpy as np
import os
from datetime import datetime

def augment_data(inputs, img_shape, rotate_rad, num_versions):
    """
    Augments the input data by creating multiple rotated versions per image.
    Saves the augmented dataset in .npy format without overwriting previous ones.
    """
    minval = np.min(inputs)
    maxval = np.max(inputs)

    augmented_data = []
    for i in range(len(inputs)):
        original = inputs[i].reshape(img_shape)
        augmented_data.append(original.flatten())

        for _ in range(num_versions):
            rotate = (np.random.random() - 0.5) * 2 * rotate_rad
            tform = skimage.transform.AffineTransform(rotation=rotate)
            transformed = skimage.transform.warp(original, inverse_map=tform.inverse)
            augmented_data.append(np.clip(transformed, minval, maxval).flatten())

    augmented_data = np.vstack(augmented_data)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"../augmented_datasets/augmented_data_{timestamp}.npy"

    np.save(file_name, augmented_data)
    print(f"Augmented dataset saved as: {file_name}")

    return augmented_data