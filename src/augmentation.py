import skimage.transform
from util import show_dataset

import numpy as np
from matplotlib import pyplot as plt
import skimage


def augment_data(input, noise_level, rel_scale, rotate_rad, rel_translate, img_shape):
    """
    Augments the input data with random noise, scale, rotation and translation.
    """
    minval = np.min(input)
    maxval = np.max(input)

    output = np.empty_like(input)

    for i in range(len(input)):
        scale = rel_scale + (1 - rel_scale) * np.random.random()
        rotate = (np.random.random() - 0.5) * 2 * rotate_rad
        translate = (np.random.random(2) - 0.5) * 2 * rel_translate * img_shape

        noise = np.random.normal(0, noise_level * maxval, img_shape)
        tform = skimage.transform.AffineTransform(
            scale=scale, rotation=rotate, shear=0, translation=translate
        )

        transformed = skimage.transform.warp(
            input[i].reshape(img_shape), inverse_map=tform.inverse
        )

        output[i] = np.clip(transformed + noise, minval, maxval).flatten()

    return output


# Test for augmenting data
if __name__ == "__main__":
    import preprocessing

    np.random.seed(42)

    dataset_path = "small/mfeat-pix"

    # Augmentation parameters. All params are ratios except for rotation
    augment_noise = 0.2
    augment_scale = 0.7
    augment_rotate = np.radians(15)
    augment_translate = 0.15

    img_shape = (16, 15)

    data = preprocessing.load_data(dataset_path)

    show_dataset(
        data,
        "Original data",
        img_shape,
    )

    augmented = augment_data(
        data, augment_noise, augment_scale, augment_rotate, augment_translate, img_shape
    )
    show_dataset(augmented, "augmented data", img_shape)

    plt.show()
