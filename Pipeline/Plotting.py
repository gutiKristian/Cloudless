import numpy as np
import matplotlib.pyplot as plt


class Plot:

    @staticmethod
    def plot_image_mine(band: np.ndarray):
        plt.imshow(band)
        plt.show()

    @staticmethod
    def plot_mask_mine(mask):
        plt.imshow(mask, cmap='binary')
        plt.show()

    # SOURCE: https://github.com/sentinel-hub/sentinel2-cloud-detector/blob/master/examples/plotting_utils.py
    @staticmethod
    def plot_image(image=None, mask=None, ax=None, factor=3.5 / 255, clip_range=(0, 1), **kwargs):
        """ Utility function for plotting RGB images and masks.
        """
        if ax is None:
            _, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))

        mask_color = [255, 255, 255, 255] if image is None else [255, 255, 0, 100]

        if image is None:
            if mask is None:
                raise ValueError('image or mask should be given')
            image = np.zeros(mask.shape + (3,), dtype=np.uint8)

        ax.imshow(np.clip(image * factor, *clip_range), **kwargs)

        if mask is not None:
            cloud_image = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)

            cloud_image[mask == 1] = np.asarray(mask_color, dtype=np.uint8)

            ax.imshow(cloud_image)
        plt.show()

    @staticmethod
    def plot_probabilities(image, proba, factor=3.5 / 255):
        """ Utility function for plotting a RGB image and its cloud probability map next to each other.
        """
        plt.figure(figsize=(15, 15))
        ax = plt.subplot(1, 2, 1)
        ax.imshow(np.clip(image * factor, 0, 1))
        ax = plt.subplot(1, 2, 2)
        ax.imshow(proba, cmap=plt.cm.inferno)
        plt.show()
