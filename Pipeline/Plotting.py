import numpy as np
import matplotlib.pyplot as plt


class Plot:

    @staticmethod
    def plot_image(band: np.ndarray):
        plt.imshow(band)
        plt.show()

    @staticmethod
    def plot_mask(mask):
        plt.imshow(mask, cmap='binary')
        plt.show()
