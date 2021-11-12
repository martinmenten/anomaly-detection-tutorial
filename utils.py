import matplotlib.pyplot as plt
import numpy as np

from typing import List


def plot(imgs: List[np.array], titles: List[str] = None, show: bool = True):
    """Plots a list of grayscale numpy images with ndim=2"""
    if not isinstance(imgs, list):
        imgs = [imgs]
    n = len(imgs)

    fig = plt.figure(figsize=(n * 3, 3))

    for i in range(n):
        ax = fig.add_subplot(1, n, i + 1)
        plt.imshow(imgs[i], cmap='gray', vmin=0., vmax=1.)
        plt.axis('off')

        if titles is not None:
            ax.set_title(titles[i])

    if show:
        plt.show()
    else:
        return fig
