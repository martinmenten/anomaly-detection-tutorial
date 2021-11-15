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


def plot_anomaly_scores(scores: np.ndarray, labels: np.ndarray):
    """Plot a histogram of the anomaly scores"""
    # Divide the scores into normal and anomal
    normal_scores = scores[np.where(labels == 0)]
    anomal_scores = scores[np.where(labels == 1)]
    # Use only scores up to 1.
    normal_scores = normal_scores[normal_scores < 1.]
    anomal_scores = anomal_scores[anomal_scores < 1.]
    plt.hist(normal_scores, bins=100, alpha=0.5, label='normal', color='b')
    plt.hist(anomal_scores, bins=100, alpha=0.5, label='anomal', color='r')
    plt.xlabel('anomaly score')
    plt.ylabel('count')
    plt.legend()
    plt.show()
