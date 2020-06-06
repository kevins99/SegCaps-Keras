import numpy as np
import cv2
import scipy.misc
from PIL import Image

def rgb2gray(rgb):
    print(rgb.shape)
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def fractal_dimension(Z, threshold=100):
    # Only for 2d image
    assert(len(Z.shape) == 2)

    def boxcount(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0), np.arange(0, Z.shape[1], k), axis=1)

        
        return len(np.where((S > 0) & (S < k*k))[0])

    Z = (Z > threshold)

    p = min(Z.shape)

    n = 2**np.floor(np.log(p)/np.log(2))
    n = int(np.log(n)/np.log(2))

    sizes = 2**np.arange(n, 1, -1)
    counts = []
    for size in sizes:
        counts.append(boxcount(Z, size))

    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)   #forms an equation with x and y and gives us the slope and intercept. -slope is the fractal number for that image
    return coeffs[0]
