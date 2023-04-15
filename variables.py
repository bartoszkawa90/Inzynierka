# IMPORTS
import cv2
import sys
import numpy as np
import scipy.signal as sig
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from skimage.filters import threshold_otsu



# VARIABLES

# Horizontal Edge Detection
kernel_hor = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])

# Vertical Edge Detection
kernel_ver = np.array([
    [-1, -1, -1],
    [0, 0, 0],
    [1, 1, 1]
])
