# IMPORTS
import cv2
from numba import jit,vectorize,njit
import sys
import os
import math
import numpy as np
import scipy.signal as sig
import  pandas
import itertools
import time

from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from PIL import Image
# from skimage.filters import threshold_otsu



# VARIABLES
CLEAR_BACKGROUND = 1
LEAVE_BACKGROUND = 0
MAX_AREA_OF_SINGLE_CELL = 0
MIN_AREA_OF_SINGLE_CELL = 0

exampleArray = np.array([[25, 100, 75, 49, 130],
                         [50, 80, 0, 70, 100],
                         [5, 10, 20, 30, 0],
                         [60, 50, 12, 24, 32],
                         [37, 53, 55, 21, 90],
                         [140, 17, 0, 23, 222]])
MALAexample = np.array([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]])
exampleKernel = np.array([[1, 0, 1],
                          [0, 1, 0],
                          [0, 0, 1]])

XSobelKernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
YSobelKernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

