# IMPORTS
import cv2
from numba import jit,vectorize,njit
import sys
import numpy as np
import  pandas
import itertools
import time
# import scipy.signal as sig
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from PIL import Image
from skimage.filters import threshold_otsu



# VARIABLES
CLEAR_BACKGROUND = 1
LEAVE_BACKGROUND = 0
MAX_AREA_OF_SINGLE_CELL = 0
MIN_AREA_OF_SINGLE_CELL = 0

