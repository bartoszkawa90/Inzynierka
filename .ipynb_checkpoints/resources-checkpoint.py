import numpy as np

from variables import *


# FUNCTIONS / KEYWORDS
def plot_photo(title, image,height, widht):
    ''' Plots photo is given  resolution
        title - title of ploted photo
        image - image to plot
        height - height of ploted photo
        width - width of ploted photo
    '''
    # while True:
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, height, widht)
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    sys.exit()


# @jit(nopython=False)
def find_extreme_contours(contours):
    SIZE_MAX = contours[0].shape[0]
    size_min = contours[0].shape[0]
    id_min = 0
    ID_MAX = 0
    count = 0

    for con in contours:
        if con.shape[0] < size_min:
            size_min = con.shape[0]
            id_min = count
        if con.shape[0] > SIZE_MAX:
            SIZE_MAX = con.shape[0]
            ID_MAX = count
        count += 1;

    return contours[ID_MAX], contours[id_min]


# @jit(nopython=False)
def extract_cell(contour=None, img=None, clear=0):
    x_min, y_min, x_max, y_max = cv2.boundingRect(contour)
    cell = img[y_min:y_min + y_max, x_min:x_min + x_max]

    # clear background => set to white //255
    if clear == 1:
        for line_id in range(cell.shape[0]):
            for pixel_id in range(cell.shape[1]):
                if cell[line_id][pixel_id] > 150:
                    cell[line_id][pixel_id] = 255
        # remove single pixels outsize of cell
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cell = cv2.morphologyEx(cell, cv2.MORPH_OPEN, kernel)

    return cell


# @njit
def Edge_Detection():
    pass


# @njit
def Convolution2D():
    pass





















