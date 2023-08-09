import sys

import cv2
import numpy as np
import scipy.signal

from variables import *


# FUNCTIONS / KEYWORDS
def plot_photo(title="None", image=None, height=900, widht=900):
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
def delete_incorrect_contours(contours):
    c = tuple([con for con in contours if cv2.contourArea(con) > 2000 or cv2.contourArea(con) < 50])
    contours = c
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
        count += 1

    largest, smallest = contours[ID_MAX], contours[id_min]
    if (largest.shape[0] > 1000 or smallest.shape[0] < 30):
        conts = tuple([con for con in contours if (con.shape[0] < 1000 and con.shape[0] > 30)])
    elif (smallest.shape[0] < 30):
        conts = tuple([con for con in contours if (con.shape[0] > 30)])
    elif (largest.shape[0] > 1000):
        conts = tuple([con for con in contours if (con.shape[0] < 1000)])
    else:
        conts = contours

    conts = c
    return conts


# Version for colors
# @jit(nopython=False)
def extract_cell(contour=None, img=None, clear=0):
    x_min, y_min, x_max, y_max = cv2.boundingRect(contour)
    cell = img[y_min:y_min + y_max, x_min:x_min + x_max]

    # clear background => set to white //255
    if clear == 1:
        for line_id in range(cell.shape[0]):
            for pixel_id in range(cell.shape[1]):
                if cell[line_id][pixel_id][0] > 150 or cell[line_id][pixel_id][1] > 150 or cell[line_id][pixel_id][2] > 150:
                    cell[line_id][pixel_id] = 255
        # remove single pixels outsize of cell
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # cell = cv2.morphologyEx(cell, cv2.MORPH_OPEN, kernel)

    return cell


def background_procentage(cell):
    white_count = 0
    cell_count = 0
    for i in cell:
        for j in i:
            if len(j[j.__gt__(158)]) != 0:
                white_count += 1
            cell_count += 1
    return white_count/cell_count


def MAC(M1, M2):    # Multiply-accumulate function
    """
    :param M1: first array
    :param M2: second array
    :return: returns product of multiply and accumulate operation
    """
    return np.sum(M1 * M2)


def Convolution2D(x, h, mode="full"):
    """
    :param x: Input array
    :param h: kernel
    :param mode: mode of convolution ( determine how the output will look like
    :return result: returns product of 2D convolution ==>  x * h
    """

    if h.shape[0] != h.shape[1]:
        raise ValueError('Kernel must be square matrix.')

    elif h.shape[0] % 2 == 0 or h.shape[1] % 2 == 0:
        raise ValueError('Kernel must have odd number of elements so it can have a center.')

    h = h[::-1, ::-1]
    # shape[0]  -  | num of rows
    # shape[1]  -  - num of columns
    y_shift = h.shape[0] // 2
    x_shift = h.shape[1] // 2

    if mode == "same":
        zeros = np.zeros((x.shape[0] + h.shape[0] - 1, x.shape[1] + h.shape[1] - 1))
        zeros[y_shift:y_shift + x.shape[0], x_shift:x_shift + x.shape[1]] = x
        result = zeros.copy()
        for i in range(y_shift, y_shift + x.shape[0]):
            for j in range(x_shift, x_shift + x.shape[1]):
                # print(h)
                # print(zeros[i - y_shift: i + y_shift + 1, j - x_shift:j + x_shift + 1])
                # print(h * zeros[i - y_shift: i + y_shift + 1, j - x_shift:j + x_shift + 1])
                result[i, j] = MAC(h, zeros[i - y_shift: i + y_shift + 1, j - x_shift:j + x_shift + 1])
        return result[1:result.shape[0]-1, 1:result.shape[1]-1].astype(np.uint8)

    elif mode == "full":
        zeros = np.zeros((x.shape[0] + h.shape[0] + 1, x.shape[1] + h.shape[1] + 1))
        result = zeros.copy()
        print(result.shape)
        zeros[y_shift + 1:y_shift + x.shape[0] + 1, x_shift + 1:x_shift + x.shape[1] + 1] = x
        for i in range(y_shift, result.shape[0]-1):
            for j in range(x_shift, result.shape[1]-1):
                # print(h)
                # print(zeros[i - y_shift: i + y_shift + 1, j - x_shift:j + x_shift + 1])
                # print(h * zeros[i - y_shift: i + y_shift + 1, j - x_shift:j + x_shift + 1])
                result[i, j] = MAC(h, zeros[i - y_shift: i + y_shift + 1, j - x_shift:j + x_shift + 1])
        return result[1:result.shape[0]-1, 1:result.shape[1]-1].astype(np.uint8)


def gaussianFilterGenerator(size=3, sigma=1):
    X = np.zeros((size, size))
    Y = np.zeros((size, size))
    for i in range(2*size):
        if i < size:
            X[0, i] = Y[i, 0] = -1
        else:
            X[size-1, i-size-1] = Y[i-size-1, size-1] = 1
    result = (1/(2*np.pi*sigma*sigma)) * np.exp(  (-1*(np.power(X, 2) + np.power(Y, 2))) / (2*sigma*sigma)  )
    return result


def Laplace_Mask(alfa=0):
    arr = np.zeros((3, 3))
    arr[0][0:2:2] = arr[0][2] = arr[2][0] = arr[2][2] = alfa/4
    arr[0][1] = arr[1][0] = arr[1][2] = arr[2][1] = (1-alfa)/4
    arr[1][1] = -1
    return (4/(alfa+1))*arr


# @njit
# def LoG(gray):
#     # zastosowanie filtru Gaussa w celu ograniczenia szumów
#     gauss = gaussianFilterGenerator(size=5, sigma=0.5)
#     print(gauss)
#     gImage = Convolution2D(gray, gauss, mode="same")
#
#     ## nałozenie maski Laplace'a , znajdowanie miejsc zerowych mozna załatwić przez przefiltrowanie obrazu przez maske
#     ##     Laplace'a z alfa = 0  ( standardowo takiej maski uzywamy )
#     ##     Raczej uzywamy laplasjanu zamiast gradientu bo zapewnia dodatkowe informacje
#     Lmask = Laplace_Mask(-0.5)
#     print(Lmask)
#     lImage = Convolution2D(gray, Lmask, mode="same")
#
#
#
#     return lImage


#  II wersja
def Canny(gray):
    # zastosowanie filtru Gaussa w celu ograniczenia szumów
    gauss = gaussianFilterGenerator(size=3, sigma=4)
    gImage = Convolution2D(gray, gauss, mode="same")

    # # uzycie Sobel filter
    x = Convolution2D(gImage, XSobelKernel, 'same')
    y = Convolution2D(gImage, YSobelKernel, "same")

    G = np.hypot(x, y)
    G = G/G.max() * 255
    theta = np.arctan2(x, y)
    G = G.astype(np.uint8)
    theta = theta.astype(np.uint8)

    # print(G.shape, "\n", G)
    # print(theta.shape, "\n", theta)

    return G



## test Canny 1

# img = cv2.imread('spodnie.jpeg')
img = cv2.imread('Wycinki/resized_wycinek_4_67nieb_82czar.jpg')
# img = cv2.imread('zdj_z_arykułu.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
# Canny(gray)
# plot_photo("From Canny", LoG(gray))
plot_photo("From Canny", Canny(gray))



# test Canny 2
# img = cv2.imread('Wycinki/resized_wycinek_4_67nieb_82czar.jpg)
# gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
# plot_photo("From Canny", Canny(gray))


## test COnvolution
# print("mine   \n", Convolution2D(exampleArray, exampleKernel, mode="full"))
# print("scipy.signal  \n", sig.convolve2d(exampleArray, exampleKernel, mode='full'))
# print("scipy.ndimage   \n", scipy.ndimage.convolve(exampleArray, exampleKernel, mode="constant"))
# print(scipy.ndimage.filters.convolve(MALAexample, exampleKernel))   # to samo co wyzej



















