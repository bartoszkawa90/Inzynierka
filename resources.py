import sys

# import cv2
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


def printArr(array=None):
    '''
    1 arg - prints array and its shape
    2 arg - prints additionaly max value from array
    '''
    print(" Array shape : {} \n {} \n Max : {}".format(array.shape, array, array.max()))




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


    # if h.shape[0] != h.shape[1]:
    #     # raise ValueError('Kernel must be square matrix.')

    if h.shape[0] % 2 == 0 or h.shape[1] % 2 == 0:
        raise ValueError('Kernel must have odd number of elements so it can have a center.')

    h = h[::-1, ::-1]
    # shape[0]  -  | num of rows
    # shape[1]  -  - num of columns
    y_shift = h.shape[0] // 2
    x_shift = h.shape[1] // 2

    if mode == "same":
        # zeros is x copy in bigger array and we work on it
        zeros = np.zeros((x.shape[0] + h.shape[0] - 1, x.shape[1] + h.shape[1] - 1))
        zeros[y_shift:y_shift + x.shape[0], x_shift:x_shift + x.shape[1]] = x
        # result is the array final values
        result = zeros.copy()
        # size corection which is essential to extract only final values
        endSizeCorrection = [int((i-j)/2) for i, j in zip(result.shape, x.shape)]

        for i in range(y_shift, y_shift + x.shape[0]):
            for j in range(x_shift, x_shift + x.shape[1]):
                # print(h)
                # print(zeros[i - y_shift: i + y_shift + 1, j - x_shift:j + x_shift + 1])
                # print(h * zeros[i - y_shift: i + y_shift + 1, j - x_shift:j + x_shift + 1])
                result[i, j] = MAC(h, zeros[i - y_shift: i + y_shift + 1, j - x_shift:j + x_shift + 1])
        return result[endSizeCorrection[0]:result.shape[0]-endSizeCorrection[0],
               endSizeCorrection[1]:result.shape[1]-endSizeCorrection[1]].astype(np.uint8)

    elif mode == "full":
        # zeros is x copy in bigger array and we work on it
        zeros = np.zeros((x.shape[0] + h.shape[0] + 1, x.shape[1] + h.shape[1] + 1))
        # result is the array final values
        result = zeros.copy()
        zeros[y_shift + 1:y_shift + x.shape[0] + 1, x_shift + 1:x_shift + x.shape[1] + 1] = x

        for i in range(y_shift, result.shape[0]-1):
            for j in range(x_shift, result.shape[1]-1):
                # print(h)
                # print(zeros[i - y_shift: i + y_shift + 1, j - x_shift:j + x_shift + 1])
                # print(h * zeros[i - y_shift: i + y_shift + 1, j - x_shift:j + x_shift + 1])
                result[i, j] = MAC(h, zeros[i - y_shift: i + y_shift + 1, j - x_shift:j + x_shift + 1])
        return result[1:result.shape[0]-1, 1:result.shape[1]-1].astype(np.uint8)


def gaussKernelGenerator(size=3, sigma=1):
    x = np.arange(size)
    x = x - x[x.shape[0]//2]
    e = (1/np.sqrt(2*math.pi*sigma))
    temp = [ e*np.exp((-i**2)/(2*sigma**2)) for i in x]
    return np.array(temp).reshape(size, 1)


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
    gauss = gaussKernelGenerator(5, 1)

    # convolution with gaussian kernel 2 times(rows and columns) to blure whole image
    gImage = Convolution2D(Convolution2D(gray, gauss, mode='same'), gauss.T, mode="same")

    # # magnitude and angle calculation
    Gx = Convolution2D(gImage, XSobelKernel, 'same')
    Gy = Convolution2D(gImage, YSobelKernel, 'same')
    G =  np.sqrt(Gx**2 + Gy**2)
    printArr(G)
    G = G/G.max() * 255   ## Scale
    theta = np.arctan2(Gy, Gx)

    # # Non-maximum Suppression
    # https://medium.com/@ceng.mavuzer/canny-edge-detection-algorithm-with-python-17ac62c61d2e



    G = G.astype(np.uint8)
    theta = theta.astype(np.uint8)
    # print(G.shape, "\n", G)
    # print(theta.shape, "\n", theta)

    return G



## test LoG / Canny

img = cv2.imread('spodnie.jpeg')
# img = cv2.imread('Wycinki/resized_wycinek_4_67nieb_82czar.jpg')
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

# non square kernel
# print("mine   \n", Convolution2D(exampleArray, gauss, mode="same"))
# print(convolution_2d_1d_kernel(exampleArray, gauss[:, 0]))
# print("scipy.signal  \n", sig.convolve2d(exampleArray, gauss, mode='same').astype(np.uint8))
# print("scipy.ndimage   \n", scipy.ndimage.convolve(exampleArray, gauss, mode="constant"))
# print("\n")

# # square input image
# print("mine   \n", Convolution2D(MALAexample, exampleKernel, mode="same"))
# print("scipy.signal  \n", sig.convolve2d(MALAexample, exampleKernel, mode='same').astype(np.uint8))
# print("scipy.ndimage   \n", scipy.ndimage.convolve(MALAexample, exampleKernel, mode="constant"))
# print("\n")

# # non square input image
# print("mine   \n", Convolution2D(exampleArray, exampleKernel, mode="same"))
# print("scipy.signal  \n", sig.convolve2d(exampleArray, exampleKernel, mode='same').astype(np.uint8))
# print("scipy.ndimage   \n", scipy.ndimage.convolve(exampleArray, exampleKernel, mode="constant"))
# print("\n")



# print(scipy.ndimage.filters.convolve(MALAexample, gauss))   # to samo co wyzej



















