import sys

# import cv2
import numpy as np
import scipy.signal

from variables import *


# FUNCTIONS / KEYWORDS
def plot_photo(title="None", image=None, height=1500, widht=1500):
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


def printArr(*args):
    '''
    1 arg - prints array and its shape
    2 arg - prints additionaly max value from array
    '''
    for arr in  args:
        print(" Array shape : {} \n {} \n Max : {} \n Min : {} \n ".format(arr.shape, arr, arr.max(), arr.min()))


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


def Convolution2D(x, h, mode="full", returnUINT8=False):
    """
    :param x: Input array
    :param h: kernel
    :param mode: mode of convolution ( determine how the output will look like
    :param returnUINT8: if True => returned result will be type np.uint8
    :return result: returns product of 2D convolution ==>  x * h
    """
    # if h.shape[0] != h.shape[1]:
    #     # raise ValueError('Kernel must be square matrix.')

    # if h.shape[0] % 2 == 0 or h.shape[1] % 2 == 0:
    #     raise ValueError('Kernel must have odd number of elements so it can have a center.')

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
        if returnUINT8:
            return result[endSizeCorrection[0]:result.shape[0]-endSizeCorrection[0],
               endSizeCorrection[1]:result.shape[1]-endSizeCorrection[1]].astype(np.uint8)
        else:
            return result[endSizeCorrection[0]:result.shape[0]-endSizeCorrection[0],
               endSizeCorrection[1]:result.shape[1]-endSizeCorrection[1]]

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
        if returnUINT8:
            return result[1:result.shape[0]-1, 1:result.shape[1]-1].astype(np.uint8)
        else:
            return result[1:result.shape[0]-1, 1:result.shape[1]-1]


def gaussKernelGenerator(size=3, sigma=1):
    '''
    :param size: size of gauss kernel ( shape will be (size,1) )
    :param sigma: parameter used to calculate gauss kernel
    :return: returns gauss kernel
    '''
    x = np.arange(size)
    x = x - x[x.shape[0]//2]
    e = (1/np.sqrt(2*math.pi*sigma))
    temp = [ e*np.exp((-i**2)/(2*sigma**2)) for i in x]
    return np.array(temp).reshape(size, 1)


def scale(arr, newMax):
    return ((arr-arr.min()) / (arr.max() - arr.min()))*newMax


#  II wersja
def Canny(grayImage=None, mask_x=mask_x, mask_y=mask_y, lowBoundry=10, highBoundry=30):
    '''
    :param grayImage: input image in gray scale
    :param lowBoundry: low limit for thresholding
    :param highBoundry: high limit for thresholding
    :return: image with marked edges
    '''
    if grayImage is None:
        print("You have to give at least one argument")
        return

    # zastosowanie filtru Gaussa w celu ograniczenia szumów
    gauss = gaussKernelGenerator(5, 1)
    # convolution with gaussian kernel 2 times(rows and columns) to blure whole image
    gImage = Convolution2D(Convolution2D(grayImage, gauss, mode='same'), gauss.T, mode="same")
    ## horizontal and vertical masks
    # mask_x = np.zeros((3, 1))
    # mask_x[0] = -1
    # mask_x[2] = 1
    # mask_y = mask_x.T
    # printArr(mask_x, mask_y)

    Gx = Convolution2D(gImage, mask_x, mode='same')
    Gy = Convolution2D(gImage, mask_y, mode='same')

    ## gradient magnitude and angle(direction)
    GMag = np.sqrt(Gx**2 + Gy**2)
    Gangle = np.arctan2(Gy, Gx) * (180/np.pi)  ## angle in deg not in radians

    # # Non-maximum Suppression
    rowNum, colNum = GMag.shape
    GangleRounded = 45 * (np.round(Gangle / 45))  ## a way to round angle values to be multiplecation of 45
    result = np.zeros((rowNum, colNum))

    ## we want to consider 3x3 matrixes so we do not teke first and last
    for row in range(1, rowNum-1):
        for col in range(1, colNum-1):
            angle = GangleRounded[row, col]
            if angle == 180 or angle == -180 or angle == 0:
                edge1 = GMag[row-1, col]
                edge2 = GMag[row+1, col]
            elif angle == 90 or angle == -90:
                edge1 = GMag[row, col - 1]
                edge2 = GMag[row, col + 1]
            elif angle == -45 or angle == 135:
                edge1 = GMag[row + 1, col - 1]
                edge2 = GMag[row - 1, col + 1]
            elif angle == 45 or angle == -135:
                edge1 = GMag[row - 1, col - 1]
                edge2 = GMag[row + 1, col + 1]
            else:
                print("Something went wrong with Non-maximum Suppression")
                return
            # sprawdzamy po kątach w którą stone idzie nasza krawędz ale do ostatecznego wyniku
            # idą tylko najwyzsze wartosci zeby zostawic cienką krawędz
            if GMag[row, col] > edge1 and GMag[row, col] > edge2:
                result[row, col] = GMag[row, col]

    ## Thresholding
    # chodzi o to ze jest granica górna i dolna i :\
    #     jesli wartosc pixeli jest wieksza niz górna granica to na pewno mamy krawędź
    #     jesli wartość pixeli jest nizsza niz dolna granica to na pewno nie jest to krawędź
    #     jesli wartosc jest pomiedzy granicami to aby byc uznana za czesc krawedzi musi sie
    #     łączyć z pixelami o wartości powyzej górnej granicy czyli z pewną krawędzią

    np.where(result < lowBoundry, result, 0)
    np.where(result > highBoundry, result, 255)
    neighborPixels = np.zeros((3, 3))
    for i in range(1, result.shape[0] - 1):
        for j in range(1, result.shape[1] - 1):
            # if result[i, j] > highBoundry: result[i, j] = 255
            # elif result[i, j] < lowBoundry: result[i, j] = 0
            # else:
            if result[i, j] != 0 and result[i, j] != 255:
                neighborPixels = result[i-1:i+1, j-1:j+1]
                if np.any(neighborPixels >= highBoundry):
                    result[i, j] = 255
                else:
                    result[i, j] = 0
            # if result[i, j] != 0:
            #     result[i, j] = 255

    return result.astype(np.uint8)



## test LoG / Canny

# img = cv2.imread('spodnie.jpeg')
img = cv2.imread('Wycinki/resized_wycinek_4_67nieb_82czar.jpg')
# img = cv2.imread('zdj_z_arykułu.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
# Canny(gray)
# plot_photo("From Canny", LoG(gray))
plot_photo("From Canny", Canny(gray))

# edge = cv2.Canny(gray, 1, 10)
# contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# conts = delete_incorrect_contours(contours)
# plot_photo("From Canny", edge)
# cv2.drawContours(img, conts, -1, (0, 255, 0), 3)
# plot_photo("From Canny", img)


# test Canny 2
# img = cv2.imread('Wycinki/resized_wycinek_4_67nieb_82czar.jpg)
# gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
# plot_photo("From Canny", Canny(gray))


## test COnvolution
# mask_x = np.zeros((3, 1))
# mask_x[0] = -1
# mask_x[2] = 1
# mask_y = mask_x.T
# printArr(mask_x, mask_y)

# non square kernel
# print("mine   \n", Convolution2D(exampleArray, mask_y, mode="same"))
# print("scipy.signal  \n", sig.convolve2d(exampleArray, mask_y, mode='same').astype(np.uint8))
# print("scipy.ndimage   \n", scipy.ndimage.convolve(exampleArray, mask_y, mode="constant"))
# print("cv2 filter \n", cv2.filter2D(exampleArray, -1, mask_x))
# print("corelation \n ", scipy.ndimage.correlate(exampleArray, mask_x))
# print("\n")

# # square input image
# print("mine   \n", Convolution2D(MALAexample, mask_y, mode="same"))
# print("scipy.signal  \n", sig.convolve2d(MALAexample, mask_y, mode='same').astype(np.uint8))
# print("scipy.ndimage   \n", scipy.ndimage.convolve(MALAexample, mask_y, mode="constant"))
# print("cv2 filter \n", cv2.filter2D(MALAexample, -1, mask_x))
# print("\n")

# # non square input image
# print("mine   \n", Convolution2D(exampleArray, exampleKernel, mode="same"))
# print("scipy.signal  \n", sig.convolve2d(exampleArray, exampleKernel, mode='same').astype(np.uint8))
# print("scipy.ndimage   \n", scipy.ndimage.convolve(exampleArray, exampleKernel, mode="constant"))
# print("\n")



# print(scipy.ndimage.filters.convolve(MALAexample, gauss))   # to samo co wyzej



















