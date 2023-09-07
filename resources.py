## IMPORTS
from variables import *
import time
import cv2
import skimage
from copy import deepcopy
from copy import deepcopy
from pprint import pprint
from sys import exit


## FUNCTIONS / KEYWORDS
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
    exit()


def printArr(*args):
    '''
    arg :  array which max min and shape will be printed
    '''
    for arr in  args:
        print(" Array name ::   {}\n Array shape : {} \n {} \n Max : {} \n Min : {} \n ".format('ada', arr.shape, arr, arr.max(), arr.min()))


# @jit(nopython=False)
def contours_processing(contours, lowBoundry=55, highBoundry=2000, RETURN_ADDITIONAL_INFORMATION=0):
    '''
    Function for finding smallest and largest contours and removing too small and too large contours
    :param contours: contours given to a function for processing
    :param lowBoundry: low limit of contour size
    :param highBoundry: high limit of contour size
    :return: tuple of selected contours with correct size
    '''

    # Filter contours according to size of contours
    conts = tuple([con for con in contours if con.shape[0] > lowBoundry and con.shape[0] < highBoundry])

    # Additional Data
    if RETURN_ADDITIONAL_INFORMATION == 1:
        contours = conts
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
        return conts, smallest, largest, id_min, ID_MAX

    return conts


def filterContoursValue(contours=None, img=None, lowPixelBoundry=155, highPixelBoundry=193, cellProcentage=0.5):
    '''
    :param contours: tuple of contours which will be filtered
    :param img: photo from which the contours were taken
    :param lowPixelBoundry:  low limit to decide if cell is white, blue or black
    :param highPixelBoundry: high limit to decide if cell is white, blue or black
    :param cellProcentage: procentage of pixels which determines the color
            // example:  if white pixels > cellProcentage * Number Of Pixels  ==>  cell is white etc.
    :return: tuple of chosen contours
    '''
    # conts = []
    # for con in contours:
    #     x_min, y_min, x_max, y_max = cv2.boundingRect(con)
    #     cell = img[y_min:y_min + y_max, x_min:x_min + x_max]
        # if 193 > np.mean(cell) > 3:
        #     conts.append(con)

    # return tuple(conts)
    if contours == None or img == None:
        print("Something went wrong")
        return

    black = 0
    blue = 0
    white = 0
    conts = []
    blackCon = []
    blueCon = []

    for con in contours:
        x_min, y_min, x_max, y_max = cv2.boundingRect(con)
        cell = img[y_min:y_min + y_max, x_min:x_min + x_max]
        size = cell.shape[0]*cell.shape[1]

        for line_id in range(cell.shape[0]):
            for pixel_id in range(cell.shape[1]):
                if cell[line_id][pixel_id][0] > highPixelBoundry:
                    blue += 1
                elif cell[line_id][pixel_id][0] < lowPixelBoundry:
                    black += 1
                elif highPixelBoundry >= cell[line_id][pixel_id][0] >= lowPixelBoundry:
                    white += 1

        if white <= cellProcentage * size:
            conts.append(con)
        if blue >= cellProcentage * size:
            blueCon.append(con)
        if black >= cellProcentage * size:
            blackCon.append(con)

        white = black = blue = 0

    return conts, blueCon, blackCon


    # x_min, y_min, x_max, y_max = cv2.boundingRect(contour)
    # cell = img[y_min:y_min + y_max, x_min:x_min + x_max]
    #
    # # clear background => set to white //255
    # for line_id in range(cell.shape[0]):
    #     for pixel_id in range(cell.shape[1]):
    #         if cell[line_id][pixel_id][0] > 150 or cell[line_id][pixel_id][1] > 150 or cell[line_id][pixel_id][2] > 150:
    #             cell[line_id][pixel_id] = 255
    #
    # return cell


# Version for colors
def extract_cell(contour=None, img=None):
    x_min, y_min, x_max, y_max = cv2.boundingRect(contour)
    cell = img[y_min:y_min + y_max, x_min:x_min + x_max]

    # clear background => set to white //255
    # for line_id in range(cell.shape[0]):
    #     for pixel_id in range(cell.shape[1]):
    #         if cell[line_id][pixel_id][0] > 150 or cell[line_id][pixel_id][1] > 150 or cell[line_id][pixel_id][2] > 150:
    #             cell[line_id][pixel_id] = 255

    return cell


# def background_procentage(cell):
#     white_count = 0
#     cell_count = 0
#     for i in cell:
#         for j in i:
#             if len(j[j.__gt__(158)]) != 0:
#                 white_count += 1
#             cell_count += 1
#     return white_count/cell_count


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
    NOTE: to filter whole image 2x 2DConvolution is required
    :param size: size of gauss kernel ( shape will be (size,1) )
    :param sigma: parameter used to calculate gauss kernel
    :return: returns gauss kernel
    '''
    x = np.arange(size)
    x = x - x[x.shape[0]//2]
    e = (1/np.sqrt(2*np.pi*sigma))
    temp = [ e*np.exp((-i**2)/(2*sigma**2)) for i in x]
    return np.array(temp).reshape(size, 1)


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


def scale(arr, newMax):
    return ((arr-arr.min()) / (arr.max() - arr.min()))*newMax


def Canny(grayImage=None, mask_x=mask_x, mask_y=mask_y, lowBoundry=10.0, highBoundry=30.0, performNMS=False):
    '''
    :param grayImage: input image in gray scale
    :param mask_x: vertical kernel
    :param mask_y: horizontal kernel
    :param lowBoundry: low limit for thresholding
    :param highBoundry: high limit for thresholding
    :return: image with marked edges
    '''
    if grayImage is None:
        print("You have to give at least one argument")
        return

    # # zastosowanie filtru Gaussa w celu ograniczenia szumów
    # gaussKernel = gaussKernelGenerator(5, 1)
    # # convolution with gaussian kernel 2 times(rows and columns) to blure whole image
    # gImage = Convolution2D(Convolution2D(grayImage, gaussKernel, mode='same'), gaussKernel.T, mode="same")

    gaussKernel = gaussianFilterGenerator(3, 5)
    gImage = Convolution2D(grayImage, gaussKernel, mode='same')

    Gx = Convolution2D(gImage, mask_x, mode='same')
    Gy = Convolution2D(gImage, mask_y, mode='same')

    ## gradient magnitude and angle(direction)
    GMag = np.sqrt(Gx**2 + Gy**2)
    Gangle = np.arctan2(Gy, Gx) * (180/np.pi)  ## angle in deg not in radians

    ## Non-maximum Suppression   ######  IN THESE SITUATION  'NMS' MAY GIVE WORSE RESULTS
    if performNMS == True:
        rowNum, colNum = GMag.shape
        result = np.zeros((rowNum, colNum))
        # we want to consider 3x3 matrixes so we do not teke first and last
        for row in range(1, rowNum-1):
            for col in range(1, colNum-1):
                angle = Gangle[row, col]
                if (angle>=0 and angle<=22.5) or (angle<0 and angle>=-22.5) or (angle>=157.5 and angle<=180) \
                    or (angle>=-180 and angle<=-157.5):
                    edge1 = GMag[row-1, col]
                    edge2 = GMag[row+1, col]
                elif (abs(angle)<112.5 and abs(angle)>67.5):
                    edge1 = GMag[row, col - 1]
                    edge2 = GMag[row, col + 1]
                elif (angle>22.5 and angle<=67.5) or (angle>-157.5 and angle<=-112.5):
                    edge1 = GMag[row + 1, col - 1]
                    edge2 = GMag[row - 1, col + 1]
                elif (angle<-22.5 and angle>=-67.5) or (angle>=112.5 and angle<157.5):
                    edge1 = GMag[row - 1, col - 1]
                    edge2 = GMag[row + 1, col + 1]
                else:
                    print("Something went wrong with Non-maximum Suppression")
                    return
                # sprawdzamy po kątach w którą stone idzie nasza krawędz ale do ostatecznego wyniku
                # idą tylko najwyzsze wartosci zeby zostawic cienką krawędz
                if GMag[row, col] > edge1 and GMag[row, col] > edge2:
                    result[row, col] = GMag[row, col]
    else:
        result = GMag

    ## Thresholding
    # chodzi o to ze jest granica górna i dolna i :\
    #     jesli wartosc pixeli jest wieksza niz górna granica to na pewno mamy krawędź
    #     jesli wartość pixeli jest nizsza niz dolna granica to na pewno nie jest to krawędź
    #     jesli wartosc jest pomiedzy granicami to aby byc uznana za czesc krawedzi musi sie
    #     łączyć z pixelami o wartości powyzej górnej granicy czyli z pewną krawędzią

    np.where(result < lowBoundry, result, 0.0)
    np.where(result > highBoundry, result, 255.0)
    neighborPixels = np.zeros((3, 3))
    for i in range(1, result.shape[0] - 1):
        for j in range(1, result.shape[1] - 1):
            if result[i, j] != 0 and result[i, j] != 255:
                neighborPixels = result[i-1:i+1, j-1:j+1]
                if np.any(neighborPixels >= highBoundry):
                    result[i, j] = 255
                else:
                    result[i, j] = 0

    return skimage.morphology.skeletonize(result).astype(np.uint8)#scale(GMag, 255).astype(np.uint8)#result.astype(np.uint8)


def imageThreshold(grayImage, localNeighborhood=61):
    '''
    :param image: input image which will be thresholded
    :param lcoalNeighborhood: size of part of image which will be considered for threshold
    :return: image after thresholding
    '''

    # what if given image is not in gray scale
    if len(grayImage.shape) >= 3:
        image = cv2.cvtColor(grayImage, cv2.COLOR_BGRA2GRAY)

    result = np.zeros_like(grayImage)   # zeros_like creates copy of given array and filled with zeros

    # iteration through every pixel on image
    for row in range(grayImage.shape[0]):
        for col in range(grayImage.shape[1]):
            # Calculate the size of neighborhood
            min_row = max(0, row - localNeighborhood // 2)
            max_row = min(grayImage.shape[0], row + localNeighborhood // 2 + 1)
            min_col = max(0, col - localNeighborhood // 2)
            max_col = min(grayImage.shape[1], col + localNeighborhood // 2 + 1)

            # Extract the neighborhood part of image
            neighborhood = grayImage[min_row:max_row, min_col:max_col]

            # Calculate the local threshold using Gaussian weighted average
            # np.std function to calculate standard deviation(odchylenie standardowe) , equation of
            # np.std() is  np.sqrt(np.mean(abs(a - a.mean())**2))
            # std maybe useful but it is not necessary
            local_threshold = np.mean(neighborhood) #- 0.1 * np.std(neighborhood)

            ## Use previously calculated local threshold
            if grayImage[row, col] > local_threshold:
                result[row, col] = 255
            else:
                result[row, col] = 0

    # apply morphology -- get getStructuringElement składa nam maciez o zadanych wymiarach która bedzie nam potrzebna
    #   -- morphologyEx pozwala wyłapać kontur : MORPH_OPEN czysci tło ze smieci a MORPH_CLOSE czysci kontury komórek
    #      ze smieci
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    return thresh



## test LoG / Canny  -----------------------------------------------------------

# img = cv2.imread('spodnie.jpeg')
img = cv2.imread('Wycinki/resized_wycinek_4_67nieb_82czar.jpg')
# img = cv2.imread('zdj_z_arykułu.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
# Canny(gray)
# plot_photo("From Canny", LoG(gray))
# plot_photo("From Canny", Canny(gray, lowBoundry=1.0, highBoundry=10.0))
# plot_photo("cv2 Canny", cv2.Canny(gray, 100, 200, 10, L2gradient=True))

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



















