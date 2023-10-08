## IMPORTS
import sys

from variables import *
import time
import cv2
# import skimage
from copy import deepcopy
from pprint import pprint
from sys import exit
import random
import sklearn
from Document import *


## CLASSES

class Set():
    def __init__(self, first, second):
        self.first = first
        self.second = second

    def get_firsts(self):
        return self.first

    def get_seconds(self):
        return self.second

    def print(self):
        print(f'first {self.first} second {self.second} \n')

# class Parameters():
#     def __init__(self, img_path, thresholdRange, thresholdMask, CannyGaussSize, CannyGaussSigma, CannyLowBoundry,
#          CannyHighBoundry, CannyUseGauss, CannyPerformNMS, CannySharpen, conSizeLow,
#          conSizeHigh, whiteCellBoundry, blackCellBoundry, whiteBlackMode,
#          returnOriginalContours):
#         self.



## FUNCTIONS / KEYWORDS
def plot_photo(title='None', image=None, height=1500, widht=1500):
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
    for arr in args:
        print(" Array name ::   {}\n Array shape : {} \n {} \n Max : {} \n Min : {} \n ".format('ada', arr.shape, arr,
                                                                                                arr.max(), arr.min()))


def preprocess(img, xmin=0, xmax=None, ymin=0, ymax=None):
    '''
    :param xmin: ->| cuts from left side
    :param xmax:  |<- cuts from right side
    :param ymin:  cuts from the top   // should be 800 for central photos and ~2000 for the one situated on the bottom
    :param ymax:  cuts from the bottom
    '''
    image = cv2.resize(img, (3000, 3000), cv2.INTER_AREA)
    if ymax == None: ymax = img.shape[0]
    if xmax == None: xmax = img.shape[1]
    new = image[ymin:ymin + ymax, xmin:xmin + xmax]

    return new


def contoursProcessing(contours, lowBoundry=15, highBoundry=500, RETURN_ADDITIONAL_INFORMATION=0):
    '''
    Function for finding smallest and largest contours and removing too small and too large contours
    :param contours: contours given to a function for processing
    :param lowBoundry: low limit of contour size
    :param highBoundry: high limit of contour size
    :param RETURN_ADDITIONAL_INFORMATION: if set to 1, the function returns additional information ::
            // returns smallest and largest contour and their IDs
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


def filterWhiteAndBlackCells(contours, img, whiteCellsBoundry=193):
    '''
    :param contours: contours to filter
    :param img: image on which the contours will be applied
    :return: tuple of contours with wrong contours removed
            // wrong contour = contour inside which cell is white
    '''

    conts = []


    #### OLD
    # lower_boundry = np.array([0, 0, 0], dtype="uint8")
    # upper_boundry = np.array([220, 175, 175], dtype="uint8")
    #
    # for con in contours:
    #     if mode == FILTER_WHITE:
    #         ## extract cell from image
    #         x_min, y_min, x_max, y_max = cv2.boundingRect(con)
    #         cell = img[y_min:y_min + y_max, x_min:x_min + x_max]
    #
    #         ## create mask for white cells filtering
    #         mask = cv2.inRange(cell, lower_boundry, upper_boundry)
    #         detected_output = cv2.bitwise_and(cell, cell, mask = mask)
    #
    #         if np.mean(detected_output) > whiteCellsBoundry:
    #             conts.append(con)
    #     if mode == FILTER_BLACK:
    #         ## extract cell from image
    #         conts.append(con)
    #         x_min, y_min, x_max, y_max = cv2.boundingRect(con)
    #         cell = img[y_min:y_min + y_max, x_min:x_min + x_max]
    #
    #         if np.mean(cell) < blackCellsBoundry:
    #             conts = removeContour(conts, con)

    for con in contours:
        # extract cells
        x_min, y_min, x_max, y_max = cv2.boundingRect(con)
        cell = img[y_min:y_min + y_max, x_min:x_min + x_max]

        # filter cells according to mean blue value
        red, green, blue = cv2.split(cell)
        if np.mean(blue) < whiteCellsBoundry:
            conts.append(con)

    return tuple(conts)


def removeContour(contours, contourToRemove):
    '''
    :param contours:  tuple of contours // contour == numpy.ndarray
    :param contourToRemove: contour to remove
    :return: tuple of contours after deleting wrong contour
    '''
    newConts = []
    for con in contours:
        if con.shape != contourToRemove.shape:
            newConts.append(con)
        else:
            if not (con == contourToRemove).all():
                newConts.append(con)
    return newConts



def isOnTheImage(mainImg, img):
    '''
    :param mainImg: main image on which we want wo find second image
    :param img: image which we want to find on first image
    :return: True if the img is the part of mainImg , False if its not
    '''
    # Sprawdź, czy obraz do znalezienia znajduje się w obrazie głównym
    # match template przesuwa obraz do znalezienia po głównym obrazie i sprawdza na ile sie zgadzaja
    #   nastepnie na podstawie dobranego progu mozna sprawdzic gdzie te
    #   wartosci okreslaja ze jest tam ten obraz
    wynik = cv2.matchTemplate(mainImg, img, cv2.TM_CCOEFF_NORMED)
    prog_dopasowania = 0.9  # Prog dopasowania, można dostosować w zależności od potrzeb

    # znajdujemy gdzie funkcja matchTemplate znalazła cos powej progu i jesli lista tych
    #   wartosci jest wieksza niz 0 to mamy to na szukane zdjecie na głównym zdjeciu
    finalList = []
    whereSomethingFound = np.where(wynik >= prog_dopasowania)
    for arr in whereSomethingFound:
        finalList += list(arr)

    return len(finalList) > 0


def filterRepetitions(contours, img):
    '''
    :param contours: (tuple of ndarrays) contours to filter repetitions and wrong cells
    :param img: img on which the contours where found
    :return: tuple of contours after removing wrong cells
    '''
    count = 0
    # filter cells repetitions
    for con in contours:
        for con2 in contours:
            cell1 = extractCell(con, img).get_seconds()
            cell2 = extractCell(con2, img).get_seconds()
            if cell1.shape == cell2.shape:
                if (cell1 == cell2).all():
                    count += 1

        if count >= 2:
            contours = removeContour(contours, con)
        count = 0
    print(len(contours), "liczba konturów po odfiltrowaniu duplikatów")

    # filter images containing more than one cell
    for con in contours:
        for con2 in contours:
            cell1 = extractCell(con, img).get_seconds()
            cell2 = extractCell(con2, img).get_seconds()

            if cell1.shape[0] > cell2.shape[0] \
                    and cell1.shape[1] >= cell2.shape[1] \
                    and isOnTheImage(cell1, cell2):
                contours = removeContour(contours, con)

            elif cell1.shape[0] >= cell2.shape[0] \
                    and cell1.shape[1] > cell2.shape[1] \
                    and isOnTheImage(cell1, cell2):
                contours = removeContour(contours, con)

    print(len(contours), "liczba konturów po odfiltrowaniu zlepek komórek")
    return tuple(contours)


# def extract_cells(contours, img):
#     blue = []
#     black = []
#     for contour in contours:
#         x_min, y_min, x_max, y_max = cv2.boundingRect(contour)
#         cell = img[y_min:y_min + y_max, x_min:x_min + x_max]
#
#         if np.mean(cell) > 162:
#             blue.append(contour)
#         else:
#             black.append(contour)
#
#     return tuple(blue), tuple(black)


# Version for colors
def extractCell(contour=None, img=None):
    x_min, y_min, x_max, y_max = cv2.boundingRect(contour)
    cell = img[y_min:y_min + y_max, x_min:x_min + x_max]
    # cell_dict = {[x_min, x_max, y_min, y_max]: cell}
    cell_set = Set([x_min, x_max, y_min, y_max], cell)

    return cell_set


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


def Canny(grayImage=None, gaussSize=3, gaussSigma=1, mask_x=mask_x, mask_y=mask_y, lowBoundry=10.0, highBoundry=30.0,
          performNMS=False, useGaussFilter=True, sharpenImage=False):
    '''
    :param grayImage: input image in gray scale
    :param mask_x: vertical kernel
    :param mask_y: horizontal kernel
    :param lowBoundry: low limit for thresholding
    :param highBoundry: high limit for thresholding
    :param extractMore: determines values for gauss kernel 1 -> 5, 2.1 0 -> 3, 1.4
        // higher values makes cells less accurate but selects more of them
    :return: image with marked edges
    '''
    if grayImage is None:
        print("You have to give at least one argument")
        return
    if len(grayImage.shape) == 3:
        grayImage = cv2.cvtColor(grayImage, cv2.COLOR_BGRA2GRAY)

    # # zastosowanie filtru Gaussa w celu ograniczenia szumów
    # gaussKernel = gaussKernelGenerator(5, 1)
    # # convolution with gaussian kernel 2 times(rows and columns) to blure whole image
    # gImage = Convolution2D(Convolution2D(grayImage, gaussKernel, mode='same'), gaussKernel.T, mode="same")

    if useGaussFilter:
        # gaussKernel = gaussianFilterGenerator(gaussSize, gaussSigma)
        # gImage = Convolution2D(grayImage, gaussKernel, mode='same')
        gaussKernel = gaussKernelGenerator(gaussSize, gaussSigma)
        gImage = Convolution2D(Convolution2D(grayImage, gaussKernel, mode='same'), gaussKernel, mode='same')
    else:
        gImage = grayImage

    # sharpen image ???
    if sharpenImage:
        gImage = Convolution2D(gImage, sharpen, mode='same')

    Gx = Convolution2D(gImage, mask_x, mode='same')
    Gy = Convolution2D(gImage, mask_y, mode='same')

    ## gradient magnitude and angle(direction)
    GMag = np.sqrt(Gx**2 + Gy**2)
    Gangle = np.arctan2(Gy, Gx) * (180/np.pi)  ## angle in deg not in radians

    ## Non-maximum Suppression   ######  IN THESE SITUATION  'NMS' MAY GIVE WORSE RESULTS
    if performNMS == True:
        print("Performing NMS")
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
                if GMag[row, col] >= edge1 and GMag[row, col] >= edge2:
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


    return scale(result, 255).astype(np.uint8)#scale(result, 255).astype(np.uint8)#scale(GMag, 255).astype(np.uint8)


def imageThreshold(grayImage, localNeighborhood=51, lowLimitForMask=20):
    '''
    :param image: input image which will be thresholded
    :param lcoalNeighborhood: size of part of image which will be considered for threshold
    :return: image after thresholding
    '''

    # what if given image is not in gray scale
    if len(grayImage.shape) >= 3:
        image = cv2.cvtColor(grayImage, cv2.COLOR_BGRA2GRAY)

    result = np.zeros_like(grayImage)   # zeros_like creates copy of given array and filled with zeros

    # filter background and making a mask
    ret, mask = cv2.threshold(grayImage, 20, 255, cv2.THRESH_BINARY)
    grayImage = cv2.bitwise_and(grayImage, grayImage, mask=mask)

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

            local_threshold = np.mean(neighborhood)

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
    result = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    return result


def split(cell=None):
    red = [rgb[0] for rgb in cell]
    green = [rgb[1] for rgb in cell]
    blue = [rgb[2] for rgb in cell]

    return red, green, blue


def Main(img_path, thresholdRange=None, thresholdMaskValue=None, CannyGaussSize=3, CannyGaussSigma=1, CannyLowBoundry=1.0,
         CannyHighBoundry=10.0, CannyUseGauss=True, CannyPerformNMS=True, CannySharpen=False, contourSizeLow=None,
         contourSizeHigh=None, whiteCellBoundry=None, returnOriginalContours=False):
    '''
    '''
    # Reading an image in default mode
    if isinstance(img_path, str):
        img = cv2.imread(img_path)
    else:
        img = img_path

    # preprocessing
    img = preprocess(img, xmin=500, xmax=1300, ymin=500, ymax=1300)
    print(img.shape)

    # change image to grayscale
    red, green, blue = cv2.split(img)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    # plot_photo('test', b)

    ## apply adaptive threshold
    # Oficjalnie najlepsza wartość threshold dla obrazu przyciętego i resized na (3000, 3000) to 51
    if thresholdRange == None and thresholdMaskValue == None:
        blob = imageThreshold(blue)
    elif  thresholdMaskValue == None and thresholdRange != None:
        blob = imageThreshold(blue, localNeighborhood=thresholdRange)
    elif thresholdMaskValue != None and thresholdRange == None:
        blob = imageThreshold(blue, lowLimitForMask=thresholdMask)
    else:
        blob = imageThreshold(blue, localNeighborhood=thresholdRange, lowLimitForMask=thresholdMaskValue )
    # plot_photo('test', blob)

    # # Finding edges
    edged = Canny(blob, gaussSize=CannyGaussSize, gaussSigma=CannyGaussSigma, lowBoundry=CannyLowBoundry,
                  highBoundry=CannyHighBoundry, useGaussFilter=CannyUseGauss, performNMS=CannyPerformNMS,
                  sharpenImage=CannySharpen)
    # edged = cv2.Canny(blob, 10, 200, 5, L2gradient=True)

    contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print("Number of contours at first {}".format(len(contours)))

    # Filtering cells by size
    if contourSizeLow != None and contourSizeHigh != None: conts = contoursProcessing(contours, lowBoundry=contourSizeLow, highBoundry=contourSizeHigh)
    elif contourSizeLow == None and contourSizeHigh != None: conts = contoursProcessing(contours, highBoundry=contourSizeHigh)
    elif contourSizeLow != None and contourSizeHigh == None: conts = contoursProcessing(contours, lowBoundry=contourSizeLow)
    elif contourSizeLow == None and contourSizeHigh == None: conts = contoursProcessing(contours)

    print("Number of contours after size filtering : ", len(conts))

    # filtering cells by color and removing duplicats
    if whiteCellBoundry == None:
        goodConts = filterWhiteAndBlackCells(contours=conts, img=img)
    else:
        goodConts = filterWhiteAndBlackCells(contours=conts, img=img, whiteCellsBoundry=whiteCellBoundry)
    finalConts = goodConts#filterRepetitions(goodConts, img)

    if returnOriginalContours:
        cells = [extractCell(c, img) for c in contours]
        coordinates = [cell.get_firsts() for cell in cells]#list(cells_dicts.keys())
        cells = [cell.get_seconds() for cell in cells]#list(cells_dicts.values())
        return cells, coordinates, contours, img
    else:
        cells = [extractCell(c, img) for c in finalConts]
        coordinates = [cell.get_firsts() for cell in cells]#list(cells_dicts.keys())
        cells = [cell.get_seconds() for cell in cells]#list(cells_dicts.values())
        return cells, coordinates, finalConts, img


def save_cells(cells, coordinates, dir='Cells', name_addition=''):
    # SAVE Cells in ./Cells
    iter = 0
    for cell, coordiante in zip(cells, coordinates):
        print(iter, " ", cell.shape)
        cv2.imwrite(f'{dir}/xmin_{coordiante[0]} xmax_{coordiante[1]} ymin_{coordiante[2]} ymax_{coordiante[3]} cell{iter}{name_addition}.jpg',
                    cell)
        iter += 1


def get_coordinates_from_filename(path):
    return [int(cor[0]) for cor in [ele.split(' ') for ele in image.split('_')][1:5]]


# def test2():
#     print('Hello \n')
#     print(sys.version)
#     return 2
# test2()
#
#
# test()


## test LoG / Canny  -----------------------------------------------------------

# img = cv2.imread('spodnie.jpeg')
# img = cv2.imread('Wycinki/resized_wycinek_4_67nieb_82czar.jpg')
# img = cv2.imread('zdj_z_arykułu.png')

# gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
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







