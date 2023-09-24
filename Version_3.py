### classification
import os
import random

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from resources import *


def distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))
def kmeansClassify(cells):
    '''
    '''
    for cell in cells:
        R = [rgb[0] for rgb in cell]
        G = [rgb[1] for rgb in cell]
        B = [rgb[2] for rgb in cell]

        C1 = np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])
        C2 = np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])
        nearC1 = []
        nearC2 = []

        for pixel in cell:







if __name__ == '__main__':

## IMAGES
    # lista adresów do wycinków
    dir = "Wycinki/"
    list_of_images = [dir + img for img in os.listdir('./{}'.format(dir)) if img.__contains__('res')]

    # img = cv2.imread('Zdjecia/wycinek_5.jpg')
    img = cv2.imread('Wycinki/resized_Wycinek_4_59nieb_77czar.jpg')
    img_path = 'Wycinki/resized_Wycinek_4_59nieb_77czar.jpg'

    # img = cv2.imread(list_of_images[4])
    # img_path = list_of_images[4]

## ALGORITHM
    cells, conts = Main(img_path, thresholdRange=61, CannyGaussSize=3, CannyGaussSigma=1, CannyLowBoundry=0.1,
         CannyHighBoundry=10.0, CannyUseGauss=True, CannyPerformNMS=True, CannySharpen=False, conSizeLow=None,
         conSizeHigh=None, whiteCellBoundry=10, whiteBlackMode=FILTER_WHITE)


## CLASSIFICATION
    # black, blue = kmeansClassify(cells)

## verify classification save images
    # iter = 0
    # for blackCell, blueCell in zip(black, blue):
    #     # print(iter, " ", cell.shape)
    #     cv2.imwrite("Cells/black/cell"+str(iter)+".jpg", blackCell)
    #     cv2.imwrite("Cells/blue/cell"+str(iter)+".jpg", blueCell)
    #     iter += 1


## SHOWING RESULTS
    # Plot pixels of image
    cell = cells[40]
    ax = plt.axes(projection = '3d')
    for x in cell:
        print(x)
    ax.scatter([x[0] for x in cell], [x[1] for x in cell], [x[2] for x in cell])
    plt.xlabel('wartosci R')
    plt.ylabel('wartosci G')
    plt.show()



    # Draw Contours
    # cv2.drawContours(img, conts, -1, (0, 255, 0), 3)
    # cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    # cv2.imwrite("Part.jpg", img)
    # cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    # cv2.imwrite("Full.jpg", img)

    # SAVE Cells in ./Cells
    # iter = 0
    # for cell in cells:
    #     print(iter, " ", cell.shape)
    #     cv2.imwrite("Cells/cell"+str(iter)+".jpg", cell)
    #     iter += 1

    # DISPLAY
    plot_photo(img_path, cell)

    print("Finish")
    print("--- %s seconds ---" % (time.time() - start_time))
    exit()
