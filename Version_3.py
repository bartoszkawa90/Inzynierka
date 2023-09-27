### classification
import os
import random

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from resources import *


def distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))


def split(cell=None):
    red = [rgb[0] for rgb in cell]
    green = [rgb[1] for rgb in cell]
    blue = [rgb[2] for rgb in cell]

    return red, green, blue


def kmeansClassify(cells, iterations=3, numOfCenters=2):
    '''
    '''
    # first central points
    centers = [np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]) for _ in range(numOfCenters)]
    # C1 = np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])
    # C2 = np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])
    nearest = [[]] * numOfCenters

    # for cell in cells:
    cell = cells[50]
    for _ in range(iterations):
        for line in cell:
             for pixel in line:
                # print(f'pixel {pixel} center1 {centers[0]} center2 {centers[1]}')
                distances = [distance(center, pixel) for center in centers]
                # print(distances)
                closest_index = distances.index(np.min(distances))
                nearest[closest_index].append(pixel)
                # print(nearest)

        # update centers
        for center, near in zip(centers, nearest):
            print(f'center {center} , near {near.__len__()}')
            center = np.mean(near)
        for near in nearest:
            near.clear()

    print(f'mean {np.mean(cell)} , c1 {centers[0]}  , c2 {centers[1]}')
    print(f' c2 mean {np.mean(centers[0])}  c2 mean {np.mean(centers[1])}')

    # plot
    ax = plt.axes(projection='3d')
    r1, g1, b1 = split(cell)
    # r2, g2, b2 = split(nearC2)
    ax.scatter(r1, g1, b1, color='blue')
    colors = ['red', 'green', 'black']
    for center in centers:
        color = random.choice(colors)
        ax.scatter(center[0], center[1], center[2], color=color)
    # ax.scatter(r2, g2, b2, color='black')
    # ax.scatter(centers[0][0], centers[0][1], centers[0][2], color='green')
    # ax.scatter(centers[1][0], centers[1][1], centers[1][2], color='red')

    plt.xlabel('wartosci R')
    plt.ylabel('wartosci G')
    plt.show()






if __name__ == '__main__':

## IMAGES
    # lista adresów do wycinków
    dir = "Wycinki/"
    list_of_images = [dir + img for img in os.listdir('./{}'.format(dir)) if img.__contains__('res')]

    # img = cv2.imread('Zdjecia/wycinek_5.jpg')
    img_path = 'Wycinki/resized_Wycinek_4_59nieb_77czar.jpg'
    img = cv2.imread(img_path)


## ALGORITHM
    cells, conts = Main(img_path, thresholdRange=61, CannyGaussSize=3, CannyGaussSigma=1, CannyLowBoundry=0.1,
         CannyHighBoundry=10.0, CannyUseGauss=True, CannyPerformNMS=True, CannySharpen=False, conSizeLow=None,
         conSizeHigh=None, whiteCellBoundry=10, whiteBlackMode=FILTER_WHITE)


## CLASSIFICATION
    kmeansClassify(cells, 3, 2)
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
    # cell = cells[40]
    #
    # ax = plt.axes(projection='3d')
    # for x in cell:
    #     print(x)
    # ax.scatter([x[0] for x in cell], [x[1] for x in cell], [x[2] for x in cell])
    # C1 = np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])
    # C2 = np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])
    # ax.scatter(C1[0], C1[1], C1[2], color='green')
    # ax.scatter(C2[0], C2[1], C2[2], color='red')
    # plt.xlabel('wartosci R')
    # plt.ylabel('wartosci G')
    # plt.show()



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
    plot_photo(img_path, cells[50])

    print("Finish")
    print("--- %s seconds ---" % (time.time() - start_time))
    exit()
