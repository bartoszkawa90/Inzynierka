# imports

# NEW
import os
import random
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

# STANDARD
from resources import *
from Klasyfikatory import *


# additions


def distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

def split(cell=None):
    red = [rgb[0] for rgb in cell]
    green = [rgb[1] for rgb in cell]
    blue = [rgb[2] for rgb in cell]

    return red, green, blue


## ----------------------------------------------------------------------------------------------------------

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

def kNN():
    pass



# moja werscja Kmeans // do dorobienia ewentualnie
def kmeansClassify(cells, iterations=3, numOfCenters=2):
    '''
    '''
    # first central points
    centers = [np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]) for _ in range(numOfCenters)]
    print('first centers ', centers)
    nearest = [[]] * numOfCenters

    # for cell in cells:
    cell = cells[50]

    distances = []
    for _ in range(iterations):
        for line in cell:
             for pixel in line:
                for center in centers:
                    distances.append(distance(center, pixel))
                # distances = [distance(center, pixel) for center in centers]
                print('distances ', distances)
                # print(distances)
                closest_index = distances.index(np.min(distances))
                print(closest_index)
                nearest[closest_index].append(pixel)
                # print(nearest)
                distances.clear()

        print('  nearest ', nearest.__len__(), nearest[0].__len__(), nearest[1].__len__())
        # update centers
        new_centers = []
        for near in nearest:
            r, g, b = split(near)
            new_centers.append(np.array([np.mean(r), np.mean(g), np.mean(b)]))
            print('new center ', new_centers)
        centers = deepcopy(new_centers)
        new_centers.clear()
        for near in nearest:
            near.clear()
        print(f'centers {centers} new_centers {new_centers}  ', centers is new_centers)


        # for center, near in zip(centers, nearest):
        #     print(f'center {center} , near {near.__len__()}')
        #     new_centers.append(np.mean(near))
        # centers = new_centers
        # # clear
        # new_centers.clear()
        # print(centers)
        # for near in nearest:
        #     near.clear()

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
