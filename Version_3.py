### Final Version
'''
 Najwazniejsze raczej jest zmienianie filterWhiteCells ( FILTER_WHITE I  FILTER_BLACK ) bo to duzo zmienia
 Sharpen tez jest istotne bo pozwala rzeczywiscie czasem sporo dac ale tez zepsuć
'''

# NEW
import os
import random
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import sklearn

# STANDARD
from resources import *
from Klasyfikatory import *





if __name__ == '__main__':

## IMAGES
    # lista adresów do wycinków
    dir = "Wycinki/"
    list_of_images = [dir + img for img in os.listdir('./{}'.format(dir)) if img.__contains__('res')]
    print(f'Images in {dir} directory : \n {list_of_images}')

    # img = cv2.imread('Zdjecia/wycinek_5.jpg')
    img_path = list_of_images[2]#'Cells/xmin_231 xmax_70 ymin_593 ymax_71 cell47#3.jpg'#'Wycinki/resized_Wycinek_4_59nieb_77czar.jpg'
    img = cv2.imread(img_path)


## ALGORITHM
    cells, coordinates, conts = Main(img_path, thresholdRange=61, CannyGaussSize=5, CannyGaussSigma=1, CannyLowBoundry=0.1,
         CannyHighBoundry=10.0, CannyUseGauss=True, CannyPerformNMS=True, CannySharpen=False, conSizeLow=None,
         conSizeHigh=None, whiteCellBoundry=11, whiteBlackMode=FILTER_BLACK, returnOriginalContours=False)


    ### MOZNA ROZWAZYC REKURENCJE // CZY MOZE RACZEJ SZUKANIE KONTURÓW NA WYCIETYCH KOMORKACH ALE TO RACZEJ NIE DA RADY
    ###    BO TO ZNOWU TO SAMO I TO JUZ BYŁO
    # conts = list(conts)
    # for cell in cells:
    #     cel, coo, con = Main(cell)
    #     for c in con:
    #         if len(c) > 1:
    #             conts.append(con)
    # # conts = tuple(conts)
    #
    # print(conts, ' \n ', len(conts))

## CLASSIFICATION
      # sklearn.cluster.KMeans()
    # black, blue = kmeansClassify(cells)

## verify classification save images
    # SAVE Cells in ./Cells
    # save_cells(cells, coordinates, name_addition='#3', dir="Cells")


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
    cv2.drawContours(img, conts, -1, (0, 255, 0), 3)
    # cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    # cv2.imwrite("Part.jpg", img)
    # cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    # cv2.imwrite("Full.jpg", img)


    # DISPLAY
    plot_photo(img_path, img)

    print("Finish")
    print("--- %s seconds ---" % (time.time() - start_time))
    exit()
