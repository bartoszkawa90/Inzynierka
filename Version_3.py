### Final Version
'''
 Najwazniejsze raczej jest zmienianie filterWhiteCells ( FILTER_WHITE I  FILTER_BLACK ) bo to duzo zmienia
 Sharpen tez jest istotne bo pozwala rzeczywiscie czasem sporo dac ale tez zepsuć, plus thresholdRange
 Generalnie najważniejsze do zmian bo może dużo poprawić:
   -  thresholdRange  // dobrze dla wiekszych komorek dac 61/63 a dla mniejszych 29/30
   -  CannySharpen    // trzeba po prostu sobie zobaczyć kiedy jest dobrze to dać
   -  whiteBlackMode( FILTER_BLACK/FILTER_WHITE )   // FILTER_BLACK jak są rozmazane jasne komorki i je odrzuca
'''

# NEW
import os
import random
import sys

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
    dir = "Zdjecia/"
    # lista zdjec
    list_of_images = [dir + img for img in os.listdir('./{}'.format(dir))]
    # lista zdjec wycinkow
    # list_of_images = [dir + img for img in os.listdir('./{}'.format(dir)) if img.__contains__('res')]
    print(f'\nImages in {dir} directory : ', *list_of_images, sep='\n'), print('\n')

    # img = cv2.imread('Zdjecia/wycinek_5.jpg')
    img_path = list_of_images[0]#'Zdjecia/Histiocytoza z komórek Langerhansa, Ki-67 ok. 15%.jpg'#list_of_images[0]#'Cells/xmin_231 xmax_70 ymin_593 ymax_71 cell47#3.jpg'#'Wycinki/resized_Wycinek_4_59nieb_77czar.jpg'
    img = cv2.imread(img_path)

    # # preprocessing // cut and reshape original image
    # if img_path.split('/')[0] == 'Zdjecia':
    #     # img = preprocess(img, xmin=800, xmax=1300, ymin=800, ymax=1300, resize=True)
    #     print("Image after reshape ", img.shape)
    #     img = cv2.resize(new, (3000, 3000), cv2.INTER_AREA)
    #     img =
    #
    #     if ymax == None: ymax = img.shape[0]
    #     if xmax == None: xmax = img.shape[1]
    #     new = img[ymin:ymin + ymax, xmin:xmin + xmax]
    #     if resize:
    #         final = cv2.resize(new, (3000, 3000), cv2.INTER_AREA)
    #         return final
    #     return new


## ALGORITHM£
    cells, coordinates, conts, finalImage = Main(img_path, thresholdRange=81, thresholdMask=20, CannyGaussSize=3, CannyGaussSigma=1, CannyLowBoundry=0.1,
         CannyHighBoundry=10.0, CannyUseGauss=True, CannyPerformNMS=False, CannySharpen=False, conSizeLow=None,
         conSizeHigh=None, whiteCellBoundry=11, whiteBlackMode=FILTER_WHITE, returnOriginalContours=True)

    conts, smallest, largest, id_min, ID_MAX = contoursProcessing(conts, lowBoundry=15, highBoundry=500, RETURN_ADDITIONAL_INFORMATION=1)
    print(smallest.shape, largest.shape)

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


## SHOWING RESULTS CLASSIFICATION
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


    # DISPLAY
    # Draw Contours
    cv2.drawContours(finalImage, conts, -1, (0, 255, 0), 3)
    # śDisplay
    plot_photo('final', finalImage)

    print("Finish")
    print("--- %s seconds ---" % (time.time() - start_time))
    exit()
