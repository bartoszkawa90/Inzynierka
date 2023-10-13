### Final Version
'''
 KONIEC :: WSZYSTKIE PARAMETRY SĄ NAJWAZNIEJSZE , TAK NA PRAWDE KAZDY Z PARAMETRÓW MOZE POPRAWIC DZIAŁANIE
        WSZYSTKO SIE DA OGARNĄĆ JAK SIĘ DOBRZE DOBIERZE WSZYSTKIE PARAMETRY
 Generalnie najważniejsze do zmian bo może dużo poprawić:
   -  thresholdRange  // dobrze dla wiekszych komorek dac 41/51 mozna dac mniej (25/31) dla malych komórek
   -  CannySharpen    // trzeba po prostu sobie zobaczyć kiedy jest dobrze to dać
   -  whiteCellBoundry  // ważne i tak ok. 194 jest OK ale może być potrzebne troche więcej albo trochę mniej zalezy od zdjecia
                        // niby najlepsze 193/4/5 ale czasem jak jest duzo niebieskich to moze byc nawet 170/80
   -  contour size Low and High też zależy od zdjęcia tak na prawde ale 15 i 500 to raczej wystarczające granice
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
    img_path = list_of_images[3]#'Zdjecia/Histiocytoza z komórek Langerhansa, Ki-67 ok. 15%.jpg'#list_of_images[0]#'Cells/xmin_231 xmax_70 ymin_593 ymax_71 cell47#3.jpg'#'Wycinki/resized_Wycinek_4_59nieb_77czar.jpg'
    print(f"Chosen image {img_path}")
    img = cv2.imread(img_path)


## ALGORITHM

    parameters = Parameters(img_path=img_path, thresholdRange=41, thresholdMaskValue=20, CannyGaussSize=3, CannyGaussSigma=0.7,
                            CannyLowBoundry=0.1, CannyHighBoundry=10.0, CannyUseGauss=True, CannyPerformNMS=False,
                            CannySharpen=False, contourSizeLow=15, contourSizeHigh=500, whiteCellBoundry=193,
                            returnOriginalContours=False)
    segmentation_results = main(parameters)

    # cells, coordinates, conts, finalImage = Main(img_path, thresholdRange=41, thresholdMaskValue=20, CannyGaussSize=3, CannyGaussSigma=0.7, CannyLowBoundry=0.1,
    #      CannyHighBoundry=10.0, CannyUseGauss=True, CannyPerformNMS=False, CannySharpen=False, contourSizeLow=15,
    #      contourSizeHigh=500, whiteCellBoundry=193, returnOriginalContours=False)

    print(len(segmentation_results.contours))
    # conts, smallest, largest, id_min, ID_MAX = contoursProcessing(conts, lowBoundry=15, highBoundry=500, RETURN_ADDITIONAL_INFORMATION=1)
    # print(smallest.shape, largest.shape)


## CLASSIFICATION
    # black_path = "../Reference/black"
    # blue_path = "../Reference/blue"
    # black = []
    # blue = []
    # kNN(segmentation_results.cells[0], black_path, blue_path)
    # black, blue = kmeansClassify(cells)

## verify classification save images
    # SAVE Cells in ./Cells  or   ../Reference
    # save_dir = "./Cells"
    # save_dir = "../Reference/"
    # save_cells(segmentation_results.cells, segmentation_results.coordinates, name_addition=f'#{img_path.split("/")[-1]}', dir=save_dir)


    # DISPLAY
    # Draw Contours
    cv2.drawContours(segmentation_results.image, segmentation_results.contours, -1, (0, 255, 0), 3)
    # śDisplay
    plot_photo('final th ', segmentation_results.image)

    print("Finish")
    print("--- %s seconds ---" % (time.time() - start_time))
    exit()
