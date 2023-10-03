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
    img_path = 'Wycinki/wycinek_5.jpg'#list_of_images[4]#'Cells/xmin_231 xmax_70 ymin_593 ymax_71 cell47#3.jpg'#'Wycinki/resized_Wycinek_4_59nieb_77czar.jpg'
    img = cv2.imread(img_path)

    # img = preprocess(img)


## ALGORITHM£
    cells, coordinates, conts = Main(img_path, thresholdRange=34, CannyGaussSize=5, CannyGaussSigma=1, CannyLowBoundry=0.1,
         CannyHighBoundry=10.0, CannyUseGauss=False, CannyPerformNMS=True, CannySharpen=False, conSizeLow=15,
         conSizeHigh=None, whiteCellBoundry=15, whiteBlackMode=FILTER_WHITE, returnOriginalContours=False)


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


    # DISPLAY
    # Draw Contours
    cv2.drawContours(img, conts, -1, (0, 255, 0), 3)
    # śDisplay
    plot_photo(img_path, img)

    print("Finish")
    print("--- %s seconds ---" % (time.time() - start_time))
    exit()
