### classification
import os

from resources import *

kmeansClassify(cell):
    pass



if __name__ == '__main__':
    # lista adresów do wycinków
    dir = "Wycinki/"
    list_of_images = [dir + img for img in os.listdir('./{}'.format(dir)) if img.__contains__('res')]

    # img = cv2.imread('Zdjecia/wycinek_5.jpg')
    img = cv2.imread('Wycinki/wycinek_5.jpg')
    img_path = 'Wycinki/wycinek_5.jpg'

    # img = cv2.imread(list_of_images[4])
    cells, conts = Main(img_path, thresholdRange=61, CannyGaussSize=3, CannyGaussSigma=1, CannyLowBoundry=0.1,
         CannyHighBoundry=10.0, CannyUseGauss=True, CannyPerformNMS=True, CannySharpen=False, conSizeLow=None,
         conSizeHigh=None, whiteCellBoundry=10, whiteBlackMode=FILTER_WHITE)

    # Draw Contours
    cv2.drawContours(img, conts, -1, (0, 255, 0), 3)
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
    plot_photo(img_path, img)

    print("Finish")
    print("--- %s seconds ---" % (time.time() - start_time))
    exit()
