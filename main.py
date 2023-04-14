# Main program
'''

Obrazy :
    image = cv.imread(photo)   ,   pierwsze współrzedne to  y  ,   a drugie to  x
przykład zmiany wymiarów obrazu zeby można bylo go np. dodać z innym
    image2 = cv.resize(image2, (800,600))



'''


# IMPORTS
import cv2 as cv
import sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as mpimg

# JESLI JUZ MACHINE LEARNING TO JEST COS W OPENCV  +  TensorFlow

def plot_photo(title, image,height, widht):
    ''' Plots photo is given  resolution
        title - title of ploted photo
        image - image to plot
        height - height of ploted photo
        width - width of ploted photo
    '''
    # while True:
    cv.namedWindow(title, cv.WINDOW_NORMAL)
    cv.resizeWindow(title, height, widht)
    cv.imshow(title, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    sys.exit()




if __name__ == '__main__':
    print("Start")

    # Reading an image in default mode
    image = cv.imread('Zdjecia/Ki-67 60%.jpg')

    # DISPLAY
    #plot_photo("Photo",image,900,900)
    plt.title("Photo")
    plt.xlabel("X pixel scaling")
    plt.ylabel("Y pixels scaling")
    plt.imshow(image)
    plt.show()

    # sys.exit()




# MOZE SIE PRZYDA

#TODO
# Load two images
# gekon = cv.imread('photo.jpg')
# logo = cv.imread('pobrane.png')
# assert gekon is not None, "file could not be read, check with os.path.exists()"
# assert logo is not None, "file could not be read, check with os.path.exists()"
#
#TODO I want to put logo on top-left corner, So I create a ROI
# rows,cols,channels = logo.shape
# roi = gekon[0:rows, 0:cols]
#
#TODO Now create a mask of logo and create its inverse mask also
# img2gray = cv.cvtColor(logo, cv.COLOR_BGR2GRAY)
# ret, mask = cv.threshold(img2gray, 175, 255, cv.THRESH_BINARY_INV)  # daje białe logo
# mask_inv = cv.bitwise_not(mask)   # daje bieałe tło loga
#
#TODO Wycina ANDem obramówke do logo z wycinka całego zdjecia gekona
# img1_bg = cv.bitwise_and(roi,roi,mask = mask_inv)
#
#TODO wicina ze zdjecia loga samo logo bez tła
# img2_fg = cv.bitwise_and(logo, logo, mask = mask)
#
#TODO skleja tył ze fragmentu zdjecia gekona i logo ze zdjecia z logiem
# dst = cv.add(img1_bg,img2_fg)
# gekon[0:rows, 0:cols] = dst
# plot_photo('photo',gekon)
#
#
# # cv.imshow('res',img1)
# # cv.waitKey(0)
# # cv.destroyAllWindows()

