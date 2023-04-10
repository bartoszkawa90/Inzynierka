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

# JESLI JUZ MACHINE LEARNING TO JEST COS W OPENCV  +  TensorFlow

def plot_photo(title, image):
    while True:
        cv.imshow(title, image)
        cv.waitKey(0)
        sys.exit()
    cv.destroyAllWindows()

#
# if __name__ == '__main__':
#     print("Start")
#
#     image1 = cv.imread('photo.jpg')
#     image2 = cv.imread('pobrane.png')
#
#     image2 = cv.resize(image2, (800,600))
#
#     image = cv.addWeighted(image1,0.7,image2,0.3,0.0)
#
#     plot_photo("Photo", image)

# Load two images
gekon = cv.imread('photo.jpg')
logo = cv.imread('pobrane.png')
assert gekon is not None, "file could not be read, check with os.path.exists()"
assert logo is not None, "file could not be read, check with os.path.exists()"

# I want to put logo on top-left corner, So I create a ROI
rows,cols,channels = logo.shape
roi = gekon[0:rows, 0:cols]

# Now create a mask of logo and create its inverse mask also
img2gray = cv.cvtColor(logo, cv.COLOR_BGR2GRAY)
ret, mask = cv.threshold(img2gray, 175, 255, cv.THRESH_BINARY_INV)  # daje białe logo
mask_inv = cv.bitwise_not(mask)   # daje bieałe tło loga

# # Wycina ANDem obramówke do logo z wycinka całego zdjecia gekona
img1_bg = cv.bitwise_and(roi,roi,mask = mask_inv)

# # wicina ze zdjecia loga samo logo bez tła
img2_fg = cv.bitwise_and(logo, logo, mask = mask)

# # skleja tył ze fragmentu zdjecia gekona i logo ze zdjecia z logiem
dst = cv.add(img1_bg,img2_fg)
gekon[0:rows, 0:cols] = dst
plot_photo('photo',gekon)


# cv.imshow('res',img1)
# cv.waitKey(0)
# cv.destroyAllWindows()

