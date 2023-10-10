import os
import random

import numpy as np
import numpy.lib.stride_tricks

from resources import *
from Document import *

# notes
# if __name__ == '__main__':
#     start_time = time.time()
#
#     a = np.array([1,2,4])
#     b = np.array([2,4,6])
#
#     print(len(b[b.__gt__(2)]))
#
#     print("Finish")
#     print("--- %s seconds ---" % (time.time() - start_time))




# zapisywanie wycinków
# img = cv2.imread('Resized/Zdj_1.jpg')
# res = img[1000:1500, 1800:2300]
# cv2.imwrite("Wycinki/wycinek_5.jpg", res)
# plot_photo("Contours",res,900,900)




# # reshape images

#     ONE
# img = cv2.imread('Zdjecia/Ziarniszczak jajnika, Ki-67 ok. 2%.jpg')
# print(img.shape)
# new_image = cv2.resize(img, (3500, 4666), cv2.INTER_AREA)
# plot_photo('dawd', new_image)
# cv2.imwrite("/Users/bartoszkawa/Desktop/REPOS/GitLab/inzynierka/Wycinki/resized_Wycinek_6.jpg", new_image)


#     ALL
# folder = "/Users/bartoszkawa/Desktop/REPOS/GitLab/inzynierka/Zdjecia/"
# iter = 1
# for img in os.listdir(folder):
#     image = cv2.imread(folder + img)
#     # print(image.__len__())
#     #
#     # print(img)
#     new_image = cv2.resize(image, (3000,4000), cv2.INTER_AREA)
#     cv2.imwrite("/Users/bartoszkawa/Desktop/REPOS/GitLab/inzynierka/Resized/Zdj_" + str(iter) + ".jpg", new_image)
#     iter += 1



# test wydajności
# print("Start")
# start_time = time.time()
#
# img = cv2.imread('Wycinki/resized_wycinek_4_67nieb_82czar.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
# # result = cv2.Canny(gray,  100, 200, 10, L2gradient=True)
# result = Canny(gray)
#
# print("Finish")
# print("--- %s seconds ---" % (time.time() - start_time))


# img1 = cv2.imread('Cells/cell112.jpg')
# IMG = cv2.imread('Wycinki/resized_Wycinek_4_59nieb_77czar.jpg')
# IMG2 = cv2.cvtColor(IMG, cv2.COLOR_BGRA2GRAY)
# IMG2 = Convolution2D(IMG2, edgeDetection, mode="same")
# print(type(IMG2), IMG2.shape)
#
# # def setBlackToWhite(img):
# #     for line in range(img.shape[0]):
# #         for pixel in range(img.shape[1]):
# #             if img[line][pixel] <= 8:
# #                 print(pixel)
# #                 img[line][pixel] = 255
# #     return img
#
# plot_photo('dawd', IMG2)
# dir = "Wycinki/"
#
# list_of_images = [dir + img for img in os.listdir('./{}'.format(dir)) if img.__contains__('res')]
#
# print(list_of_images)

# def isOnImage(main, img):
#     y, x = img.shape[:2]
#     Y, X = main.shape[:2]
#
#     for i in range(Y-y+1):
#         for j in range(X-x+1):
#             # print(f"i {i}, i+y {i+y}, j {j}, j+x {j+x}")
#             cut = main[i:i+y, j:j+x]
#             # plot_photo(cut)
#             if (cut == img).all():
#                 return True
#     return False

def isOnImage(template, image):

    # Perform template matching
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # Define a threshold to determine if the template is found
    threshold = 0.8  # You can adjust this value based on your needs

    # Check if the maximum correlation value is above the threshold
    if max_val >= threshold:
        return True, max_loc
    else:
        return False, None


b = 'Cells/xmin_71 xmax_55 ymin_676 ymax_43 cell331#new.jpg'
s = 'Cells/xmin_72 xmax_53 ymin_680 ymax_34 cell332#new.jpg'
mainCor = get_coordinates_from_filename('xmin_49 xmax_55 ymin_403 ymax_48 cell332#new')
imgCor = get_coordinates_from_filename('xmin_50 xmax_53 ymin_407 ymax_39 cell333#new')
print(mainCor, imgCor)
big = cv2.imread(b)
small = cv2.imread(s)
print(big.shape, small.shape)


# print(isOnImage(big, small))

a = np.array([[[1, 2, 3]]])
print(np.atleast_3d(a))
