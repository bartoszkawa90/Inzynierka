import os
import random

import cv2
import numpy as np
import numpy.lib.stride_tricks

from resources import *
# from Document import *
# from Klasyfikatory import *

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




## HED edge detection

# img = imread('Zdjecia/Szpiczak, Ki-67 ok. 95%.jpg')
# img = preprocess(img, xmin=800, xmax=1400, ymin=800, ymax=1400)



# The pre-trained model that OpenCV uses has been trained in Caffe framework
# #Download from the link above
# protoPath = "deploy.prototxt"
# modelPath = "hed_pretrained_bsds.caffemodel"
# net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
#
#
# # load the input image and grab its dimensions, for future use while defining the blob
# # plt.imshow(img)
# (H, W) = img.shape[:2]

# construct a blob out of the input image
#blob is basically preprocessed image.
#OpenCV’s new deep neural network (dnn ) module contains two functions that
#can be used for preprocessing images and preparing them for
#classification via pre-trained deep learning models.
# It includes scaling and mean subtraction
#How to calculate the mean?
# mean_pixel_values= np.average(img, axis = (0,1))
# blob = cv2.dnn.blobFromImage(img, scalefactor=0.7, size=(W, H),
#                              mean=(mean_pixel_values[0], mean_pixel_values[1], mean_pixel_values[2]),
#                              # mean=(105.0, 117.0, 123.0),
#                              swapRB= False, crop=False)
#
# #View image after preprocessing (blob)
# blob_for_plot = np.moveaxis(blob[0,:,:,:], 0,2)
# # plt.imshow(blob_for_plot)
#
#
# # set the blob as the input to the network and perform a forward pass
# # to compute the edges
# net.setInput(blob)
# hed = net.forward()
# hed = hed[0,0,:,:]  #Drop the other axes
# #hed = cv2.resize(hed[0, 0], (W, H))
# hed = (255 * hed).astype("uint8")  #rescale to 0-255
#
# plot_photo(hed)
#

