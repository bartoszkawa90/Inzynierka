from resources import *

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
# img = cv2.imread('Resized/Zdj_2.jpg')
# res = img[1000:1500, 1800:2300]
# cv2.imwrite("Wycinki/wycinek_5.jpg", res)
# plot_photo("Contours",res,900,900)




# # reshape images

#     ONE
# img = cv2.imread('Wycinki/wycinek_3.jpg')
# new_image = cv2.resize(img, (1200,900), cv2.INTER_AREA)
# cv2.imwrite("/home/bartosz/Desktop/astudia/inzynierka/Wycinki/resized_wycinek_3.jpg", new_image)


#     ALL
# folder = "/home/bartosz/Desktop/astudia/inzynierka/Zdjecia/"
# iter = 1
# for img in os.listdir(folder):
#     image = cv2.imread(folder + img)
#     # print(image.__len__())
#     #
#     # print(img)
#     new_image = cv2.resize(image, (2500,2500), cv2.INTER_AREA)
#     cv2.imwrite("/home/bartosz/Desktop/astudia/inzynierka/Resized/Zdj_" + str(iter) + ".jpg", new_image)
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


# import cv2
# import numpy as np
#
# # read image
# img = cv2.imread('Wycinki/resized_wycinek_4_67nieb_82czar.jpg')
#
# # convert to gray
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# # threshold
# thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)[1]
#
# # morphology edgeout = dilated_mask - mask
# # morphology dilate
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
# dilate = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)
#
# # get absolute difference between dilate and thresh
# edged = cv2.absdiff(dilate, thresh)
#
# # edged = Canny(blob, lowBoundry=1.0, highBoundry=10.0)
# contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
# cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
#
# plot_photo('test', edged)
