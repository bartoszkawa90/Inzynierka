import numpy as np
import scipy.linalg

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
print("Start")
start_time = time.time()

img = cv2.imread('Wycinki/resized_wycinek_4_67nieb_82czar.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
# result = cv2.Canny(gray,  100, 200, 10, L2gradient=True)
result = Canny(gray)

print("Finish")
print("--- %s seconds ---" % (time.time() - start_time))




# def fun(*args):
#     vars = locals()
#     print(vars)
#     for i in zip(args, vars.keys()):
#         print(i)
#     print(locals())
# arr = np.array([0, 90, 180])
#
# fun(arr)
#
# print(45 * np.round(arr/45))

# print(np.any(MALAexample))
# if np.any(MALAexample == 4):
#     print('dawd')


