
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
# img = cv2.imread('Zdjecia/Rak surowiczy high-grade, Ki-67 ok. 65%.jpg')
# res = img[7000:9000, 3000:7000]
# cv2.imwrite("Wycinki/wycinek_4.jpg", res)
# plot_photo("Contours",res,900,900)




# # reshape images
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
# i = 0
# for a in range(100000000):
#     i = i - 2/(a+1)  + a*4
#
# print(i)
#
# print("Finish")
# print("--- %s seconds ---" % (time.time() - start_time))