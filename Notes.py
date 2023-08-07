
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
# i = 0
# for a in range(100000000):
#     i = i - 2/(a+1)  + a*4
#
# print(i)
#
# print("Finish")
# print("--- %s seconds ---" % (time.time() - start_time))

h = exampleKernel[::-1, ::-1]
# shape[0]  -  | num of rows
# shape[1]  -  - num of columns
x = MALAexample
zeros = np.zeros((x.shape[0] + h.shape[0] + 1, x.shape[1] + h.shape[1] + 1))
result = zeros.copy()
y_shift = h.shape[0] // 2
x_shift = h.shape[1] // 2
zeros[y_shift+1:y_shift + x.shape[0]+1, x_shift+1:x_shift + x.shape[1]+1] = x

print(zeros)
for i in range(y_shift, y_shift + result.shape[0]-2):
    for j in range(x_shift, x_shift + result.shape[1]-2):
        print(h)
        print(zeros[i - y_shift: i + y_shift + 1, j - x_shift:j + x_shift + 1])
        print(h * zeros[i - y_shift: i + y_shift + 1, j - x_shift:j + x_shift + 1])
        print("-----------------------------------------------------------------------------------------------")
        result[i, j] = MAC(h, zeros[i - y_shift: i + y_shift + 1, j - x_shift:j + x_shift + 1])

print("result : \n", result)
print(zeros)

