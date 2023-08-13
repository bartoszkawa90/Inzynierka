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


def gaussianFilterGenerator(size=3, sigma=1):
    X = np.zeros((size, size))
    Y = np.zeros((size, size))
    for i in range(2*size):
        if i < size:
            X[0, i] = Y[i, 0] = -1
        else:
            X[size-1, i-size-1] = Y[i-size-1, size-1] = 1
    print(X, "\n")
    print(Y, "\n")
    result = (1/(2*np.pi*sigma*sigma)) * np.exp(  (-1*(np.power(X, 2) + np.power(Y, 2))) / (2*sigma*sigma)  )
    return result


def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    print(x, "\n")
    print(y, "\n")
    normal = 1 / (2.0 * np.pi * sigma**2)
    g = np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g


# gaussian_kernel(5, 1.4)
# gaussianFilterGenerator(5, 1.4)

print(cv2.getGaussianKernel(5, 1.4))
print(gaussian_kernel(5, 1.4))
print(gaussianFilterGenerator(5, 1.4))







