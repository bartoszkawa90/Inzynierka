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

#
# import cv2
# import numpy as np
#
# # Load your image
# image = cv2.imread('Wycinki/resized_wycinek_4_67nieb_82czar.jpg')
#
# # Convert the image to grayscale
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# # Define the size of the neighborhood (must be an odd number)
# neighborhood_size = 61
#
# # Create an empty output image
# output_image = np.zeros_like(gray_image)
#
# # Iterate through each pixel in the image
# for row in range(gray_image.shape[0]):
#     for col in range(gray_image.shape[1]):
#         # Define the neighborhood boundaries
#         min_row = max(0, row - neighborhood_size // 2)
#         max_row = min(gray_image.shape[0], row + neighborhood_size // 2 + 1)
#         min_col = max(0, col - neighborhood_size // 2)
#         max_col = min(gray_image.shape[1], col + neighborhood_size // 2 + 1)
#
#         # Extract the neighborhood
#         neighborhood = gray_image[min_row:max_row, min_col:max_col]
#
#         # Calculate the local threshold using Gaussian weighted average
#         local_threshold = np.mean(neighborhood) - 0.2 * np.std(neighborhood)
#
#         # Compare the pixel value with the local threshold
#         if gray_image[row, col] > local_threshold:
#             output_image[row, col] = 255
#         else:
#             output_image[row, col] = 0
#
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
# blob = cv2.morphologyEx(output_image, cv2.MORPH_OPEN, kernel)
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
# blob = cv2.morphologyEx(blob, cv2.MORPH_CLOSE, kernel)
#
# edged = cv2.Canny(blob, 100, 200, 10, L2gradient=True)
#
# # Display the resulting binary image
# cv2.imshow('Adaptive Gaussian Threshold', edged)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Load your image
image = cv2.imread('Wycinki/resized_wycinek_4_67nieb_82czar.jpg')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blob = imageThreshold(gray_image, localNeighborhood=51)

plot_photo('dawdad', blob)
