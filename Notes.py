# Notes





a=2
print(a)



# to ML od hindusa ???

# import pandas

# from resources import *

# img = cv2.imread('Zdjecia/NET G2, Ki-67 około 5% --copy.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

# # Jakby tabela w pythonie / excel w pythonie
# df = pandas.DataFrame()

# # Add original pixel values to the data frame as feature #1
# img2 = gray.reshape(-1)
# df['Original Image'] = img2  # to nam wrzuci kolumne o zadanej nazwie do tabeli
# print(df.head())  # pokazuje pierwsze 5 wartosci


# # Add Other features

# # First set Gabor features
# # Generate Gabor features
# num = 1  # To count numbers up in order to give Gabor features a lable in the data frame
# kernels = []  # Create empty list to hold all kernels that we will generate in a loop
# for theta in range(2):  # Define number of thetas. Here only 2 theta values 0 and 1/4 . pi
#     theta = theta / 4. * np.pi
#     for sigma in (1, 3):  # Sigma with values of 1 and 3
#         for lamda in np.arange(0, np.pi, np.pi / 4):  # Range of wavelengths
#             for gamma in (0.05, 0.5):  # Gamma values of 0.05 and 0.5

#                 gabor_label = 'Gabor' + str(num)  # Label Gabor columns as Gabor1, Gabor2, etc.
#                 #                print(gabor_label)
#                 ksize = 5
#                 kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)
#                 kernels.append(kernel)
#                 # Now filter the image and add values to a new column
#                 fimg = cv2.filter2D(img2, cv2.CV_8UC3, kernel)
#                 filtered_img = fimg.reshape(-1)
#                 df[gabor_label] = filtered_img  # Labels columns as Gabor1, Gabor2, etc.
#                 print(gabor_label, ': theta=', theta, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma)
#                 num += 1  # Increment for gabor column label
# #-----------------------------------------------------------------------------------------------------------------------

# print(df.head())






# check time
# import time
# start_time = time.time()
# print(sys.version)
# a = 0
# for i in range(10000000):
#     a = i +3 *2 - 1.5*i
# print(a)
# print("--- %s seconds ---" % (time.time() - start_time))


# DISPLAY
#     plot_photo("Contours",cell,900,900)