# Main program
'''



'''
import cv2

from resources import *

if __name__ == '__main__':
    print("Start")

# Reading an image in default mode
    img = cv2.imread('Zdjecia/NET G2, Ki-67 około 5% --copy.jpg')
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

## METODA K-Means
    # twoDimage = img.reshape((-1, 3))
    # twoDimage = np.float32(twoDimage)
    #
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # K = 4
    # attempts = 10
    #
    # ret, label, center = cv2.kmeans(twoDimage, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    # center = np.uint8(center)
    # res = center[label.flatten()]
    # print(img.shape)
    # result_image = res.reshape((img.shape))
    # cv2.imwrite('Segmentowanie_metoda_k-means.jpg',result_image)


# Łapanie Komórek  /  Tworzenie Kontórów
    assert img is not None, "file could not be read, check with os.path.exists()"

    # contours, hierarchy = cv2.findContours(edged,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Adaptive threshold , raczej widać czarno na białym
    thresh = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 163, 3)
    edged = cv2.Canny(thresh, 10, 250)

    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

    contours_new = tuple(con for con in contours if con.shape[0] > 100)
    # print(len(contours_new))

    cv2.drawContours(gray_img, contours_new, -1, (0, 255, 0), 3)




    # DISPLAY
    plot_photo("Contours",gray_img,900,900)
    # plt.title("Photo")
    # plt.xlabel("X pixel scaling")
    # plt.ylabel("Y pixels scaling")
    # plt.imshow(thresh,cmap='gray')
    # plt.show()

    # sys.exit()



