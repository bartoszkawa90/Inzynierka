# Main program
'''

Obrazy :
    image = cv.imread(photo)   ,   pierwsze współrzedne to  y  ,   a drugie to  x
przykład zmiany wymiarów obrazu zeby można bylo go np. dodać z innym
    image2 = cv.resize(image2, (800,600))

'''

from resources import *

if __name__ == '__main__':
    print("Start")

# Reading an image in default mode
    img = cv2.imread('Zdjecia/Ki-67 60% --copy.jpg')

    ## METODA K-Means
    twoDimage = img.reshape((-1, 3))
    twoDimage = np.float32(twoDimage)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 4
    attempts = 10

    ret, label, center = cv2.kmeans(twoDimage, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape((img.shape))
    cv2.imwrite('Segmentowanie_metoda_k-means.jpg',result_image)



# DISPLAY
    ##plot_photo("Photo",result_image,900,900)
    # plt.title("Photo")
    # plt.xlabel("X pixel scaling")
    # plt.ylabel("Y pixels scaling")
    # plt.imshow(temp1,cmap='gray')
    # plt.show()

    # sys.exit()



