# Main program
'''



'''
import cv2

from resources import *

if __name__ == '__main__':
    print("Start")

# Reading an image in default mode
    img = cv2.imread('Zdjecia/NET G2, Ki-67 około 5% --copy.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)


# Extracting edges and cells contours from image

    # do adaptive threshold on gray image
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 101, 3)

    # apply morphology open then close
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    blob = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    blob = cv2.morphologyEx(blob, cv2.MORPH_CLOSE, kernel)

    edged = cv2.Canny(blob, 10, 250)
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# CLEANING too small and too large contours
    largest,smallest = find_extreme_contours(contours)
    if (largest.shape[0] > 1000 or smallest.shape[0] < 100):
        conts = tuple(con for con in contours if (con.shape[0] < 1000 or con.shape[0] > 100))
    else:
        conts = contours



    # Collecting part of image with one cell
    x,y,z,w = cv2.boundingRect(contours[2000])
    cell = img[y:y+w,x:x + z]

    cv2.drawContours(img, conts, -1, (0, 255, 0), 3)





# raczej bezuzyteczne
    # invert blob
    # blob = (255 - blob)

    # Get contours
    # cnts = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # big_contour = max(cnts, key=cv2.contourArea)

    # # test blob size
    # blob_area_thresh = 1000
    # blob_area = cv2.contourArea(big_contour)
    # if blob_area < blob_area_thresh:
    #     print("Blob Is Too Small")
    #
    # # draw contour
    # result = img.copy()
    # cv2.drawContours(result, [big_contour], -1, (0, 0, 255), 1)

    # zapis
    # cv2.imwrite("doco3_threshold.jpg", thresh)
    # cv2.imwrite("doco3_blob.jpg", blob)
    # cv2.imwrite("doco3_contour.jpg", result)



# DISPLAY
    plot_photo("Contours",cell,900,900)
    # plt.title("Photo")
    # plt.xlabel("X pixel scaling")
    # plt.ylabel("Y pixels scaling")
    # plt.imshow(thresh,cmap='gray')
    # plt.show()

    # sys.exit()



