# Main program
'''

'''

from resources import *

if __name__ == '__main__':
    print("Start")

# Reading an image in default mode
    img = cv2.imread('Zdjecia/NET G2, Ki-67 około 5% --copy.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

    # Extracting edges and cells contours from image
    # do adaptive threshold on gray image
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 101, 3)

    # apply morphology -- get getStructuringElement składa nam maciez o zadanych wymiarach która bedzie nam potrzebna
    #   -- morphologyEx pozwala wyłapać kontur : MORPH_OPEN czysci tło ze smieci a MORPH_CLOSE czysci kontury komórek
    #      ze smieci
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    blob = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
    blob = cv2.morphologyEx(blob, cv2.MORPH_CLOSE, kernel)

    edged = cv2.Canny(blob, 10, 250)
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# CLEANING too small and too large contours
    largest, smallest = find_extreme_contours(contours)
    if (largest.shape[0] > 1000 or smallest.shape[0] < 55):
        conts = tuple([con for con in contours if (con.shape[0] < 1000 and con.shape[0] > 55)])

    elif (smallest.shape[0] < 55):
        conts = tuple([con for con in contours if (con.shape[0] > 55)])

    elif (largest.shape[0] > 1000):
        conts = tuple([con for con in contours if (con.shape[0] < 1000)])

    else:
        conts = contours

    #Collecting part of image with one cell
    x,y,z,w = cv2.boundingRect(conts[1000])
    cell = img[y:y+w,x:x + z]

    # Draw Contours
    cv2.drawContours(img, conts[1000], -1, (0, 255, 0), 3)



# SAVE
#     cv2.imwrite("example_cell.jpg", cell)
    # cv2.imwrite("doco3_blob.jpg", blob)
    # cv2.imwrite("doco3_contour.jpg", result)



# DISPLAY
    plot_photo("Contours",img,900,900)
    # plt.title("Photo")
    # plt.xlabel("X pixel scaling")
    # plt.ylabel("Y pixels scaling")
    # plt.imshow(thresh,cmap='gray')
    # plt.show()

    # sys.exit()



