# Version_2 : wyciąganie komórek przy użyciu własnych fukcji oraz OpenCV
import cv2

from resources import *

if __name__ == '__main__':
    print("Start")
    start_time = time.time()

    # Reading an image in default mode
    img = cv2.imread('Wycinki/resized_wycinek_4_67nieb_82czar.jpg')
    img = np.where(img == 0, img, 255)
    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

    # Finding edges
    # Extracting edges and cells contours from image
    # do adaptive threshold on gray image
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 151, 2)

    # apply morphology -- get getStructuringElement składa nam maciez o zadanych wymiarach która bedzie nam potrzebna
    #   -- morphologyEx pozwala wyłapać kontur : MORPH_OPEN czysci tło ze smieci a MORPH_CLOSE czysci kontury komórek
    #      ze smieci
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    blob = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
    blob = cv2.morphologyEx(blob, cv2.MORPH_CLOSE, kernel)

    edged = Canny(blob, lowBoundry=1.0, highBoundry=5.0)
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)



    # Extracting and Cleaning?? cells
    # conts = tuple(con for con in contours if con.shape[0].__gt__(4) and con.shape[0].__lt__(200))
    # cells = [extract_cell(c, img, LEAVE_BACKGROUND) for c in contours]


    ####------------------------------------------------------------------------------------------------------------
    # Draw Contours
    cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    # cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    # cv2.imwrite("Part.jpg", img)
    # cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    # cv2.imwrite("Full.jpg", img)

    # SAVE Cells in ./Cells
    # iter = 1
    # for c in cells:
    #     cv2.imwrite("Cells/cell"+str(iter)+".jpg", c)
    #     iter += 1

    # DISPLAY
    plot_photo("Contours", img, 900, 900)

    print("Finish")
    print("--- %s seconds ---" % (time.time() - start_time))
    exit()
