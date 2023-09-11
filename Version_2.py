# Version_2 : wyciąganie komórek przy użyciu własnych fukcji oraz OpenCV
import cv2

from resources import *

if __name__ == '__main__':
    print("Start")
    start_time = time.time()

    # Reading an image in default mode
    # img = cv2.imread('Zdjecia/Ki-67 60%.jpg')
    img = cv2.imread('Wycinki/resized_Wycinek_4_59nieb_77czar.jpg')
    print("Image ", img.shape)
    # set shape for big/whole images // this works not bad and pretty quick for 3000/4000
        # and works better for 3500/4666 but loooonggg
    if img.shape[0] > 3000 or img.shape[1] > 4000:
        img = cv2.resize(img, (3000, 4000), cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

    # # Finding edges
    # # Extracting edges and cells contours from image
    blob = imageThreshold(gray)

    edged = Canny(blob, lowBoundry=1.0, highBoundry=7.0)
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Extracting and Cleaning  Cells
    conts = contours_processing(contours)
    goodCells = filterWhiteCells(conts, img)  # final contours are all black and blue cells
    blue, black = extract_cells(goodCells, img)
    print('black ', len(black),'blue ', len(blue))
    # cells = [extract_cell(c, img) for c in contours]

    ####------------------------------------------------------------------------------------------------------------
    # Draw Contours
    cv2.drawContours(img, goodCells, -1, (0, 255, 0), 3)
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
