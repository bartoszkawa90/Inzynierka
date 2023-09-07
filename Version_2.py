# Version_2 : wyciąganie komórek przy użyciu własnych fukcji oraz OpenCV


from resources import *

if __name__ == '__main__':
    print("Start")
    start_time = time.time()

    # Reading an image in default mode
    img = cv2.imread('Wycinki/resized_wycinek_4_67nieb_82czar.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

    # # Finding edges
    # # Extracting edges and cells contours from image
    blob = imageThreshold(gray)

    edged = Canny(blob, lowBoundry=1.0, highBoundry=5.0)
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    print(type(contours))

    # Extracting and Cleaning  Cells
    conts = contours_processing(contours)
    FinalContours, blue, black = filterContoursValue(conts, img)
    # cells = [extract_cell(c, img) for c in contours]

    ####------------------------------------------------------------------------------------------------------------
    # Draw Contours
    cv2.drawContours(img, FinalContours, -1, (0, 255, 0), 3)
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
