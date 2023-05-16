# Version_1 : wyciąganie komórek przy użyciu biblioteki OpenCV
 

from resources import *

if __name__ == '__main__':
    print("Start")
    start_time = time.time()

# Reading an image in default mode
    img = cv2.imread('Wycinki/resized_wycinek_4_67nieb_82czar.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

# Finding edges
    # Extracting edges and cells contours from image
    # do adaptive threshold on gray image
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 131, 6)

    # apply morphology -- get getStructuringElement składa nam maciez o zadanych wymiarach która bedzie nam potrzebna
    #   -- morphologyEx pozwala wyłapać kontur : MORPH_OPEN czysci tło ze smieci a MORPH_CLOSE czysci kontury komórek
    #      ze smieci
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    blob = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
    blob = cv2.morphologyEx(blob, cv2.MORPH_CLOSE, kernel)

    edged = cv2.Canny(blob, 100, 200, 10, L2gradient=True)
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


# Extracting and Cleaning?? cells
#     conts = tuple(con for con in contours if con.shape[0].__gt__(40) and con.shape[0].__lt__(1000))
    cells = [extract_cell(c, img, LEAVE_BACKGROUND) for c in contours]


# CALCULATE
    # do dupy taka klasyfikacja
    dark = 0
    light = 0
    iter = 1
    for cell in cells:
        print(iter,'  ', np.mean(cell))
        iter += 1
        if np.mean(cell).__gt__(160):
            light += 1
        else:
            dark += 1

    print('ciemne : {dark} , jasne : {light}'.format(dark=dark, light=light))




####------------------------------------------------------------------------------------------------------------
# Draw Contours
    cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    # cv2.drawContours(img, conts, -1, (0, 255, 0), 3)
    # cv2.imwrite("Part.jpg", img)
    # cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    # cv2.imwrite("Full.jpg", img)


#SAVE Cells in ./Cells
    # iter = 1
    # for c in cells:
    #     cv2.imwrite("Cells/cell"+str(iter)+".jpg", c)
    #     iter += 1


# DISPLAY
    plot_photo("Contours", img, 900, 900)
#     plt.title("Photo")
#     plt.xlabel("X pixel scaling")
#     plt.ylabel("Y pixels scaling")
#     plt.imshow(extracted_cell,cmap='gray')
#     plt.show()



    print("Finish")
    print("--- %s seconds ---" % (time.time() - start_time))
    sys.exit()
