# Main program
 #/*

# MOZNABY POUZYWAC JAKOS LEPSZYCH  ITERATORÓW
# MOZNABY OGARNĄC Multiprocessing  ZEBY DZIAŁALO SZYBCIEJ

 #/*
 

from resources import *

if __name__ == '__main__':
    print("Start")
    start_time = time.time()

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
    if (largest.shape[0] > 1000 or smallest.shape[0] < 100):
        conts = tuple([con for con in contours if (con.shape[0] < 1000 and con.shape[0] > 100)])

    elif (smallest.shape[0] < 100):
        conts = tuple([con for con in contours if (con.shape[0] > 100)])

    elif (largest.shape[0] > 1000):
        conts = tuple([con for con in contours if (con.shape[0] < 1000)])

    else:
        conts = contours

    cells = [extractCell(c, gray, CLEAR_BACKGROUND) for c in conts]
    Cells = [cell for cell in cells if ((np.sum(cell==255)/cell.size)<0.8 and (np.sum(cell<30)/cell.size)<0.8)]


# Draw Contours
    # cv2.drawContours(gray, conts, -1, (0, 255, 0), 3)


#WORK WITH CELLS






#SAVE Cells in ./Cells
    # iter = 1
    # for c in Cells:
    #     cv2.imwrite("Cells/cell"+str(iter)+".jpg", c)
    #     iter += 1

# DISPLAY
#     plot_photo("Contours",Cells[100],900,900)
#     plt.title("Photo")
#     plt.xlabel("X pixel scaling")
#     plt.ylabel("Y pixels scaling")
#     plt.imshow(extracted_cell,cmap='gray')
#     plt.show()

    print("Finish")
    print("--- %s seconds ---" % (time.time() - start_time))
    sys.exit()
