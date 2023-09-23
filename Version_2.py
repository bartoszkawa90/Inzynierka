# Version_2 : wyciąganie komórek przy użyciu własnych fukcji oraz OpenCV
'''
! - najwazniesze
Co moze sie zmieniac :
! - rozmiar sąsiedniego obszaru w imageThreshold raczej powinien byc domyslny 61 bo działa raczej najlepiej
!!! - do kernela gaussa najlepiej 3,1 lub 5,2  -> 3,1 daje ładniejsze i kompletniejsze komórki ale 5,2 moze dac takie
      krórych 3,1 nie wykryło a powinno
 - filtrowanie kernelem gaussa lub wyostrzanie moze nieco poprawic albo pogorszyc
 - NMS raczej lepiej uzyc moze nie pomóc ale lepiej zeby było
 - w contourProcessing() trzeba dac granice do wywalenia za małych i za duzych konrórów domyslne raczej ok
! - w filterWhiteCells mozna dac inną granice do wywalania białych komórek(niekomórek) ale to ewentualnie mozna
      dac mniejszą granice
'''


from resources import *

if __name__ == '__main__':
    print("Start")
    start_time = time.time()

    # Reading an image in default mode
    # img = cv2.imread('Zdjecia/Ziarniszczak jajnika, Ki-67 ok. 2%.jpg')
    # img = cv2.imread('Wycinki/resized_Wycinek_4_59nieb_77czar.jpg')
    img = cv2.imread('Wycinki/resized_wycinek_3.jpg')
    print("Image ", img.shape)
    # set shape for big/whole images // this works not bad and pretty quick for 3000/4000
        # and works better for 3500/4666 but loooonggg
    if img.shape[0] > 3000 or img.shape[1] > 4000:
        img = cv2.resize(img, (3000, 4000), cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

    # # Finding edges
    # # Extracting edges and cells contours from image
    blob = imageThreshold(gray)

    edged = Canny(blob, gaussSize=3, gaussSigma=1, lowBoundry=0.1, highBoundry=10.0,
                  useGaussFilter=True, performNMS=True, sharpenImage=True)
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))

    # Extracting and Cleaning  Cells
    conts = contoursProcessing(contours)
    goodCells = filterWhiteAndBlackCells(conts, img, mode=FILTER_BLACK, whiteCellsBoundry=23)#, mode=FILTER_WHITE, blackCellsBoundry=10)  # final contours are all black and blue cells
    finalCells = filterRepetitions(goodCells, img)
    print(len(goodCells), len(finalCells))
    cells = [extractCell(c, img) for c in finalCells]

    ####------------------------------------------------------------------------------------------------------------
    # Draw Contours
    cv2.drawContours(img, finalCells, -1, (0, 255, 0), 3)
    # cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    # cv2.imwrite("Part.jpg", img)
    # cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    # cv2.imwrite("Full.jpg", img)

    # SAVE Cells in ./Cells
    # iter = 0
    # for cell in cells:
    #     print(iter, " ", cell.shape)
    #     cv2.imwrite("Cells/cell"+str(iter)+".jpg", cell)
    #     iter += 1

    # DISPLAY
    plot_photo("Contours", img)

    print("Finish")
    print("--- %s seconds ---" % (time.time() - start_time))
    exit()


#  contours to krotka kontórów a każdy to numpyowa tablica punktów ( współrzedne x i y)
