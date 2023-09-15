# #Version_1 : wyciąganie komórek przy użyciu biblioteki OpenCV
#
#
# from resources import *
#
# if __name__ == '__main__':
#     print("Start")
#     start_time = time.time()
#
# # Reading an image in default mode
#     img = cv2.imread('Wycinki/resized_Wycinek_4_59nieb_77czar.jpg')
#     gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
#
# # Finding edges
#     # Extracting edges and cells contours from image
#     # do adaptive threshold on gray image
#     thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 187, -1)
#
#     # apply morphology -- get getStructuringElement składa nam maciez o zadanych wymiarach która bedzie nam potrzebna
#     #   -- morphologyEx pozwala wyłapać kontur : MORPH_OPEN czysci tło ze smieci a MORPH_CLOSE czysci kontury komórek
#     #      ze smieci
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#     blob = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
#     blob = cv2.morphologyEx(blob, cv2.MORPH_CLOSE, kernel)
#
#     edged = cv2.Canny(blob, 10, 200, 5, L2gradient=True)
#     contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     # Extracting and Cleaning  Cells
#     conts = contoursProcessing(contours, lowBoundry=35, highBoundry=1000)  ##### DOTĄD JEST NA PEWNO OKKK
#     FinalContours = filterWhiteCells(conts, img)
#     # cells = [extract_cell(c, img) for c in contours]
#
# ####------------------------------------------------------------------------------------------------------------
# # Draw Contours
#     cv2.drawContours(img, FinalContours, -1, (0, 255, 0), 3)
#     # cv2.drawContours(img, conts, -1, (0, 255, 0), 3)
#     # cv2.imwrite("Part.jpg", img)
#     # cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
#     # cv2.imwrite("Full.jpg", img)
#
#
# #SAVE Cells in ./Cells
#     # iter = 1
#     # for c in cells:
#     #     cv2.imwrite("Cells/cell"+str(iter)+".jpg", c)
#     #     iter += 1
#
#
# # DISPLAY
#     plot_photo("Contours", edged, 900, 900)
#
#     print("Finish")
#     print("--- %s seconds ---" % (time.time() - start_time))
#     exit()
