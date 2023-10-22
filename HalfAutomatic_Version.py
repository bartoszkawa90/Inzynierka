from resources import *
import sys

#This will display all the available mouse click events
events = [i for i in dir(cv2) if 'EVENT' in i]
print(events)

#This variable we use to store the pixel location
selected_points = []

#click event function
def click_event(event, x, y, flags, param):
    # Show coordinates and RGB
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(img.shape) == 3:
            print(x, ",", y, " : ", img[y, x, 0], ' , ', img[y, x, 1], ' , ', img[y, x, 2])
            selected_points.append([x, y])
            font = cv2.FONT_HERSHEY_SIMPLEX
            strXY = str(x)+", "+str(y) + " - " + str(img[y,x,0]) +" , "+ str(img[y,x,1]) +" , "+ str(img[y,x,2])
            cv2.putText(img, strXY, (x,y), font, 0.5, (0,255,255), 2)
            cv2.imshow("image", img)
        else:
            print(x, ",", y, " : ", img[y, x])
            selected_points.append([x, y])
            font = cv2.FONT_HERSHEY_SIMPLEX
            strXY = str(x)+", "+str(y) + " - " + str(img[y, x])
            cv2.putText(img, strXY, (x,y), font, 0.5, (0,255,255), 2)
            cv2.imshow("image", img)

    # Show only RGB
    if event == cv2.EVENT_RBUTTONDOWN:
        if len(img.shape) == 3:
            blue = img[y, x, 0]
            green = img[y, x, 1]
            red = img[y, x, 2]
            font = cv2.FONT_HERSHEY_SIMPLEX
            strBGR = str(blue)+", "+str(green)+","+str(red)
            cv2.putText(img, strBGR, (x,y), font, 0.5, (0,255,255), 2)
            cv2.imshow("image", img)
        else:
            value = img[y, x]
            font = cv2.FONT_HERSHEY_SIMPLEX
            # strBGR = str(blue)+", "+str(green)+","+str(red)
            cv2.putText(img, str(value), (x,y), font, 0.5, (0,255,255), 2)
            cv2.imshow("image", img)


#Choose photo
photo = 'Zdjecia/Szpiczak, Ki-67 ok. 95%.jpg'
# bluecell = 'Cells/blue/cell50.jpg'
blackcell = 'Cells/black/cell55.jpg'
# bluecell = 'Cells/blue/cell37.jpg'
# blackcell = 'Cells/black/cell44.jpg'
img = cv2.imread(blackcell)
print('dwwdaw')
# r, g, b = cv2.split(img)
# img = b
# img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

cv2.namedWindow("image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("image", 900, 900)
cv2.imshow("image", 900)

#Mause click
cv2.setMouseCallback("image", click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()
sys.exit()
