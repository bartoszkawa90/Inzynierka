from resources import *

#This will display all the available mouse click events
events = [i for i in dir(cv2) if 'EVENT' in i]
print(events)

#This variable we use to store the pixel location
selected_points = []

#click event function
def click_event(event, x, y, flags, param):
    # Show coordinates and RGB
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x,",",y)
        selected_points.append([x, y])
        font = cv2.FONT_HERSHEY_SIMPLEX
        strXY = str(x)+", "+str(y) + " - " + str(img[y,x,0]) +" , "+ str(img[y,x,1]) +" , "+ str(img[y,x,2])
        cv2.putText(img, strXY, (x,y), font, 0.5, (0,255,255), 2)
        cv2.imshow("image", img)

    # Show only RGB
    if event == cv2.EVENT_RBUTTONDOWN:
        blue = img[y, x, 0]
        green = img[y, x, 1]
        red = img[y, x, 2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        strBGR = str(blue)+", "+str(green)+","+str(red)
        cv2.putText(img, strBGR, (x,y), font, 0.5, (0,255,255), 2)
        cv2.imshow("image", img)


#Choose photo
photo = 'Wycinki/wycinek_4_67nieb_82czar.jpg'
img = cv2.imread(photo)

cv2.namedWindow("image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("image", 900, 900)
cv2.imshow("image", 900)

#Mause click
cv2.setMouseCallback("image", click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()
sys.exit()
