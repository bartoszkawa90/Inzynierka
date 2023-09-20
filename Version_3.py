# Version_2 : wyciąganie komórek przy użyciu własnych fukcji oraz OpenCV
import cv2
from matplotlib import pyplot as plt
from IPython.display import Image, display
def imshow(img, ax=None):
    if ax is None:
        ret, encoded = cv2.imencode(".jpg", img)
        display(Image(encoded))
    else:
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.axis('off')


from resources import *


print("Start")
start_time = time.time()

# Reading an image in default mode
# img = cv2.imread('Zdjecia/Ki-67 60%.jpg')
img = cv2.imread('Wycinki/resized_wycinek_2.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)


thresh = cv2.bitwise_not(imageThreshold(gray))

# noise removal
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
bin_img = cv2.morphologyEx(thresh,
                           cv2.MORPH_OPEN,
                           kernel,
                           iterations=7)


# Create subplots with 1 row and 2 columns
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
# sure background area
sure_bg = cv2.dilate(bin_img, kernel, iterations=3)
# imshow(sure_bg, axes[0,0])
# axes[0, 0].set_title('Sure Background')

# Distance transform
dist = cv2.distanceTransform(bin_img, cv2.DIST_L2, 5)
# imshow(dist, axes[0,1])
# axes[0, 1].set_title('Distance Transform')

#foreground area
ret, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, cv2.THRESH_BINARY)
sure_fg = sure_fg.astype(np.uint8)
# imshow(sure_fg, axes[1,0])
# axes[1, 0].set_title('Sure Foreground')

# unknown area
unknown = cv2.subtract(sure_bg, sure_fg)
# imshow(unknown, axes[1,1])
# axes[1, 1].set_title('Unknown')

# plt.show()


# ret, markers = cv2.connectedComponents(sure_fg)
#
# # Add one to all labels so that background is not 0, but 1
# markers += 1
# # mark the region of unknown with zero
# markers[unknown == 255] = 0

# fig, ax = plt.subplots(figsize=(6, 6))
# ax.imshow(markers, cmap="tab20b")
# ax.axis('off')
# plt.show()

edged = Canny(dist, gaussSize=3, gaussSigma=1, lowBoundry=2.0, highBoundry=10.0,
                  useGaussFilter=1, performNMS=True, sharpenImage=False)

contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))

conts = contoursProcessing(contours, 10, 100)
goodCells = filterWhiteCells(conts, img, 10)  # final contours are all black and blue cells
finalCells = filterRepetitions(goodCells, img)
print(len(goodCells), len(finalCells))
cells = [extractCell(c, img) for c in finalCells]

cv2.drawContours(img, contours, -1, (0, 255, 0), 3)



plot_photo('dwada', img)

