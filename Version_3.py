# metoda HED z wykorzystaniem przetrenowanej sieci
import cv2

from resources import *


# Notes
# import cv2
#
# # Load the image
# image_path = "Zdjecia/NET G2, Ki-67 około 5% --copy.jpg"
# image = cv2.imread(image_path)
#
# # Preprocess the image
# # Convert the image to grayscale
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# # Perform edge detection using HED
# hed = cv2.createHEDD()
# edges = hed.detectEdges(gray)
#
# # Display the original image and the detected edges
# cv2.imshow("Original Image", image)
# cv2.imshow("Detected Edges", edges)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Load the pre-trained HED model
model_path = "deploy.prototxt"
weights_path = "hed_pretrained_bsds.caffemodel"
net = cv2.dnn.readNetFromCaffe(model_path, weights_path)

# Load the image
image_path = "Wycinki/wycinek_4_67nieb_82czar.jpg"
image = cv2.imread(image_path)
print(image.shape)
# Preprocess the image
blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(1200, 900), mean=(104.00698793, 116.66876762, 122.67891434), swapRB=False, crop=False)

# Set the blob as input to the network
net.setInput(blob)

# Perform forward pass and get the output
output = net.forward()

# Reshape the output array
output = output[0, 0, :, :]

# Binarize the output (optional)
_, binary = cv2.threshold(output, 0.6, 1, cv2.THRESH_BINARY)

   # współrzędne konturów
# edged = cv2.Canny(binary, 100, 200, 10, L2gradient=True)
# contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
# cv2.drawContours(image, contours, -1, (0, 255, 0), 3)

# Display the original image and the detected edges
# cv2.imshow("Original Image", image)

plot_photo("Contours",binary,1200,900)