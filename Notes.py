# Notes

from resources import *

cell = cv2.imread('example_cell.jpg')
gray = cv2.cvtColor(cell, cv2.COLOR_BGRA2GRAY)

lista = [1,2,3,4]
print(lista)
lista.clear()
print(lista)



# DISPLAY
#     plot_photo("Contours",cell,900,900)