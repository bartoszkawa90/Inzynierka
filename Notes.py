# Notes

# from resources import *
import sys

# cell = cv2.imread('example_cell.jpg')
# gray = cv2.cvtColor(cell, cv2.COLOR_BGRA2GRAY)
#
# lista = [1,2,3,4]
# print(lista)
# lista.clear()
# print(lista)


# check time
import time
start_time = time.time()
print(sys.version)
a = 0
for i in range(10000000):
    a = i +3 *2 - 1.5*i
print(a)
print("--- %s seconds ---" % (time.time() - start_time))


# DISPLAY
#     plot_photo("Contours",cell,900,900)