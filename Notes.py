import time

import numpy as np

from resources import *

if __name__ == '__main__':
    a = np.array([1,2,4])
    b = np.array([2,4,6])

    print(len(b[b>10]))






# print("Start")
# start_time = time.time()
#
# i = 0
# for a in range(100000000):
#     i = i - 2/(a+1)  + a*4
#
# print(i)
#
# print("Finish")
# print("--- %s seconds ---" % (time.time() - start_time))