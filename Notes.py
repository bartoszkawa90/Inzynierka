import os
import random

import cv2
import numpy as np
import numpy.lib.stride_tricks

from resources import *
from Document import *

# notes
# if __name__ == '__main__':
#     start_time = time.time()
#
#     a = np.array([1,2,4])
#     b = np.array([2,4,6])
#
#     print(len(b[b.__gt__(2)]))
#
#     print("Finish")
#     print("--- %s seconds ---" % (time.time() - start_time))




# zapisywanie wycinków
# img = cv2.imread('Resized/Zdj_1.jpg')
# res = img[1000:1500, 1800:2300]
# cv2.imwrite("Wycinki/wycinek_5.jpg", res)
# plot_photo("Contours",res,900,900)




# # reshape images

#     ONE
# img = cv2.imread('Zdjecia/Ziarniszczak jajnika, Ki-67 ok. 2%.jpg')
# print(img.shape)
# new_image = cv2.resize(img, (3500, 4666), cv2.INTER_AREA)
# plot_photo('dawd', new_image)
# cv2.imwrite("/Users/bartoszkawa/Desktop/REPOS/GitLab/inzynierka/Wycinki/resized_Wycinek_6.jpg", new_image)


#     ALL
# folder = "/Users/bartoszkawa/Desktop/REPOS/GitLab/inzynierka/Zdjecia/"
# iter = 1
# for img in os.listdir(folder):
#     image = cv2.imread(folder + img)
#     # print(image.__len__())
#     #
#     # print(img)
#     new_image = cv2.resize(image, (3000,4000), cv2.INTER_AREA)
#     cv2.imwrite("/Users/bartoszkawa/Desktop/REPOS/GitLab/inzynierka/Resized/Zdj_" + str(iter) + ".jpg", new_image)
#     iter += 1



# test wydajności
# print("Start")
# start_time = time.time()
#
# img = cv2.imread('Wycinki/resized_wycinek_4_67nieb_82czar.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
# # result = cv2.Canny(gray,  100, 200, 10, L2gradient=True)
# result = Canny(gray)
#
# print("Finish")
# print("--- %s seconds ---" % (time.time() - start_time))


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def generate_data():
    np.random.seed(7)
    x1 = np.random.standard_normal((100,2))*0.6+np.ones((100,2))
    x2 = np.random.standard_normal((100,2))*0.5-np.ones((100,2))
    x3 = np.random.standard_normal((100,2))*0.4-2*np.ones((100,2))+5
    X = np.concatenate((x1,x2,x3),axis=0)

    print(f"shape {len(X)} \n and X {X}")
    return X

def generate_centroids(X,k):
    cx = np.random.rand(k)
    cy = np.random.rand(k)

    centroids = np.zeros((k,2))
    centroids[:,0] = cx
    centroids[:,1] = cy
    return centroids

def plot_data(X,labels,centroids,s):
    plt.figure()
    plt.plot(X[labels==9,0],X[labels==9,1],'k.')
    plt.plot(X[labels==0,0],X[labels==0,1],'r.', label='cluster 1')
    plt.plot(X[labels==1,0],X[labels==1,1],'b.', label='cluster 2')
    plt.plot(X[labels==2,0],X[labels==2,1],'g.', label='cluster 3')
    plt.plot(centroids[:,0],centroids[:,1],'mo',markersize=8, label='centroids')
    plt.legend()
    plt.title(s)
    plt.show()



n = 2
X = generate_data()
k_means = KMeans(n_clusters=n)
model = k_means.fit(X)
centroids = k_means.cluster_centers_
print(centroids)
labels = k_means.labels_

plt.figure()
plt.plot(X[labels==0,0],X[labels==0,1],'r.', label='cluster 1')
plt.plot(X[labels==1,0],X[labels==1,1],'b.', label='cluster 2')
plt.plot(X[labels==2,0],X[labels==2,1],'g.', label='cluster 3')

plt.plot(centroids[:,0],centroids[:,1],'mo',markersize=8, label='centroids')

plt.legend(loc='best')
plt.show()













