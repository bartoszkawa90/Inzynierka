####  Example of multiple threads in Python
def count(name):
    for i in range(1000):
        print(str(i) + name + "\n")

import threading


##  multiple threads in a loop
names = ["thread_1", "thread_2", "thread_3"]

for name in names:
    thread = threading.Thread(target=count, args=(name, ))
    thread.start()

##  Single threads one after another
# name1 = "thread_1"
# thread_1 = threading.Thread(target=count, args=(name1,))
# thread_1.start()
#
# name2 = "thread_2"
# thread_2 = threading.Thread(target=count, args=(name2, ))
# thread_2.start()
#
# name3 = "thread_3"
# thread_3 = threading.Thread(target=count, args=(name3, ))
# thread_3.start()

#-----------------------------------------------------------------------------------------------------------------------

### Stare funkcje jakby sie miały przydać
def gaussianFilterGenerator(size=3, sigma=1):
    X = np.zeros((size, size))
    Y = np.zeros((size, size))
    for i in range(2*size):
        if i < size:
            X[0, i] = Y[i, 0] = -1
        else:
            X[size-1, i-size-1] = Y[i-size-1, size-1] = 1
    print(X, "\n")
    print(Y, "\n")
    result = (1/(2*np.pi*sigma*sigma)) * np.exp(  (-1*(np.power(X, 2) + np.power(Y, 2))) / (2*sigma*sigma)  )
    return result


def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    print(x, "\n")
    print(y, "\n")
    normal = 1 / (2.0 * np.pi * sigma**2)
    g = np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g


def Laplace_Mask(alfa=0):
    '''
    :param alfa: parameter given to create Laplace mask
    :return: returns Laplace mask
    '''
    arr = np.zeros((3, 3))
    arr[0][0:2:2] = arr[0][2] = arr[2][0] = arr[2][2] = alfa/4
    arr[0][1] = arr[1][0] = arr[1][2] = arr[2][1] = (1-alfa)/4
    arr[1][1] = -1
    return (4/(alfa+1))*arr

