####  Example of multiple threads in Python
# def count(name):
#     for i in range(1000):
#         print(str(i) + name + "\n")
#
# import threading
#
#
# ##  multiple threads in a loop
# names = ["thread_1", "thread_2", "thread_3"]
#
# for name in names:
#     thread = threading.Thread(target=count, args=(name, ))
#     thread.start()

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


### Decorators and subprocesses in Python // a way to use pypy3
#------------------------------------------------------------------------------------------------------------------------------------------------

# using multiple python versions
import subprocess
import sys
import os
import cv2
import numpy as np

# if __name__ == "__main__":
#     # Specify the path to the desired Python 3 interpreter
#     python3 = "/Library/Frameworks/Python.framework/Versions/3.11/bin/python3"
#     pypy3 = "/opt/homebrew/bin/pypy3"  # Replace with the actual path
#
#     # Call the Python 3 function using the specified interpreter
#     result = subprocess.run([python3, "-c", "from resources import test; print(test())"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
#     # test = subprocess.check_output()
#     print(result.stdout.strip())

# def run_as_python3_fun():
#     def wrapper(func):
#         python3 = "/Library/Frameworks/Python.framework/Versions/3.11/bin/python3"
#         fun_name = func.__name__
#         # if len(args) > 0:
#         #     name = args[0].__name__
#         # else:
#         #     pass
#
#         sub = subprocess.run([python3, "-c", f"from resources import {fun_name}; print({fun_name}())"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
#         result = sub.stdout.strip()
#         return result
#
#     return wrapper


# def run_with_python_version(python_version):
#     def decorator(func):
#         def wrapper(*args, **kwargs):
#             # Construct the command to run the function with the specified Python version
#             command = [f"{python_version}", "-c", f"from {func.__module__} import {func.__name__}; {func.__name__}(*{args}, **{kwargs})"]
#
#             try:
#                 # Run the command
#                 subprocess.run(command, check=True)
#                 # result = sub.stdout.strip()
#                 # print(result)
#             except subprocess.CalledProcessError as e:
#                 print(f"Error running function with Python {python_version}: {e}")
#
#         return wrapper
#
#     return decorator
#

# @run_with_python_version("/Library/Frameworks/Python.framework/Versions/3.11/bin/python3")
# def test2(a):
#     print(f'Hello {a}\n')
#     print(sys.version)
#     return 2
#
# if __name__ == "__main__":
#     print('before')
#     test2(1)
#
# test2()
# def run_as_python3_fun(func):
#     result = subprocess.run(["/Library/Frameworks/Python.framework/Versions/3.11/bin/python3",
#                              "-c", f"from resources import {str(func.__name__)}; print({str(func.__name__)}())"],
#                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
#     return result.stdout.strip()
#
#
# @run_as_python3_fun
# def test():
#     print('Hello \n')
#     print(sys.version)
#     return 2
#
# run_as_python3_fun(test)



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


def isOnTheImage(mainImg, img):
    '''
    :param mainImg: main image on which we want wo find second image
    :param img: image which we want to find on first image
    :return: True if the img is the part of mainImg , False if its not
    '''
    # Sprawdź, czy obraz do znalezienia znajduje się w obrazie głównym
    # match template przesuwa obraz do znalezienia po głównym obrazie i sprawdza na ile sie zgadzaja
    #   nastepnie na podstawie dobranego progu mozna sprawdzic gdzie te
    #   wartosci okreslaja ze jest tam ten obraz
    result = cv2.matchTemplate(mainImg, img, cv2.TM_CCORR_NORMED) #TM_CCOEFF_NORMED
    # print(result)
    matchingRate = 0.8#0.99  # Prog dopasowania, można dostosować w zależności od potrzeb

    # znajdujemy gdzie funkcja matchTemplate znalazła cos powej progu i jesli lista tych
    #   wartosci jest wieksza niz 0 to mamy to na szukane zdjecie na głównym zdjeciu
    finalList = []
    whereSomethingFound = np.where(result >= matchingRate)
    for arr in whereSomethingFound:
        finalList += list(arr)
    # if len(finalList) > 0:
        # print('dlugosc listy z isontheimage ', len(finalList))
        # print(img.shape, )

    return len(finalList) > 0

