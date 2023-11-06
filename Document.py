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
import cv2
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



#-----------------------------------------------------------------------------------------------------------------------

### wytrenowanie nowej wersji canny
##  MOZNA TAK ALE NA JAKIEJS CHMURZE JUZ Z TĄ APLIKACJĄ SPRÓBOWAĆ ODPALIĆ ŻEBY TO PRZESZLO BO POCHŁONELO 73GB RAMU


from Klasyfikatory import *
from resources import *

import urllib.request
from time import sleep

def get_set_of_random_images(number=5):
    num = number
    for i in range(num):
        url = "https://source.unsplash.com/random"
        filename = "/Users/bartoszkawa/Desktop/REPOS/GitLab/random_images/image{}.jpg".format(i+3802)#/Users/udosreis/Downloads/Images/image{}.png".format(i)
        urllib.request.urlretrieve(url, filename)
        print(f"Iteracja {i}")
        # sleep(1.5)

def create_set_of_data(func, path_to_images="/Users/bartoszkawa/Desktop/REPOS/GitLab/random_images/", resize=3000, *args):
    list_of_images = [path_to_images + img for img in listdir(f'{path_to_images}') if "DS" not in img]

    # open images
    images, y = [], []
    # print(list_of_images[0])
    # plot_photo(imread(list_of_images[0]))
    iter = 1
    for idx in range(1100):
        image = skresize(imread(list_of_images[idx]), (resize, resize))
        image = scale(image, 255).astype(np.uint8)
        images.append(image)
        # process threshold and Canny algorithm
        blue = split(image)[2]
        blob = imageThreshold(blue)

        imwrite(f"/Users/bartoszkawa/Desktop/REPOS/GitLab/random_images_reshaped/photo"+str(iter)+".jpg", image)
        imwrite(f"/Users/bartoszkawa/Desktop/REPOS/GitLab/random_images_after_Canny/photoCanny"+str(iter)+".jpg", func(blob, 100, 200, 5, L2gradient=False))
        print(f"image {iter} saved")
        iter += 1
        # y.append(func(blob))


from tensorflow.keras.layers import Input
from tensorflow.keras.models import load_model


#### Training Canny model
# def create_model_for_Canny(image, images_path, reference_path):
#     list_of_train_images = [images_path + img for img in listdir(f'{images_path}') if "DS" not in img]
#     list_of_train_labels = [reference_path + img for img in listdir(f'{reference_path}') if "DS" not in img]
#     list_of_train_images = np.sort(list_of_train_images)[:500]
#     list_of_train_labels = np.sort(list_of_train_labels)[:500]

    #
    # X, y = [], []
    # for idx in range(len(list_of_train_images)):
    #     X.append(skresize(cv2.cvtColor(imread(list_of_train_images[idx]), cv2.COLOR_BGR2GRAY), (1000, 1000)))
    #     y.append(skresize(cv2.cvtColor(imread(list_of_train_labels[idx]), cv2.COLOR_BGR2GRAY), (1000, 1000)))
    #     # X.append(imread(list_of_train_images[idx]))
    #     # y.append(imread(list_of_train_labels[idx]))
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    # X_train , X_test, y_train, y_test = tf.constant(X_train), tf.constant(X_test), tf.constant(y_train), tf.constant(y_test)
    #
    # preprocessed_images = []
    # for im in image:
    #     preprocessed_images.append(skresize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (1000, 1000)))
    #
    # model = tf.keras.models.Sequential()

    # # Input layer
    # input_shape = (1000, 1000, 1)
    # model.add(layers.InputLayer(input_shape=input_shape))
    #
    # # Convolutional layers
    # model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    #
    # # Up-sampling and Convolutional layers
    # model.add(layers.UpSampling2D((2, 2)))
    # model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    # model.add(layers.UpSampling2D((2, 2)))
    # model.add(layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same'))
    #
    # # Compile the model
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    #
    # model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
    # loss, accuracy = model.evaluate(X_test, y_test)
    # print(f" CNN model loss : {loss}, and accuracy : {accuracy}")
    # #
    # #
    # model.save('/Users/bartoszkawa/Desktop/REPOS/GitLab/inzynierka/canny.model')
#     model = load_model('canny.model')
#
#     pred = model.predict(tf.constant(preprocessed_images))
#     edged = CVCanny(cv2.resize(scale(pred[0], 255).astype(np.uint8), (3000, 3000)), 100, 200, 5, L2gradient=False)
#     # plot_photo(cv2.resize(scale(pred[0], 255).astype(np.uint8), (3000, 3000)))
#     plot_photo(edged)
# #
# # ## NIE ODPALAC BO WYWALI I TAK TEN PROCESS
# create_model_for_Canny(image=[imread('Zdjecia/Szpiczak, Ki-67 ok. 95%.jpg')],
#                        images_path="/Users/bartoszkawa/Desktop/REPOS/GitLab/random_images_reshaped/",
#                        reference_path="/Users/bartoszkawa/Desktop/REPOS/GitLab/random_images_after_Canny/")
# # edged = CVCanny(blob, 100, 200, 5, L2gradient=False)


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


def kNN(cells, blackCellsPath, blueCellsPath):

    list_of_blue_cells = [blueCellsPath + img for img in os.listdir(f'{blueCellsPath}') if ".DS" not in img]
    list_of_black_cells = [blackCellsPath + img for img in os.listdir(f'{blackCellsPath}') if ".DS" not in img]

    black_cells, blue_cells = [], []
    for cell_id in range(len(list_of_black_cells)):
        black_cells.append(cv2.imread(list_of_black_cells[cell_id]))
    for cell_id in range(len(list_of_blue_cells)):
        blue_cells.append(cv2.imread(list_of_blue_cells[cell_id]))
    print("odczytane referencyjne black", len(list_of_black_cells))
    print("odczytane referencyjne blue", len(list_of_blue_cells))

    cells_RGB = [get_mean_rgb_from_cell(cell) for cell in cells]
    black_RGB = [get_mean_rgb_from_cell(cell) for cell in black_cells]
    blue_RGB = [get_mean_rgb_from_cell(cell) for cell in blue_cells]

    blue_cells_result, black_cells_result = [], []

    for cell_id in range(len(cells)):
        distance_from_nearest_black = distance(cells[cell_id], black_RGB[0])
        distance_from_nearest_blue = distance(cells[cell_id], blue_RGB[0])

        for black in black_RGB:
            if distance(cells[cell_id], black) < distance_from_nearest_black:
                distance_from_nearest_black = distance(cells[cell_id], black)

        for blue in blue_RGB:
            if distance(cells[cell_id], blue) < distance_from_nearest_blue:
                distance_from_nearest_blue = distance(cells[cell_id], blue)

        if distance_from_nearest_blue > distance_from_nearest_black:
            black_cells_result.append(cells[cell_id])
        else:
            blue_cells_result.append(cells[cell_id])

    return black_cells_result, blue_cells_result


import cv2
def edge_detection_HED(img):
    protoPath = "deploy.prototxt"
    modelPath = "hed_pretrained_bsds.caffemodel"
    net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    (H, W) = img.shape[:2]

    mean_pixel_values= np.average(img, axis = (0,1))
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.7, size=(W, H),
                                 mean=(mean_pixel_values[0], mean_pixel_values[1], mean_pixel_values[2]),
                                 # mean=(105.0, 117.0, 123.0),
                                 swapRB= False, crop=False)

    net.setInput(blob)
    hed = net.forward()
    hed = hed[0,0,:,:]  #Drop the other axes
    #hed = cv2.resize(hed[0, 0], (W, H))
    hed = (255 * hed).astype("uint8")  #rescale to 0-255

    return scale(skimage.morphology.skeletonize(hed, method='lee'), 255).astype(np.uint8)


