# imports

# NEW
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

# STANDARD
from resources import *

# for Kmeans
from sklearn.cluster import KMeans

# for KNN
from sklearn.neighbors import KNeighborsClassifier
from skimage.transform import resize as skresize

# for SVC
from skimage.io import imread as skimread
from skimage.transform import resize as skresize
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score


def distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

def get_mean_rgb_from_cell(cell):
    # MY WAY
    # print('extracting rgb from cell', type(cell))
    # print(cell.shape)
    red = [rgb[0] for rgb in cell]
    green = [rgb[1] for rgb in cell]
    blue = [rgb[2] for rgb in cell]

    rmean, gmean, bmean = np.mean(red), np.mean(green), np.mean(blue)

    return [rmean, gmean, bmean]



## ---------------------------------------------------------------------------------------------------------------------
# bez nauczyciela
# kMeans
def kMeans(k_iterations=3, num_of_clusters=2, cells=[]):
    '''
    number od iteration does not really matter
    num of clusters is what matters , cluster with highest mean value is blue and the rest is ponetialy black
    '''
    # start
    black, blue, blackCenter, blueCenter, blueCenter2 = [], [], [], [], []
    cells_RGB = [get_mean_rgb_from_cell(cell) for cell in cells]

    # kMeans
    k_means = KMeans(n_clusters=num_of_clusters, random_state=0)
    model = k_means.fit(cells_RGB)
    centroids = k_means.cluster_centers_

    # classify
    means = [np.mean(center) for center in centroids]
    blueCenter = centroids[means.index(max(means))]

    for cell_id in range(len(cells_RGB)):
        distances = [distance(center, cells_RGB[cell_id]) for center in centroids] #distance(centroids[0], cells_RGB[cell_id]), distance(centroids[1], cells_RGB[cell_id]),
        #              distance(centroids[3], cells_RGB[cell_id])]
        nearest = centroids[distances.index(min(distances))]
        if (nearest == blueCenter).all():# and np.mean(cells[cell_id]) > 170:
            blue.append(cells[cell_id])
        else:
            black.append(cells[cell_id])

    return black, blue, centroids


def simple_color_classyfication(cells):
    black, blue = [], []
    cells_RGB = [get_mean_rgb_from_cell(cell) for cell in cells]
    for cell_id in range(len(cells)):
        if cells_RGB[cell_id][2] > 165:
            blue.append(cells[cell_id])
        else:
            black.append(cells[cell_id])

    return black, blue

# z nauczycielem
def KNN(cells, blackCellsPath, blueCellsPath, k=3):
    list_of_blue_cells = [blueCellsPath + img for img in listdir(f'{blueCellsPath}') if ".DS" not in img]
    list_of_black_cells = [blackCellsPath + img for img in listdir(f'{blackCellsPath}') if ".DS" not in img]

    black_cells, blue_cells, X, y = [], [], [], []
    for cell_id in range(len(list_of_black_cells)):
        black_cells.append(imread(list_of_black_cells[cell_id]))
        y.append(0)
    for cell_id in range(len(list_of_blue_cells)):
        blue_cells.append(imread(list_of_blue_cells[cell_id]))
        y.append(1)

    cells_RGB = [get_mean_rgb_from_cell(cell) for cell in cells]
    black_RGB = [get_mean_rgb_from_cell(cell) for cell in black_cells]
    blue_RGB = [get_mean_rgb_from_cell(cell) for cell in blue_cells]
    X = black_RGB + blue_RGB

    # test
    # test_blue_path = './Reference/blue_test/'
    # test_black_path = './Reference/black_test/'
    # list_of_blue_cells_test = [test_blue_path + img for img in listdir(f'{test_blue_path}') if ".DS" not in img]
    # list_of_black_cells_test = [test_black_path + img for img in listdir(f'{test_black_path}') if ".DS" not in img]
    #
    # black_test, blue_test, X_test, y_test = [], [], [], []
    # for cell_id in range(len(list_of_black_cells_test)):
    #     black_test.append(imread(list_of_black_cells_test[cell_id]))
    #     y_test.append(0)
    # for cell_id in range(len(list_of_blue_cells_test)):
    #     blue_test.append(imread(list_of_blue_cells_test[cell_id]))
    #     y_test.append(1)
    #
    # black_RGB_test = [get_mean_rgb_from_cell(cell) for cell in black_test]
    # blue_RGB_test = [get_mean_rgb_from_cell(cell) for cell in blue_test]
    # X_test = black_RGB_test + blue_RGB_test

    # KNN
    black, blue = [], []
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)
    # print('score', knn.score(X_test, y_test))

    for cell_id in range(len(cells)):
        # print(knn.predict([cells_RGB[cell_id]]))
        if knn.predict([cells_RGB[cell_id]]) == 0:
            black.append(cells[cell_id])
        else:
            blue.append(cells[cell_id])

    return black, blue


def classification_using_svc(cells, blackCellsPath, blueCellsPath, imageResize=15):
    list_of_blue_cells = [blueCellsPath + img for img in listdir(f'{blueCellsPath}') if ".DS" not in img]
    list_of_black_cells = [blackCellsPath + img for img in listdir(f'{blackCellsPath}') if ".DS" not in img]

    # black_cells, blue_cells, X, y, cells_after_preparations = [], [], [], [], []
    X, y, cells_after_preparations = [], [], []
    for cell_id in range(len(list_of_black_cells)):
        cell = skresize(imread(list_of_black_cells[cell_id]), (imageResize, imageResize))
        # black_cells.append(cell.flatten())
        X.append(cell.flatten())
        y.append(0)
    for cell_id in range(len(list_of_blue_cells)):
        cell = skresize(imread(list_of_blue_cells[cell_id]), (imageResize, imageResize))
        # blue_cells.append(cell.flatten())
        X.append(cell.flatten())
        y.append(1)

    # prepare input data
    for cell_id in range(len(cells)):
        cell = skresize(cells[cell_id], (imageResize, imageResize))
        cells_after_preparations.append(cell.flatten())

    # split data for test and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

    # classification
    classifier = SVC()
    parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]
    grid_search = GridSearchCV(classifier, parameters)

    grid_search.fit(X_train, y_train)

    # test performance
    best_extimator = grid_search.best_estimator_
    y_prediction = best_extimator.predict(X_test)
    score = accuracy_score(y_prediction, y_test)
    print(f"{score*100} % of samples were corretly classified")

    # classify cells
    result = best_extimator.predict(cells_after_preparations)
    print(result)
    black, blue = [], []
    for idx in range(len(cells)):
        if result[idx] == 0:
            black.append(cells[idx])
        if result[idx] == 1:
            blue.append(cells[idx])

    return black, blue



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
import tensorflow as tf
from tensorflow.keras import layers


def cnn_classifier(cells, blackCellsPath, blueCellsPath, imageResize=15):
    list_of_blue_cells = [blueCellsPath + img for img in listdir(f'{blueCellsPath}') if "cell" in img]
    list_of_black_cells = [blackCellsPath + img for img in listdir(f'{blackCellsPath}') if "cell" in img]

    # black_cells, blue_cells, X, y, cells_after_preparations = [], [], [], [], []
    X, y, cells_after_preparations = [], [], []
    for cell_id in range(len(list_of_black_cells)):
        cell = skresize(imread(list_of_black_cells[cell_id]), (imageResize, imageResize))
        # black_cells.append(cell.flatten())
        X.append(cell)
        y.append(0)
    for cell_id in range(len(list_of_blue_cells)):
        cell = skresize(imread(list_of_blue_cells[cell_id]), (imageResize, imageResize))
        # blue_cells.append(cell.flatten())
        X.append(cell)
        y.append(1)

    # prepare input data
    for cell_id in range(len(cells)):
        cell = skresize(cells[cell_id], (imageResize, imageResize))
        # cell = tf.constant(cell)
        # cell = tf.reshape()
        cells_after_preparations.append(cell)

    # split data for test and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    X_train , X_test, y_train, y_test = tf.constant(X_train), tf.constant(X_test), tf.constant(y_train), tf.constant(y_test)

    # creating model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(imageResize, imageResize, 3)))
    model.add(MaxPooling2D((3, 3)))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    # model = Sequential([
    #     # layers.Rescaling(1./255, input_shape=(imageResize, imageResize, 3)),
    #     layers.Conv2D(16, 3, padding='same', activation='relu'),
    #     layers.MaxPooling2D(),
    #     layers.Conv2D(32, 3, padding='same', activation='relu'),
    #     layers.MaxPooling2D(),
    #     layers.Conv2D(64, 3, padding='same', activation='relu'),
    #     layers.MaxPooling2D(),
    #     layers.Flatten(),
    #     layers.Dense(128, activation='relu'),
    #     layers.Dense(2)
    # ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f" CNN model loss : {loss}, and accuracy : {accuracy}")

    pred = model.predict(tf.constant(cells_after_preparations))
    print(pred)



    # black, blue = [], []
    # for cell_id in range(len(cells)):
    #     pred = model.predict(tf.constant(cells_after_preparations))
    #     if pred[0][0] > pred[0][1]:
    #         black.append(cells[cell_id])
    #     else:
    #         blue.append(cells[cell_id])
    
    return [],[]#black, blue






### TEST
black_path = "./Reference/black/"
blue_path = "./Reference/blue/"
# black, blue = KNN([imread('./Reference/black_test/xmin_144 xmax_50 ymin_1112 ymax_34 cell53#Szpiczak, Ki-67 ok. 95%.jpg')]
#        , black_path, blue_path, 4)


black, blue = cnn_classifier([imread('./Reference/blue/cell23#new10.jpg'),imread('./Reference/black/cell787.jpg')]
     , black_path, blue_path)




