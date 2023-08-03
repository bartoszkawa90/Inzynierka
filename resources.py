import numpy as np

from variables import *


# FUNCTIONS / KEYWORDS
def plot_photo(title, image,height, widht):
    ''' Plots photo is given  resolution
        title - title of ploted photo
        image - image to plot
        height - height of ploted photo
        width - width of ploted photo
    '''
    # while True:
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, height, widht)
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    sys.exit()


# @jit(nopython=False)
def delete_incorrect_contours(contours):
    c = tuple([con for con in contours if cv2.contourArea(con) > 2000 or cv2.contourArea(con) < 50])
    contours = c
    SIZE_MAX = contours[0].shape[0]
    size_min = contours[0].shape[0]
    id_min = 0
    ID_MAX = 0
    count = 0

    for con in contours:
        if con.shape[0] < size_min:
            size_min = con.shape[0]
            id_min = count
        if con.shape[0] > SIZE_MAX:
            SIZE_MAX = con.shape[0]
            ID_MAX = count
        count += 1

    largest, smallest = contours[ID_MAX], contours[id_min]
    if (largest.shape[0] > 1000 or smallest.shape[0] < 30):
        conts = tuple([con for con in contours if (con.shape[0] < 1000 and con.shape[0] > 30)])
    elif (smallest.shape[0] < 30):
        conts = tuple([con for con in contours if (con.shape[0] > 30)])
    elif (largest.shape[0] > 1000):
        conts = tuple([con for con in contours if (con.shape[0] < 1000)])
    else:
        conts = contours

    conts = c
    return conts


# Version for colors
# @jit(nopython=False)
def extract_cell(contour=None, img=None, clear=0):
    x_min, y_min, x_max, y_max = cv2.boundingRect(contour)
    cell = img[y_min:y_min + y_max, x_min:x_min + x_max]

    # clear background => set to white //255
    if clear == 1:
        for line_id in range(cell.shape[0]):
            for pixel_id in range(cell.shape[1]):
                if cell[line_id][pixel_id][0] > 150 or cell[line_id][pixel_id][1] > 150 or cell[line_id][pixel_id][2] > 150:
                    cell[line_id][pixel_id] = 255
        # remove single pixels outsize of cell
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # cell = cv2.morphologyEx(cell, cv2.MORPH_OPEN, kernel)

    return cell


def background_procentage(cell):
    white_count = 0
    cell_count = 0
    for i in cell:
        for j in i:
            if len(j[j.__gt__(158)]) != 0:
                white_count += 1
            cell_count += 1
    return white_count/cell_count




# @njit
def Canny():
    pass


def MAC(M1, M2):    # Multiply-accumulate
    result = 0
    for i, j in zip(np.nditer(M1), np.nditer(M2)):
        result += i*j
    return result

# @njit
def Convolution2D(I, K):
    """
    :param I: Input array
    :param K: kernel
    :return result: result = I * K
    """
    result = np.zeros((I.shape[0] + K.shape[0] - 1, I.shape[1] + K.shape[1] - 1))
    fliped_kernel = np.flip(K)
    # for i in range(result.shape[0] * result.shape[1]):
    #     temp = I[0:N, 0:N]
    #     for j, k in zip(temp, fliped_kernel):
    #         result[i] += j*k
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            for m in range(-1, 2):
                sum = 0
                for n in range(-1, 2):
                    sum += fliped_kernel[m, n] * I[i-m, j-n]

                result[i, j] += sum

    return result


# a = np.ones(1) * 2
# b = np.ones((2,2)) * 3
# print(a)
# print(b)
# print(exampleKernel[2:3, 1:3])
# print(exampleArray[0:1, 0:2])
# print(MAC(exampleArray[0:1, 0:2], exampleKernel[2:3, 1:3]))


# TEST
print(Convolution2D(exampleArray, exampleKernel))




















