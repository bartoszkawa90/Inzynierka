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
    # for i, j in zip(np.nditer(M1), np.nditer(M2)):
    #     print("i j ", i, j)
    #     result += i*j
    for i in range(M1.shape[0]):
        for j in range(M1.shape[1]):
            print("i j ", i, j)
            result += M1[i][j] * M2[i][j]
    return result

# @njit
def Convolution2D(x, h):
    """
    :param I: Input array
    :param K: kernel
    :return result: result = I * K
    """

    if h.shape[0] != h.shape[1]:
        raise ValueError('Kernel should be a square matrix.')
    elif h.shape[0] % 2 == 0 or h.shape[1] % 2 == 0:
        raise ValueError('Kernel should have a center (odd number or elements in rows and columns).')
    N = h.shape[0]
    result = np.zeros((x.shape[0] + h.shape[0] - 1, x.shape[1] + h.shape[1] - 1))
    # y_shift = h.shape[0] // 2
    # x_shift = h.shape[1] // 2
    h = np.flip(h)

    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            print("x", x[:i+1, :j+1])
            print("h", h[N-1-i:N, N-1-j:N])
            result[i, j] = MAC(x[:i+1, :j+1], h[N-1-i:N, N-1-j:N])
            print(result[i, j])
    # for i in range(result.shape[0]):
    #     for j in range(result.shape[1]):
    #         for m in range(-1, 2):
    #             sum = 0
    #             for n in range(-1, 2):
    #                 sum += fliped_kernel[m, n] * x[i-m, j-n]
    #
    #             result[i, j] += sum

    return result


def conv2(x, h):
    """
    This function performs the 2D convolution.

    Parameters:
        x - An input array.
        h - 2D impulse response (kernel).
    Returns:
        y - 2D convolution of x and h.
    """

    if h.shape[0] != h.shape[1]:
        raise ValueError('Kernel should be a square matrix.')

    elif h.shape[0] % 2 == 0 or h.shape[1] % 2 == 0:
        raise ValueError('Kernel should have a center (odd number or elements in rows and columns).')

    "Your code goes below here."
    h = h[::-1, ::-1]
    y_kernel, x_kernel = h.shape
    y_image, x_image = x.shape
    x_zeros = np.zeros((y_image + y_kernel - 1, x_image + x_kernel - 1))
    y = x_zeros.copy()
    y_shift = y_kernel // 2
    x_shift = x_kernel // 2
    x_zeros[y_shift:y_shift + y_image, x_shift:x_shift + x_image] = x

    for i in range(y_shift, y_shift + y_image):
        for j in range(x_shift, x_shift + x_image):
            y[i, j] = np.sum(h * x_zeros[i - y_shift: i + y_shift + 1, j - x_shift:j + x_shift + 1])

    return y

def convolve2D(image, kernel, padding=0, strides=1):
    # Cross Correlation
    kernel = np.flipud(np.fliplr(kernel))

    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]

    # Shape of Output Convolution
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    output = np.zeros((xOutput, yOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
        print(imagePadded)
    else:
        imagePadded = image

    # Iterate through image
    for y in range(image.shape[1]):
        # Exit Convolution
        if y > image.shape[1] - yKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(image.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > image.shape[0] - xKernShape:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                except:
                    break

    return output

# a = np.ones(1) * 2
# b = np.ones((2,2)) * 3
# print(a)
# print(b)
# print(exampleKernel[2:3, 1:3])
# print("a", exampleKernel[:1, :1])
# print(exampleArray[0:1, 0:2])
# print(MAC(exampleArray[0:1, 0:2], exampleKernel[2:3, 1:3]))


# TEST
print(Convolution2D(exampleArray, exampleKernel))
# print(conv2(exampleArray, exampleKernel))
# print(convolve2D(exampleArray, exampleKernel))
# print(scipy.signal.convolve2d(exampleArray, exampleKernel))




















