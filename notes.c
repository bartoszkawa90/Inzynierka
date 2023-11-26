#include <stdio.h>

// Function to perform 2D convolution
void convolution2D(int input[][5], int output[][5], int height, int width, int kernel[][3]) {
    int i, j, m, n, conv_sum;

    // Iterate over each pixel in the input image
    for (i = 1; i < height - 1; i++) {
        for (j = 1; j < width - 1; j++) {
            conv_sum = 0;

            // Apply the convolution kernel
            for (m = 0; m < 3; m++) {
                for (n = 0; n < 3; n++) {
                    conv_sum += input[i - 1 + m][j - 1 + n] * kernel[m][n];
                }
            }

            // Set the result in the output image
            output[i][j] = conv_sum;
        }
    }
}

// Function to print a 2D matrix
void printMatrix(int matrix[][5], int height, int width) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }
}

int main() {
    // Define input image, kernel, and output image
    int input[6][5] = {
            {25, 100, 75, 49, 130},
            {50, 80, 0, 70, 100},
            {5, 10, 20, 30, 0},
            {60, 50, 12, 24, 32},
            {37, 53, 55, 21, 90},
            {140, 17, 0, 23, 222}
    };

    int kernel[3][3] = {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}
    };

    int output[5][5] = {0};

    // Get the dimensions of the input image
    int height = sizeof(input) / sizeof(input[0]);
    int width = sizeof(input[0]) / sizeof(input[0][0]);

    // Perform 2D convolution
    convolution2D(input, output, 6, 5, kernel);

    // Print the input, kernel, and output matrices
    printf("Input Image:\n");
    printMatrix(input, height, width);

    printf("\nKernel:\n");
    printMatrix(kernel, 3, 3);

    printf("\nOutput Image after Convolution:\n");
    printMatrix(output, height, width);

    return 0;
}
