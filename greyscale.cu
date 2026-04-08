// simply need to read the image file
// then perform the greyscale conversion on each pixel
// that part is parallelized through a kernel function
// save the resulting image

// obviously pulling from slides where I can (most of the kernel function)


#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

// helper for checking CUDA calls for errors
// without this i am just floundering wondering what happened
#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}

// kernel function to parallelize the converting RGB->grayscale
__global__ void RGBToGrayscale(unsigned char * grayImage, unsigned char * rgbImage, int width, int height) {
    // get the col and row for the thread
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;

    // bounds check
    if (Col < width && Row < height) {
        // get 1D coords for grayscale image
        int grayOffset = Row * width + Col;
        int rgbOffset = grayOffset * 3; // 3 channels (red, green, blue)

        unsigned char r = rgbImage[rgbOffset + 0];
        unsigned char g = rgbImage[rgbOffset + 1];
        unsigned char b = rgbImage[rgbOffset + 2];

        // calculate grayscale value
        grayImage[grayOffset] = 0.21f*r + 0.71f*g + 0.07f*b;
    }
}

int main() {
    // image dimensions
    const int width = 1024;
    const int height = 1024;
    // data sizes for input/output images
    const size_t rgb_size = width * height * 3 * sizeof(unsigned char);
    const size_t gray_size = width * height * sizeof(unsigned char);

    // allocate memory with malloc on the host
    unsigned char *h_rgbImage = (unsigned char *)malloc(rgb_size);
    unsigned char *h_grayImage = (unsigned char *)malloc(gray_size);

    // import image into memory assigned above
    FILE *fptr_in;
    fptr_in = fopen("gc_conv_1024x1024.raw", "rb");
    // helps with debugging (lord knows i need every printout i can get)
    if (fptr_in == NULL) {
        printf("Failed to open input file.\n");
        return 1;
    }
    fread(h_rgbImage, 1, rgb_size, fptr_in);
    fclose(fptr_in);

    // allocate memory on the device (GPU)
    unsigned char *d_rgbImage, *d_grayImage;
    // use the CHECK function on all cuda calls to actually get error output
    CHECK(cudaMalloc((void **)&d_rgbImage, rgb_size));
    CHECK(cudaMalloc((void **)&d_grayImage, gray_size));

    // copy image from host to device. dont just pass pointers. doesnt work
    CHECK(cudaMemcpy(d_rgbImage, h_rgbImage, rgb_size, cudaMemcpyHostToDevice));

    // define block and grid sizes
    dim3 threadsPerBlock(16, 16); // try 8 and 32 for testing
    dim3 numBlocks(width / threadsPerBlock.x, height / threadsPerBlock.y);

    // call kernel function
    RGBToGrayscale<<<numBlocks, threadsPerBlock>>>(d_grayImage, d_rgbImage, width, height);
    CHECK(cudaGetLastError());

    // cuda sync barrier
    CHECK(cudaDeviceSynchronize());

    // copy result from device to host
    CHECK(cudaMemcpy(h_grayImage, d_grayImage, gray_size, cudaMemcpyDeviceToHost));

    // save image to output file
    FILE *fptr_out;
    fptr_out = fopen("gc.raw", "wb");
    if (fptr_out == NULL) {
        printf("Failed to open output file.\n");
        return 1;
    }
    fwrite(h_grayImage, 1, gray_size, fptr_out);
    fclose(fptr_out);

    printf("Successfully converted image to grayscale and saved as gc.raw\n");
    
    // cleanup memory for both device and host
    CHECK(cudaFree(d_rgbImage));
    CHECK(cudaFree(d_grayImage));
    free(h_rgbImage);
    free(h_grayImage);

    return 0;
}