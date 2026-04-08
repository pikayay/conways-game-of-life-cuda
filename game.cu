// reusing some cuda code from previous assignment for obvious reasons

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
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

// naive CPU implementation
void cpuVersion(int width, int height, int size, int runtime, uint8_t *input, uint8_t *output) {
    // now for the simulation:
    for (int i = 0; i < runtime; i++) {
        // for every cell:
        for (int j = 0; j < size; j++) {
            uint8_t cell = input[j];
            int neighborAddresses[8] = {j-width-1, j-width, j-width+1, j-1, j+1, j+width-1, j+width, j+width+1};
            int neighbors = 0;
            // for that cell's neighbors:
            // is the cell on the left edge?
            if (j % width == 0) {
                neighborAddresses[0] = -1;
                neighborAddresses[3] = -1;
                neighborAddresses[5] = -1;
            }
            // is the cell on the top edge?
            if (j < width) {
                neighborAddresses[0] = -1;
                neighborAddresses[1] = -1;
                neighborAddresses[2] = -1;
            }
            // is the cell on the right edge?
            if ((j+1) % width == 0) {
                neighborAddresses[2] = -1;
                neighborAddresses[4] = -1;
                neighborAddresses[7] = -1;
            }
            // is the cell on the bottom edge?
            if (j > size - width) {
                neighborAddresses[5] = -1;
                neighborAddresses[6] = -1;
                neighborAddresses[7] = -1;
            }

            // check the valid neighbors:
            for (int n = 0; n < 8; n++) {
                if (neighborAddresses[n] == -1) continue;
                neighbors += input[neighborAddresses[n]];
            }

            // now we can do the sim part
            // is the cell alive?
            if (cell == 1) {
                if      (neighbors < 2) output[j] = 0;
                else if (neighbors < 4) output[j] = 1;
                else if (neighbors > 3) output[j] = 0;
            } // cell's dead, is it easter?
            else {
                if (neighbors == 3) output[j] = 1;
            }
        }
        // copy output grid to input grid to set up for the repeat
        memcpy(input, output, size);
    }
}


__global__ void gpuGlobal(int width, int height, int size, int runtime, uint8_t *input, uint8_t *output) {
    // get the col and row for the thread
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;

    // bounds check
    if (Col < width && Row < height) {
        int j = Col + Row * width;
        uint8_t cell = input[j];
        int neighborAddresses[8] = {j-width-1, j-width, j-width+1, j-1, j+1, j+width-1, j+width, j+width+1};
        int neighbors = 0;

        // for that cell's neighbors:
        // is the cell on the left edge?
        if (j % width == 0) {
            neighborAddresses[0] = -1;
            neighborAddresses[3] = -1;
            neighborAddresses[5] = -1;
        }
        // is the cell on the top edge?
        if (j < width) {
            neighborAddresses[0] = -1;
            neighborAddresses[1] = -1;
            neighborAddresses[2] = -1;
        }
        // is the cell on the right edge?
        if ((j+1) % width == 0) {
            neighborAddresses[2] = -1;
            neighborAddresses[4] = -1;
            neighborAddresses[7] = -1;
        }
        // is the cell on the bottom edge?
        if (j > size - width) {
            neighborAddresses[5] = -1;
            neighborAddresses[6] = -1;
            neighborAddresses[7] = -1;
        }

        // check the valid neighbors:
        for (int n = 0; n < 8; n++) {
            if (neighborAddresses[n] == -1) continue;
            neighbors += input[neighborAddresses[n]];
        }

        // now we can do the sim part
        // is the cell alive?
        if (cell == 1) {
            if      (neighbors < 2) output[j] = 0;
            else if (neighbors < 4) output[j] = 1;
            else if (neighbors > 3) output[j] = 0;
        } // cell's dead, is it easter?
        else {
            if (neighbors == 3) output[j] = 1;
        }
    }
}


int main() {
    // import grid from raw file
    const int width = 1024;
    const int height = 1024;
    const int size = width * height;
    const int runtime = 100;

    // this sizing may be incorrect
    uint8_t *input = (uint8_t *)malloc(size);
    uint8_t *output = (uint8_t *)malloc(size);

    // import image into memory assigned above
    FILE *fptr_in;
    fptr_in = fopen("gc_1024x1024-uint8.raw", "rb");
    // helps with debugging
    if (fptr_in == NULL) {
        printf("Failed to open input file.\n");
        return 1;
    }
    fread(input, 1, size, fptr_in);
    fclose(fptr_in);

    // now input contains the grid.
    
    cpuVersion(width, height, size, runtime, input, output);

    // gpu
    // allocate memory on the device (GPU)
    uint8_t *d_input, *d_output;
    // use the CHECK function on all cuda calls to actually get error output
    CHECK(cudaMalloc((void **)&d_input, size));
    CHECK(cudaMalloc((void **)&d_output, size));
    
    // copy image from host to device.
    CHECK(cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice));

    // define block and grid sizes
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(width / threadsPerBlock.x, height / threadsPerBlock.y);
    
    // call kernel function
    gpuGlobal<<<numBlocks, threadsPerBlock>>>(width, height, size, runtime, d_input, d_output);
    CHECK(cudaGetLastError());

    // cuda sync barrier
    CHECK(cudaDeviceSynchronize());

    // copy result from device to host
    CHECK(cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost));

    // save image to output file
    FILE *fptr_out;
    fptr_out = fopen("gc.raw", "wb");
    if (fptr_out == NULL) {
        printf("Failed to open output file.\n");
        return 1;
    }
    fwrite(output, 1, size, fptr_out);
    fclose(fptr_out);

    printf("all good");

    return 0;
}