/*
    Task 4 - Sobel edge detection on the GPU, across one or more PNG images.

    Follows the same approach covered in the workshop material: images are
    decoded/encoded with lodepng as 32-bit RGBA, and the kernel converts each
    pixel to grayscale on the fly rather than pre-converting on the host.

    Two things extended beyond the workshop version:
    - true zero padding at the borders (the brief: "we imagine that the
      space outside the image is filled with zeros"), rather than just
      setting border pixels to black directly
    - multiple images processed in one run, since the brief asks for
      edge detection "across many PNG images", not just one

    usage: ./sobel_edges image1.png [image2.png ...]
    output: <name>_edges.png for each input
*/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include "lodepng.h"

#define CHECK_CUDA(expr)                                                   \
    do {                                                                   \
        cudaError_t status = (expr);                                       \
        if (status != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error (%s:%d): %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(status));       \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

__device__ float grayAt(const unsigned char *img, int w, int h, int x, int y) {
    if (x < 0 || x >= w || y < 0 || y >= h) return 0.0f;
    int idx = (y * w + x) * 4;
    unsigned char r = img[idx], g = img[idx + 1], b = img[idx + 2];
    return 0.299f * r + 0.587f * g + 0.114f * b;
}

__global__ void sobelKernel(const unsigned char *inImg, unsigned char *outImg,
                             unsigned int width, unsigned int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= (int)width || y >= (int)height) return;

    const int gx[3][3] = {{-1,0,1},{-2,0,2},{-1,0,1}};
    const int gy[3][3] = {{-1,-2,-1},{0,0,0},{1,2,1}};

    float Gx = 0.0f, Gy = 0.0f;
    for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {
            float gray = grayAt(inImg, width, height, x + kx, y + ky);
            Gx += gray * gx[ky + 1][kx + 1];
            Gy += gray * gy[ky + 1][kx + 1];
        }
    }

    float magnitude = sqrtf(Gx * Gx + Gy * Gy);
    if (magnitude > 255.0f) magnitude = 255.0f;
    unsigned char edge = (unsigned char)magnitude;

    int index = (y * width + x) * 4;
    outImg[index]     = edge;
    outImg[index + 1] = edge;
    outImg[index + 2] = edge;
    outImg[index + 3] = 255;
}

static void buildOutputName(const char *inputPath, char *outBuf, size_t outBufSize) {
    const char *slash = strrchr(inputPath, '/');
    const char *base = slash ? slash + 1 : inputPath;

    char stem[256];
    strncpy(stem, base, sizeof(stem) - 1);
    stem[sizeof(stem) - 1] = '\0';
    char *dot = strrchr(stem, '.');
    if (dot) *dot = '\0';

    snprintf(outBuf, outBufSize, "%s_edges.png", stem);
}

static void processImage(const char *inFileName) {
    printf("\n--- %s ---\n", inFileName);

    unsigned char *c_inImg = NULL;
    unsigned int c_width = 0, c_height = 0;

    unsigned int err = lodepng_decode32_file(&c_inImg, &c_width, &c_height, inFileName);
    if (err) {
        fprintf(stderr, "failed to decode '%s': %u %s\n", inFileName, err, lodepng_error_text(err));
        return;
    }
    if (c_width == 0 || c_height == 0) {
        fprintf(stderr, "'%s' decoded to zero size, skipping\n", inFileName);
        free(c_inImg);
        return;
    }
    printf("read %u x %u\n", c_width, c_height);

    unsigned int imgSize = 4 * c_width * c_height;
    unsigned char *c_outImg = (unsigned char *)malloc(imgSize);

    unsigned char *d_inImg, *d_outImg;
    CHECK_CUDA(cudaMalloc((void **)&d_inImg, imgSize));
    CHECK_CUDA(cudaMalloc((void **)&d_outImg, imgSize));
    CHECK_CUDA(cudaMemcpy(d_inImg, c_inImg, imgSize, cudaMemcpyHostToDevice));

    dim3 blockSize(16, 16);
    dim3 gridSize((c_width + blockSize.x - 1) / blockSize.x,
                  (c_height + blockSize.y - 1) / blockSize.y);

    printf("grid: %u x %u blocks, %u x %u threads/block\n",
           gridSize.x, gridSize.y, blockSize.x, blockSize.y);

    sobelKernel<<<gridSize, blockSize>>>(d_inImg, d_outImg, c_width, c_height);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(c_outImg, d_outImg, imgSize, cudaMemcpyDeviceToHost));

    char outFileName[300];
    buildOutputName(inFileName, outFileName, sizeof(outFileName));

    err = lodepng_encode32_file(outFileName, c_outImg, c_width, c_height);
    if (err) {
        fprintf(stderr, "failed to encode '%s': %u %s\n", outFileName, err, lodepng_error_text(err));
    } else {
        printf("wrote %s\n", outFileName);
    }

    CHECK_CUDA(cudaFree(d_inImg));
    CHECK_CUDA(cudaFree(d_outImg));
    free(c_inImg);
    free(c_outImg);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s image1.png [image2.png ...]\n", argv[0]);
        return 1;
    }

    for (int i = 1; i < argc; i++) {
        processImage(argv[i]);
    }

    printf("\nfinished - processed %d image(s)\n", argc - 1);
    return 0;
}
