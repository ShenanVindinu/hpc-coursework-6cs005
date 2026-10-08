/*
    Task 3 - cracking a file of passwords on the GPU.

    Background: each password in the input file was produced by running a
    short 4-character raw password (2 lowercase letters then 2 digits) through
    a one-way scrambling function. There's no way to reverse that function
    directly, but the space of possible raw passwords is small enough
    (26 * 26 * 10 * 10 = 67,600) that brute force is practical - one GPU
    thread takes one target password and works through every possible raw
    candidate until it finds the one that scrambles into a match.

    One thread is assigned to exactly one target password - never more than
    one - so the grid/block layout is worked out from how many passwords are
    in the file, not from the search space size.

    usage: ./password_cracker <file of scrambled passwords>
    output: decrypted.txt
*/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>

#define PLAIN_LEN 4
#define CIPHER_LEN 10

#define CHECK_CUDA(expr)                                                   \
    do {                                                                   \
        cudaError_t status = (expr);                                       \
        if (status != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error (%s:%d): %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(status));       \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

// Same scrambling logic as the version supplied with the assignment, just
// rewritten to take an output buffer as a parameter rather than keeping a
// static buffer internally - a static buffer would be shared (and corrupted)
// across every thread running this at once on the GPU.
__device__ void encodeOnDevice(const char *plain, char *cipher) {
    cipher[0] = plain[0] + 2;
    cipher[1] = plain[0] - 2;
    cipher[2] = plain[0] + 1;
    cipher[3] = plain[1] + 3;
    cipher[4] = plain[1] - 3;
    cipher[5] = plain[1] - 1;
    cipher[6] = plain[2] + 2;
    cipher[7] = plain[2] - 2;
    cipher[8] = plain[3] + 4;
    cipher[9] = plain[3] - 4;
    cipher[10] = '\0';

    for (int i = 0; i < 10; i++) {
        if (i < 6) {
            if (cipher[i] > 122) cipher[i] = (cipher[i] - 122) + 97;
            else if (cipher[i] < 97) cipher[i] = (97 - cipher[i]) + 97;
        } else {
            if (cipher[i] > 57) cipher[i] = (cipher[i] - 57) + 48;
            else if (cipher[i] < 48) cipher[i] = (48 - cipher[i]) + 48;
        }
    }
}

// one thread, one target password. `targets` holds every scrambled password
// back to back (CIPHER_LEN+1 bytes apiece); `answers` is where this thread
// writes the raw password it finds (PLAIN_LEN+1 bytes apiece).
__global__ void bruteForceKernel(const char *targets, int targetCount, char *answers) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= targetCount) return; // last block will have some idle threads if targetCount isn't a perfect square

    const char *myTarget = targets + i * (CIPHER_LEN + 1);
    char *myAnswer = answers + i * (PLAIN_LEN + 1);

    char guess[PLAIN_LEN + 1];
    char scrambled[CIPHER_LEN + 1];
    guess[PLAIN_LEN] = '\0';

    for (char a = 'a'; a <= 'z'; a++) {
        for (char b = 'a'; b <= 'z'; b++) {
            for (char c = '0'; c <= '9'; c++) {
                for (char d = '0'; d <= '9'; d++) {
                    guess[0] = a; guess[1] = b; guess[2] = c; guess[3] = d;

                    encodeOnDevice(guess, scrambled);

                    bool same = true;
                    for (int k = 0; k < CIPHER_LEN; k++) {
                        if (scrambled[k] != myTarget[k]) { same = false; break; }
                    }

                    if (same) {
                        myAnswer[0] = a; myAnswer[1] = b; myAnswer[2] = c; myAnswer[3] = d;
                        myAnswer[4] = '\0';
                        return;
                    }
                }
            }
        }
    }

    // the scrambling function maps every raw password to a different output
    // (verified separately), so this point shouldn't be reached for a
    // genuine input file - but better to leave a visible marker than an
    // uninitialised buffer if something unexpected shows up
    myAnswer[0] = myAnswer[1] = myAnswer[2] = myAnswer[3] = '?';
    myAnswer[4] = '\0';
}

// reads every line of the file into one flat buffer on the host, growing it
// as needed since we don't know the password count ahead of time
static char *readTargetsFromFile(const char *path, int *countOut) {
    FILE *fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "couldn't open '%s'\n", path);
        exit(1);
    }

    int cap = 1024, n = 0;
    char *buf = (char *)malloc(cap * (CIPHER_LEN + 1));

    char line[256];
    while (fgets(line, sizeof(line), fp)) {
        size_t len = strlen(line);
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) {
            line[--len] = '\0';
        }
        if (len == 0) continue;

        if ((int)len != CIPHER_LEN) {
            fprintf(stderr, "skipping line with odd length (%zu): '%s'\n", len, line);
            continue;
        }

        if (n == cap) { cap *= 2; buf = (char *)realloc(buf, cap * (CIPHER_LEN + 1)); }
        memcpy(buf + n * (CIPHER_LEN + 1), line, CIPHER_LEN);
        buf[n * (CIPHER_LEN + 1) + CIPHER_LEN] = '\0';
        n++;
    }

    fclose(fp);
    *countOut = n;
    return buf;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <file of scrambled passwords>\n", argv[0]);
        return 1;
    }

    // --- get the targets onto the host ---
    int targetCount;
    char *hostTargets = readTargetsFromFile(argv[1], &targetCount);
    printf("read %d passwords from '%s'\n", targetCount, argv[1]);

    if (targetCount == 0) {
        fprintf(stderr, "no usable passwords found in the file\n");
        free(hostTargets);
        return 1;
    }

    // --- move them onto the GPU ---
    char *gpuTargets;
    char *gpuAnswers;
    CHECK_CUDA(cudaMalloc(&gpuTargets, targetCount * (CIPHER_LEN + 1)));
    CHECK_CUDA(cudaMalloc(&gpuAnswers, targetCount * (PLAIN_LEN + 1)));
    CHECK_CUDA(cudaMemcpy(gpuTargets, hostTargets, targetCount * (CIPHER_LEN + 1), cudaMemcpyHostToDevice));

    // --- pick a grid shape that scales with the file, not the search space ---
    // threads/block ~= sqrt(targetCount), blocks = enough to cover the rest.
    // for 10,000 passwords this lands on 100x100, same split the assignment
    // brief uses as its own example.
    int threadsPerBlock = (int)ceil(sqrt((double)targetCount));
    if (threadsPerBlock > 1024) threadsPerBlock = 1024;
    if (threadsPerBlock < 1) threadsPerBlock = 1;
    int blocks = (targetCount + threadsPerBlock - 1) / threadsPerBlock;

    printf("grid: %d blocks x %d threads/block (%d threads total for %d passwords)\n",
           blocks, threadsPerBlock, blocks * threadsPerBlock, targetCount);

    // --- crack them ---
    bruteForceKernel<<<blocks, threadsPerBlock>>>(gpuTargets, targetCount, gpuAnswers);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // --- bring the answers home ---
    char *hostAnswers = (char *)malloc(targetCount * (PLAIN_LEN + 1));
    CHECK_CUDA(cudaMemcpy(hostAnswers, gpuAnswers, targetCount * (PLAIN_LEN + 1), cudaMemcpyDeviceToHost));

    // --- write them out ---
    FILE *out = fopen("decrypted.txt", "w");
    if (!out) {
        fprintf(stderr, "couldn't open decrypted.txt for writing\n");
    } else {
        for (int i = 0; i < targetCount; i++) {
            fprintf(out, "%s\n", hostAnswers + i * (PLAIN_LEN + 1));
        }
        fclose(out);
        printf("wrote %d cracked passwords to decrypted.txt\n", targetCount);
    }

    CHECK_CUDA(cudaFree(gpuTargets));
    CHECK_CUDA(cudaFree(gpuAnswers));
    free(hostTargets);
    free(hostAnswers);

    return 0;
}
