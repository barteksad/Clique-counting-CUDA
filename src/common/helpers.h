#ifndef __HELPERS_H__
#define __HELPERS_H__
#include <stdio.h>

#include <cuda_runtime_api.h>
#include <cuda.h>

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


#define HANDLE_NULL( a ) {if (a == NULL) { \
                            printf( "Host memory failed in %s at line %d\n", \
                                    __FILE__, __LINE__ ); \
                            exit( EXIT_FAILURE );}}

__device__ uint32_t bsearch_dev(const uint32_t *array, const uint32_t key, const uint32_t size) {
    uint32_t low = 0;
    uint32_t high = size;
    uint32_t mid;
    while (low < high) {
        mid = (low + high) / 2;
        // printf("low: %d, high: %d, mid: %d\n", low, high, mid);
        if (array[mid] < key) {
            low = mid + 1;
        } else if (array[mid] > key) {
            high = mid;
        } else {
            return mid;
        }
    }
    if(array[low] == key) {
        return low;
    } else {
        return UINT32_MAX;
    }
}

#endif  // __HELPERS_H__