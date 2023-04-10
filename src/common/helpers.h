#ifndef __HELPERS_H__
#define __HELPERS_H__
#include <stdio.h>
#include <assert.h>

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
    int low = 0;
    int high = size - 1;
    int mid;
    while (low <= high) {
        assert(low < size && high < size);
        mid = (low + high) / 2;
        if (array[mid] < key) {
            low = mid + 1;
        } else if (array[mid] > key) {
            high = mid - 1;
        } else {
            return mid;
        }
    }
    return UINT32_MAX;
}

#endif  // __HELPERS_H__