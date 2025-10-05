#include "common.h"

#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <cstdio>
#include <glm/detail/type_vec.hpp>

void checkCUDAErrorFn(const char* msg, const char* file, int line)
{
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err)
    {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file)
    {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
    getchar();
#endif // _WIN32
    exit(EXIT_FAILURE);
#endif // ERRORCHECK
}


/**
 * Handy-dandy hash function that provides seeds for random number generation.
 */
__host__ __device__ unsigned int utilhash(unsigned int a)
{
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int frames, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | frames) ^ utilhash(index);
    return thrust::default_random_engine(h);
}


__host__ __device__
bool is_nan(glm::vec3 v)
{
    return cuda::std::isnan(v.x) || cuda::std::isnan(v.y) || cuda::std::isnan(v.z);
}

__host__ __device__
bool is_inf(glm::vec3 v)
{
    return cuda::std::isinf(v.x) || cuda::std::isinf(v.y) || cuda::std::isinf(v.z);
}