#include<cuda_runtime_api.h>
#include <thrust/random.h>
#include <glm/glm.hpp>
#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)

void checkCUDAErrorFn(const char* msg, const char* file, int line);
__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int frames, int index, int depth);


__host__ __device__
bool is_nan(glm::vec3 v);

__host__ __device__
bool is_inf(glm::vec3 v);