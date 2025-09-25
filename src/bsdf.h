#include <cuda_runtime_api.h>
struct ShadeableIntersection;
struct PathSegment;
struct Material;
__global__ void shadeFakeMaterial(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials);


