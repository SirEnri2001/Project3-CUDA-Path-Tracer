#include <cuda_runtime_api.h>
struct Camera;
struct Material;
struct ShadeableIntersection;
struct Geom;
struct PathSegment;
__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    ShadeableIntersection* intersections,
    int* device_materialIds);
__global__ void generateRayFromIntersections(int iter, int numPaths,
    PathSegment* pathSegments, ShadeableIntersection* dev_intersections,
    Material* inMaterial, int geomSize, Geom* geoms, Geom* light_geoms);
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments);