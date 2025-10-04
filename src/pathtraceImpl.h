#pragma once

#include <cuda_runtime_api.h>

#include "scene.h"
struct Camera;
struct ShadeableIntersection;
struct PathSegment;
__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
	Scene::RenderProxy* scene,
    ShadeableIntersection* intersections,
    int* device_materialIds, int* dev_pathAlive, bool preCompute);
__global__ void DirectLightingShadingPathSegments(int depths, int frame, int numPaths,
    PathSegment* pathSegments, ShadeableIntersection* dev_intersections,
    Scene::RenderProxy* scene,
    int* dev_pathAlive);
__global__ void SamplingShadingPathSegments(int depths, int frame, int numPaths,
    PathSegment* pathSegments, ShadeableIntersection* dev_intersections,
    Scene::RenderProxy* scene,
    int* dev_pathAlive);
__global__ void generateRayFromCamera(Camera cam, int frames, int maxDepths,
    PathSegment* pathSegments, int* dev_pathAlive);