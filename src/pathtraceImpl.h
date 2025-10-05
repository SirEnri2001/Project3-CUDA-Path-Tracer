#pragma once

#include <cuda_runtime_api.h>
#include <thrust/random.h>

#include "scene.h"
struct Camera;
struct ShadeableIntersection;
struct PathSegment;
__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine& rng);
__host__ __device__ glm::vec3 sampleHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine& rng);
__global__ void DirectLightingShadingPathSegments(int depths, int frame, int numPaths, int numMaterials,
    PathSegment* pathSegments, ShadeableIntersection* dev_intersections,
    Scene::RenderProxy* scene,
    int* dev_pathAlive);
__global__ void SamplingShadingPathSegments(int depths, int frame, int numPaths, int numMaterials,
    PathSegment* pathSegments, ShadeableIntersection* dev_intersections,
    Scene::RenderProxy* scene,
    int* dev_pathAlive);
__global__ void generateRayFromCamera(Camera cam, int frames, int maxDepths,
    PathSegment* pathSegments, int* dev_pathAlive, ShadeableIntersection* dev_Intersections, ShadeableIntersection* path_intersect_lights);

__device__ void IntersectGeometry(
    glm::vec3& debug,
    int& hit_geom_index,
    ShadeableIntersection& OutIntersect,
    Ray& InRayWorld,
    int geoms_size,
    Geom* geoms, float t_max = FLT_MAX
);
__device__ void IntersectMeshBVH(
    glm::vec3& debug,
    int& hit_geom_index,
    ShadeableIntersection& OutIntersect,
    Ray& InRayWorld,
    int geoms_size,
    Geom* geoms, float t_max = FLT_MAX
);
__device__ float IntersectMesh(
    glm::vec3& debug,
    int& hit_geom_index,
    ShadeableIntersection& OutIntersect,
    Ray& InRayWorld,
    int geoms_size,
    Geom* geoms, float t_max = FLT_MAX
);

__global__ void PreIntersect(
    int num_paths,
    PathSegment* pathSegments,
    Scene::RenderProxy* scene,
    ShadeableIntersection* intersections,
    int* dev_geom_ids, int* dev_pathAlive);
__global__ void Intersect(
    int num_paths,
    PathSegment* pathSegments,
    Scene::RenderProxy* scene,
    ShadeableIntersection* intersections,
    int* dev_geom_ids, int* dev_pathAlive);