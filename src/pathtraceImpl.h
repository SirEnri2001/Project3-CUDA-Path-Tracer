#pragma once

#include <cuda_runtime_api.h>
#include <glm/detail/type_vec.hpp>

#include "mesh.h"
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
    int* device_materialIds, int* dev_pathAlive, StaticMesh::RenderProxy* mesh);
__global__ void generateRayFromIntersections(int iter, int frame, int numPaths,
    PathSegment* pathSegments, ShadeableIntersection* dev_intersections,
    Material* inMaterial, int geomSize, Geom* geoms, Geom* light_geoms, 
    int* dev_pathAlive, StaticMesh::RenderProxy* mesh);
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments, int* dev_pathAlive);
__global__ void calculateMeshGridSpeedup(StaticMesh::RenderProxy* InMeshData);