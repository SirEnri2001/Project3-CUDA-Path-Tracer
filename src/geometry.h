#pragma once
#include <cuda_runtime_api.h>
#include <thrust/random.h>

#include "mesh.h"
#include "glm/glm.hpp"
#include "sceneStructs.h"

struct StaticMeshData_Device;
__device__ void sampleCube(Geom& InGeom, glm::vec3& OutWorldPosition, glm::vec3& OutWorldNormal, float& OutPdf,
                           thrust::default_random_engine& rng);
__device__ void samplePlane(Geom& InGeom, glm::vec3& OutWorldPosition, glm::vec3& OutWorldNormal, float& OutPdf,
    thrust::default_random_engine& rng);
__device__ void sampleGeometry(
    Geom& InGeom, glm::vec3& OutWorldPosition, glm::vec3& OutWorldNormal, float& OutPdf,
    thrust::default_random_engine& rng
);

// CHECKITOUT
/**
 * Compute a point at parameter value `t` on ray `r`.
 * Falls slightly short so that it doesn't intersect the object it's hitting.
 */
__host__ __device__ inline glm::vec3 getPointOnRay(Ray r, float t)
{
    return r.origin + (t - .0001f) * glm::normalize(r.direction);
}

/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
__host__ __device__ inline glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v)
{
    return glm::vec3(m * v);
}

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed cube. Untransformed,
 * the cube ranges from -0.5 to 0.5 in each axis and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside);

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed sphere. Untransformed,
 * the sphere always has radius 0.5 and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside);
__host__ __device__ float planeIntersectionTest(
    Geom plane,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside);
__device__ void pdfPlane(float& OutPdf, Geom& InGeom);
__host__ __device__ float meshIntersectionTest(
    Geom plane, StaticMeshData_Device* dev_staticMeshes,
    Ray ray_World,
    glm::vec3& IntersectPos_World,
    glm::vec3& IntersectNor_World);
__device__ int GetPointBoundNextLayer(glm::vec3 p);
__device__ float meshIntersectionTest_Optimized(
    glm::vec3& debug,
    Geom mesh, StaticMesh::RenderProxy* dev_staticMeshes,
    Ray ray_World,
    glm::vec3& IntersectPos_World,
    glm::vec3& IntersectNor_World);