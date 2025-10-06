#include <glm/gtx/transform.hpp>

#include "pathtraceImpl.h"
#include "geometry.h"

__device__ void IntersectGeometry(
    glm::vec3& debug,
    int& hit_geom_index,
    ShadeableIntersection& OutIntersect,
    Ray& InRayWorld,
    int geoms_size,
    Geom* geoms, float t_max
)
{
    float t = -1.0f;
    float t_min = t_max;
    ShadeableIntersection TempIntersect;

    for (int i = 0; i < geoms_size; i++)
    {
        Geom& geom = geoms[i];

        if (geom.type == CUBE)
        {
            t = boxIntersectionTest(geom, InRayWorld, TempIntersect);
        }
        else if (geom.type == SPHERE)
        {
            t = sphereIntersectionTest(geom, InRayWorld, TempIntersect);
        }
        else if (geom.type == PLANE)
        {
            t = planeIntersectionTest(geom, InRayWorld, TempIntersect);
        }
#if 1
        else if (geom.type == MESH && geom.MeshProxy_Device != nullptr)
        {
            continue;
        }
#else
        if (geom.type == MESH && geom.MeshProxy_Device != nullptr)
        {
            glm::vec3 boxMin = geom.MeshProxy_Device->boxMin;
            glm::vec3 boxMax = geom.MeshProxy_Device->boxMax;
            Geom BoundingBox = geom;
            glm::mat4 BoundingBoxToLocal = glm::translate(glm::scale(glm::mat4(1.f), (boxMax - boxMin)), (boxMax + boxMin) * 0.5f);
            BoundingBox.transform = BoundingBox.transform * BoundingBoxToLocal;
            BoundingBox.inverseTransform = glm::inverse(BoundingBox.transform);
            BoundingBox.invTranspose = glm::transpose(BoundingBox.inverseTransform);
            t = boxIntersectionTest(BoundingBox, InRayWorld, TempIntersect);
        }
#endif
        if (t > 0.0f && t_min > t)
        {
            t_min = t;
            hit_geom_index = i;
            OutIntersect = TempIntersect;
            OutIntersect.t_min_World = t_min;
            OutIntersect.intersectBVH = geom.type == MESH;
        }
    }
}

__device__ void IntersectMeshBVH(
    glm::vec3& debug,
    int& hit_geom_index,
    ShadeableIntersection& OutIntersect,
    Ray& InRayWorld,
    int geoms_size,
    Geom* geoms, float t_max
)
{
    float t = -1.0f;
    float t_min = t_max;
    ShadeableIntersection TempIntersect;

    for (int i = 0; i < geoms_size; i++)
    {
        Geom& geom = geoms[i];
		if (geom.type == MESH && geom.MeshProxy_Device != nullptr)
        {
            glm::vec3 boxMin = geom.MeshProxy_Device->boxMin;
            glm::vec3 boxMax = geom.MeshProxy_Device->boxMax;
            t = BVHIntersectionTest(debug, geom, InRayWorld, TempIntersect, boxMin, boxMax);
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                OutIntersect = TempIntersect;
                OutIntersect.t_min_World = t_min;
                OutIntersect.intersectBVH = geom.type == MESH;
            }
        }
    }
}


__device__ float IntersectMesh(
    glm::vec3& debug,
    int& hit_geom_index,
    ShadeableIntersection& OutIntersect,
    Ray& InRayWorld,
    int geoms_size,
    Geom* geoms, float t_max
)
{
    float t = -1.0f;
    float t_min = t_max;
    ShadeableIntersection TempIntersect;
    // naive parse through global geoms
    for (int i = 0; i < geoms_size; i++)
    {
        Geom& geom = geoms[i];
        if (geom.type == MESH && geom.MeshProxy_Device != nullptr)
        {
#if USE_MESH_GRID_ACCELERATION
            t = meshIntersectionTest_Optimized(debug, geom, geom.MeshProxy_Device, InRayWorld, TempIntersect);
#else
#if DISABLE_MESH_ACCELERATION
            t = meshIntersectionTest(geom, geom.MeshProxy_Device, InRayWorld, TempIntersect);
#else
            glm::vec3 boxMin = geom.MeshProxy_Device->boxMin;
            glm::vec3 boxMax = geom.MeshProxy_Device->boxMax;
            float t2 = BVHIntersectionTest(debug, geom, InRayWorld, TempIntersect, boxMin, boxMax);
            if (t2 > 0.0f && t_min > t2)
            {
                t = meshIntersectionTest(geom, geom.MeshProxy_Device, InRayWorld, TempIntersect);
            }
#endif
#endif
        }
        // Compute the minimum t from the intersection tests to determine what
        // scene geometry object was hit first.
        if (t > 0.0f && t_min > t)
        {
            t_min = t;
            hit_geom_index = i;
            OutIntersect = TempIntersect;
        }
    }
    return t_min;
}

__global__ void PreIntersect(
    int num_paths,
    PathSegment* pathSegments,
    Scene::RenderProxy* scene,
    ShadeableIntersection* intersections,
    int* dev_geom_ids, int* dev_pathAlive)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_paths)
    {
        return;
    }
    int path_index = dev_pathAlive[tid];
    if (path_index < 0)
    {
        return;
    }
    PathSegment pathSegment = pathSegments[path_index];
    // If there are no remaining bounces, no need to trace
    if (pathSegment.remainingBounces <= 0) {
        intersections[path_index].materialId = -1;
        dev_geom_ids[path_index] = -1;
        dev_pathAlive[tid] = -1;
        return;
    }
    int hit_geom_index = -1;
    ShadeableIntersection Intersect = intersections[path_index];
    float t_max = Intersect.t_min_World;
    glm::vec3 debug = glm::vec3(0.f);
    IntersectGeometry(debug,
        hit_geom_index, Intersect,
        pathSegment.ray, scene->geoms_size, scene->geoms_Device);
    IntersectMeshBVH(debug,
        hit_geom_index, Intersect,
        pathSegment.ray, scene->geoms_size, scene->geoms_Device, Intersect.t_min_World);

    dev_geom_ids[path_index] = hit_geom_index;
    pathSegments[path_index] = pathSegment;
    if (hit_geom_index == -1)
    {
        Intersect.materialId = -1;
        dev_pathAlive[tid] = -1;
    }
    else if (Intersect.t_min_World < t_max)
    {
        // The ray hits something
        int matId = scene->geoms_Device[hit_geom_index].materialid;
        Intersect.materialId = matId;
        Intersect.t_min_World = t_max;
        pathSegments[path_index] = pathSegment;
    }
    intersections[path_index] = Intersect;
}

__global__ void Intersect(
    int num_paths,
    PathSegment* pathSegments,
    Scene::RenderProxy* scene,
    ShadeableIntersection* intersections,
    int* dev_geom_ids, int* dev_pathAlive)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_paths)
    {
        return;
    }
    int path_index = dev_pathAlive[tid];
    if (path_index < 0)
    {
        return;
    }
    PathSegment pathSegment = pathSegments[path_index];
    // If there are no remaining bounces, no need to trace
    if (pathSegment.remainingBounces <= 0) {
        intersections[path_index].materialId = -1;
        dev_geom_ids[path_index] = -1;
        dev_pathAlive[tid] = -1;
        return;
    }
    int hit_geom_index = -1;
    ShadeableIntersection Intersect = intersections[path_index];
    //if (!Intersect.intersectBVH)
    //{
    //    return;
    //}
    float t_min2 = Intersect.t_min_World;
    glm::vec3 debug;
    IntersectGeometry(debug,
        hit_geom_index, Intersect,
        pathSegment.ray, scene->geoms_size, scene->geoms_Device);
    IntersectMesh(debug,
        hit_geom_index, Intersect,
        pathSegment.ray, scene->geoms_size, scene->geoms_Device, Intersect.t_min_World);
    dev_geom_ids[path_index] = hit_geom_index;
    if (hit_geom_index == -1)
    {
        Intersect.materialId = -1;
        dev_pathAlive[tid] = -1;
    }
    else if (Intersect.t_min_World < t_min2)
    {
        // The ray hits something
        int matId = scene->geoms_Device[hit_geom_index].materialid;
        Intersect.materialId = matId;
        Intersect.t_min_World = t_min2;
    }
    if (glm::length(debug) > 0.1f)
    {
        pathSegment.debug = debug;
    }
    pathSegments[path_index] = pathSegment;
    intersections[path_index] = Intersect;
}