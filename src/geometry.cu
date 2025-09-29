#include "geometry.h"
#include <thrust/random.h>

#include "mesh.h"
#include "sceneStructs.h"
#include "utilities.h"

__device__ void pdfPlane(float& OutPdf, Geom& InGeom)
{
    float faceArea = 4.f * InGeom.scale.x * InGeom.scale.z;
    OutPdf = 1.0f / faceArea;
}

__device__ void sampleCube(Geom& InGeom, glm::vec3& OutWorldPosition, glm::vec3& OutWorldNormal, float& OutPdf,
                           thrust::default_random_engine& rng)
{
    // create random faceIndices -1 or 1
    thrust::uniform_int_distribution<int> uInt(0, 5);
    int faceIndex = uInt(rng);
    // create random float between -0.5 and 0.5
    thrust::uniform_real_distribution<float> uFloat(-0.5f, 0.5f);
    float randomU = uFloat(rng);
    float randomV = uFloat(rng);
    float faceArea = 1.0f;
    glm::vec4 localNormal;
    glm::vec4 localPosition;
    switch (faceIndex)
    {
    case 0:
        localPosition = glm::vec4(-0.5f, randomU, randomV, 1.0f);
        faceArea = 4.f * InGeom.scale.y * InGeom.scale.z;
        localNormal = glm::vec4(-1.f, 0.f, 0.f, 0.f);
        break;
    case 1:
        localPosition = glm::vec4(0.5f, randomU, randomV, 1.0f);
        faceArea = 4.f * InGeom.scale.y * InGeom.scale.z;
        localNormal = glm::vec4(1.f, 0.f, 0.f, 0.f);
        break;
    case 2:
        localPosition = glm::vec4(randomU, 0.5f, randomV, 1.0f);
        faceArea = 4.f * InGeom.scale.x * InGeom.scale.z;
        localNormal = glm::vec4(0.f, 1.f, 0.f, 0.f);
        break;
    case 3:
        localPosition = glm::vec4(randomU, -0.5f, randomV, 1.0f);
        faceArea = 4.f * InGeom.scale.x * InGeom.scale.z;
        localNormal = glm::vec4(0.f, -1.f, 0.f, 0.f);
        break;
    case 4:
        localPosition = glm::vec4(randomU, randomV, 0.5f, 1.0f);
        faceArea = 4.f * InGeom.scale.x * InGeom.scale.y;
        localNormal = glm::vec4(0.f, 0.f, 1.f, 0.f);
        break;
    case 5:
        localPosition = glm::vec4(randomU, randomV, -0.5f, 1.0f);
        faceArea = 4.f * InGeom.scale.x * InGeom.scale.y;
        localNormal = glm::vec4(0.f, 0.f, -1.f, 0.f);
        break;
    }
    glm::vec4 worldPosition = InGeom.transform * localPosition;
    glm::vec4 worldNormal = InGeom.invTranspose * localNormal;
    OutWorldPosition = glm::vec3(worldPosition) / worldPosition.w;
    OutWorldNormal = glm::normalize(glm::vec3(worldNormal));
    OutPdf = 1.0f / 6.0 / faceArea;
}

__device__ void samplePlane(Geom& InGeom, glm::vec3& OutWorldPosition, glm::vec3& OutWorldNormal, float& OutPdf,
    thrust::default_random_engine& rng)
{
    // create random float between -0.5 and 0.5
    thrust::uniform_real_distribution<float> uFloat(-0.5f, 0.5f);
    float randomU = uFloat(rng);
    float randomV = uFloat(rng);
    float faceArea = 4.f * InGeom.scale.x * InGeom.scale.z;
    glm::vec4 localNormal = glm::vec4(0.f, -1.f, 0.f, 0.f);
    glm::vec4 localPosition = glm::vec4(randomU, 0.f, randomV, 1.0f);
    glm::vec4 worldPosition = InGeom.transform * localPosition;
    glm::vec4 worldNormal = InGeom.invTranspose * localNormal;
    OutWorldPosition = glm::vec3(worldPosition) / worldPosition.w;
    OutWorldNormal = glm::normalize(glm::vec3(worldNormal));
    OutPdf = 1.0f / faceArea;
}

__device__ void sampleGeometry(
    Geom& InGeom, glm::vec3& OutWorldPosition, glm::vec3& OutWorldNormal, float& OutPdf,
    thrust::default_random_engine& rng
)
{
    if (InGeom.type == CUBE)
    {
        sampleCube(InGeom, OutWorldPosition, OutWorldNormal, OutPdf, rng);
    }
    else if (InGeom.type == PLANE)
    {
        samplePlane(InGeom, OutWorldPosition, OutWorldNormal, OutPdf, rng);
    }
}

__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside)
{
    Ray q;
    q.origin = multiplyMV(box.inverseTransform, glm::vec4(r.origin, 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/
        {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin)
            {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax)
            {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0)
    {
        outside = true;
        if (tmin <= 0)
        {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }

    return -1;
}

__host__ __device__ float UniformBoxIntersectionTest(
    Ray q,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside)
{
    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/
        {
            float t1 = (0.f - q.origin[xyz]) / qdxyz;
            float t2 = (1.f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin)
            {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax)
            {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0)
    {
        outside = true;
        if (tmin <= 0)
        {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = getPointOnRay(q, tmin);
        normal = glm::normalize(tmin_n);
        return tmin;
    }
    return -1;
}


__host__ __device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside)
{
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0)
    {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0)
    {
        return -1;
    }
    else if (t1 > 0 && t2 > 0)
    {
        t = min(t1, t2);
        outside = true;
    }
    else
    {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside)
    {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}

__host__ __device__ float planeIntersectionTest(
    Geom plane,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside)
{
    Ray q;
    q.origin = multiplyMV(plane.inverseTransform, glm::vec4(r.origin, 1.0f));
    q.direction = glm::normalize(multiplyMV(plane.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float t = -q.origin.y / q.direction.y;
    if (t < 0)
    {
        return -1;
    }
    glm::vec3 planeIntersect = getPointOnRay(q, t);
    if (glm::abs(planeIntersect.x) > 0.5f || glm::abs(planeIntersect.z) > 0.5f)
    {
        return -1;
    }
    intersectionPoint = multiplyMV(plane.transform, glm::vec4(planeIntersect, 1.0f));
    normal = glm::normalize(multiplyMV(plane.invTranspose, glm::vec4(glm::vec3(0.f, 1.f, 0.f), 0.0f)));
    if (q.direction.y>0.f)
    {
		normal = -normal;
    }
    return glm::length(r.origin - intersectionPoint);
}

__host__ __device__ float triangleIntersectionTest(
	glm::vec3 p1, glm::vec3 p2, glm::vec3 p3,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
	bool& front)
{
    glm::vec3 l1 = r.origin - p1;
	glm::vec3 p12 = p2 - p1;
	glm::vec3 p13 = p3 - p1;
	normal = glm::normalize(glm::cross(p12, p13));
	float t_vert = glm::dot(normal, l1);
	float directDot = glm::dot(normal, r.direction);
	float t = t_vert / -directDot;
    intersectionPoint = r.origin + r.direction * t;
	glm::vec3 s1 = p1 - intersectionPoint;
	glm::vec3 s2 = p2 - intersectionPoint;
	glm::vec3 s3 = p3 - intersectionPoint;
    if (glm::dot(normal, glm::cross(s1, s2)) >= 0.f &&
        glm::dot(normal, glm::cross(s2, s3)) >= 0.f &&
        glm::dot(normal, glm::cross(s3, s1)) >= 0.f)
    {
        return t;// glm::length(r.origin - intersectionPoint); // DAMN when use glm::length the ray which from the mesh will hit the mesh itself within a very short distance
	}
	return -1.f;
}

__host__ __device__ float meshIntersectionTest(
    Geom mesh, StaticMesh::RenderProxy* dev_staticMeshes,
    Ray ray_World,
    glm::vec3& IntersectPos_World,
    glm::vec3& IntersectNor_World)
{
	int totalObjectCount = dev_staticMeshes->VertexCount / 3;
    Ray ray_Local;
    ray_Local.origin = multiplyMV(mesh.inverseTransform, glm::vec4(ray_World.origin, 1.0f));
    ray_Local.direction = glm::normalize(multiplyMV(mesh.inverseTransform, glm::vec4(ray_World.direction, 0.0f)));
    float minDistance = -1.f;
    glm::vec3 IntersectPos_Local;
	glm::vec3 IntersectNor_Local;
    glm::vec3 Pos_Temp;
	glm::vec3 Nor_Temp;
	bool front;
    float t_min = FLT_MAX;
    float t;
    for (int i = 0;i < totalObjectCount;i++)
    {
	    // test each triangle in the mesh
        t = triangleIntersectionTest(
            dev_staticMeshes->raw.VertexPosition_Device[3 * i + 0],
            dev_staticMeshes->raw.VertexPosition_Device[3 * i + 1],
            dev_staticMeshes->raw.VertexPosition_Device[3 * i + 2],
            ray_Local,
            Pos_Temp,
            Nor_Temp,
            front
        );
        if (t>0.f && t < t_min)
        {
	        t_min = glm::min(t, t_min);
            IntersectPos_Local = Pos_Temp;
            IntersectNor_Local = Nor_Temp;
        }
    }
    if (t_min==FLT_MAX)
    {
		return -1.f;
    }
    IntersectPos_World = multiplyMV(mesh.transform, glm::vec4(IntersectPos_Local, 1.0f));
    IntersectNor_World = glm::normalize(multiplyMV(mesh.invTranspose, glm::vec4(IntersectNor_Local, 0.0f)));
    minDistance = glm::length(ray_World.origin - IntersectPos_World);
	return minDistance;
}

__device__ int GetPointBoundNextLayer(glm::vec3 p)
{
    int bound = 0;
    if (p.x>=0.5f)
    {
        bound += 1;
    }
    if (p.y>=0.5f)
    {
        bound += 2;
	}
    if (p.z>=0.5f)
    {
        bound += 4;
	}
    return bound;
}

__device__ int GetPointBoundIndex(glm::vec3 p1, int layer) // p1 within [0,1]
{
    int bound = -1;
	int l = 0;
    while (l < layer)
    {
        int boundP1 = GetPointBoundNextLayer(p1);
        bound = 8 * (bound + 1) + boundP1;
        p1 = glm::fract(p1 * 2.f);
        l++;
    }
    return bound;
}

__device__ float IntersectBoundingBoxLayer(
    glm::vec3& debug,
    const Geom& MeshGeom,
    Ray InRayWorld, glm::vec3& IntersectPos_World,
    glm::vec3& IntersectNor_World, 
    int layer, 
    StaticMesh::RenderProxy* dev_staticMeshes)
{
    Ray ray_Local;
    ray_Local.origin = multiplyMV(MeshGeom.inverseTransform, glm::vec4(InRayWorld.origin, 1.0f));
    ray_Local.direction = glm::normalize(multiplyMV(MeshGeom.inverseTransform, glm::vec4(InRayWorld.direction, 0.0f)));

	// calculate the first box intersected by the ray in the given layer

	// step 1: transform ray according to boxMin and boxMax
    Ray BoundSpaceRay = ray_Local;
    BoundSpaceRay.origin -= dev_staticMeshes->boxMin;
    BoundSpaceRay.origin /= (dev_staticMeshes->boxMax - dev_staticMeshes->boxMin);
    BoundSpaceRay.direction /= (dev_staticMeshes->boxMax - dev_staticMeshes->boxMin);
    BoundSpaceRay.direction = glm::normalize(BoundSpaceRay.direction);
    int GridDim = 1 << layer; // 2^layer
	float InvGridDim = 1.f / GridDim;

    bool bOutsize = false;
    
	glm::vec3 tempPos, tempNor;
    float temp_t;
    int maxSteps = 2 * GRID_WIDTH;
    // step 2: march ray to the bound if it is outside the bound
    temp_t = UniformBoxIntersectionTest(
        BoundSpaceRay,
        tempPos,
        tempNor,
        bOutsize
    );
    if (temp_t < 0.f)
    {
        // the ray is not intersect with the bounding box
        return -1.f;
    }
    if (bOutsize)
    {
        BoundSpaceRay.origin = tempPos + BoundSpaceRay.direction * 0.01f * InvGridDim; // march a bit
    }

    while (maxSteps)
    {
		maxSteps--;
        // step 2: march ray to the bound if it is outside the bound
        temp_t = UniformBoxIntersectionTest(
            BoundSpaceRay,
            tempPos,
            tempNor,
            bOutsize
        );
        if (temp_t < 0.f)
        {
            // the ray is not intersect with the bounding box
            return -1.f;
        }

        // step 3: check current bounding if intersect with any triangle
        int boundIndex = GetPointBoundIndex(BoundSpaceRay.origin, layer);
        int startTriangleIndex = dev_staticMeshes->raw.GridIndicesStart_Device[boundIndex + 1];
        int endTriangleIndex = dev_staticMeshes->raw.GridIndicesEnd_Device[boundIndex + 1];
		int triangleCount = endTriangleIndex - startTriangleIndex;
        if (triangleCount > 0 && startTriangleIndex!=-1)
        {
            float minDistance = -1.f;
            glm::vec3 IntersectPos_Local;
            glm::vec3 IntersectNor_Local;
            glm::vec3 Pos_Temp;
            glm::vec3 Nor_Temp;
            bool front;
            float t_min = FLT_MAX;
            float t;
            for (int i = 0; i < triangleCount; i++)
            {
                int triangleId = dev_staticMeshes->raw.TriangleIndices_Device[startTriangleIndex + i];
                // test each triangle in the mesh
                t = triangleIntersectionTest(
                    dev_staticMeshes->raw.VertexPosition_Device[3 * triangleId + 0],
                    dev_staticMeshes->raw.VertexPosition_Device[3 * triangleId + 1],
                    dev_staticMeshes->raw.VertexPosition_Device[3 * triangleId + 2],
                    ray_Local,
                    Pos_Temp,
                    Nor_Temp,
                    front
                );
                if (t > 0.f && t < t_min)
                {
                    t_min = glm::min(t, t_min);
                    IntersectPos_Local = Pos_Temp;
                    IntersectNor_Local = Nor_Temp;
                }
            }
            if (t_min != FLT_MAX)
            {
                IntersectPos_World = multiplyMV(MeshGeom.transform, glm::vec4(IntersectPos_Local, 1.0f));
                IntersectNor_World = glm::normalize(multiplyMV(MeshGeom.invTranspose, glm::vec4(IntersectNor_Local, 0.0f)));
                minDistance = glm::length(InRayWorld.origin - IntersectPos_World);
                return minDistance;
            }
        }

        // step 4: march to next bounding box
        Ray GridSpaceRay = BoundSpaceRay;
        GridSpaceRay.origin *= GridDim;
        glm::vec3 BoundSpaceOffset = glm::floor(GridSpaceRay.origin);
        GridSpaceRay.origin = glm::fract(GridSpaceRay.origin);
        temp_t = UniformBoxIntersectionTest(
            GridSpaceRay,
            tempPos,
            tempNor,
            bOutsize
        );
        BoundSpaceRay.origin = (BoundSpaceOffset + tempPos) * InvGridDim + InvGridDim * BoundSpaceRay.direction * 0.01f; // march a bit
    }
    return -1.f;
}

__device__ float meshIntersectionTest_Optimized(
    glm::vec3& debug,
    Geom mesh, StaticMesh::RenderProxy* dev_staticMeshes,
    Ray ray_World,
    glm::vec3& IntersectPos_World,
    glm::vec3& IntersectNor_World)
{
    float t_min = FLT_MAX;
    float t;
	glm::vec3 tempNor, tempPos;
    for (int curLayer = 0; curLayer < GRID_LAYERS; curLayer++)
    {
        t = IntersectBoundingBoxLayer(
            debug, 
            mesh,
            ray_World,
            tempPos,
            tempNor,
            curLayer,
            dev_staticMeshes
        );
        if (t > 0.f && t < t_min)
        {
            t_min = t;
            IntersectNor_World = tempNor;
            IntersectPos_World = tempPos;
		}
    }
    if (t_min!=FLT_MAX)
    {
        return t_min;
    }
    return -1;
}