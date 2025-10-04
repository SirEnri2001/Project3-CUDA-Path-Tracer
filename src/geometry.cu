#include "geometry.h"

#include <glm/gtx/intersect.hpp>
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
__device__ void sampleMesh(Geom& InGeom, glm::vec3& OutWorldPosition, glm::vec3& OutWorldNormal, float& OutPdf,
    thrust::default_random_engine& rng)
{
    // create random float between -0.5 and 0.5
    StaticMesh::RenderProxy* MeshProxy = InGeom.MeshProxy_Device;
    thrust::uniform_int_distribution<int> uInt(0, MeshProxy->VertexCount / 3 - 1);
    thrust::uniform_real_distribution<float> uFloat(0.f, 1.f);
    float randomU = uFloat(rng);
    float randomV = uFloat(rng);
    if (randomU + randomV > 1.0f)
    {
        randomU = 1.f - randomU;
        randomV = 1.f - randomV;
    }
    float randomW = 1.f - randomU - randomV;
    int randomIndex = uInt(rng);
    glm::vec3 p0 = glm::vec3(InGeom.transform * glm::vec4(MeshProxy->raw.VertexPosition_Device[randomIndex * 3 + 0], 1.f));
    glm::vec3 p1 = glm::vec3(InGeom.transform * glm::vec4(MeshProxy->raw.VertexPosition_Device[randomIndex * 3 + 1], 1.f));
    glm::vec3 p2 = glm::vec3(InGeom.transform * glm::vec4(MeshProxy->raw.VertexPosition_Device[randomIndex * 3 + 2], 1.f));
    OutWorldNormal = glm::normalize(glm::cross(p1 - p0, p2 - p0));
    float faceArea = glm::length(glm::cross(p1 - p0, p2 - p0)) * 0.5f;
    OutWorldPosition = randomU * p0 + randomV * p1 + randomW * p2;
    OutPdf = 1.0f / faceArea / (MeshProxy->VertexCount / 3);
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
    else if (InGeom.type == MESH)
    {
        sampleMesh(InGeom, OutWorldPosition, OutWorldNormal, OutPdf, rng);
    }
}

__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    ShadeableIntersection& OutIntersect)
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
        OutIntersect.outside = true;
        if (tmin <= 0)
        {
            tmin = tmax;
            tmin_n = tmax_n;
            OutIntersect.outside = false;
        }
        OutIntersect.intersectPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        OutIntersect.surfaceNormal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - OutIntersect.intersectPoint);
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
    ShadeableIntersection& OutIntersect)
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
        OutIntersect.outside = true;
    }
    else
    {
        t = max(t1, t2);
        OutIntersect.outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    OutIntersect.intersectPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    OutIntersect.surfaceNormal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    return glm::length(r.origin - OutIntersect.intersectPoint);
}

__host__ __device__ float planeIntersectionTest(
    Geom plane,
    Ray r,
    ShadeableIntersection& OutIntersect)
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
    OutIntersect.intersectPoint = multiplyMV(plane.transform, glm::vec4(planeIntersect, 1.0f));
    OutIntersect.surfaceNormal = glm::normalize(multiplyMV(plane.invTranspose, glm::vec4(glm::vec3(0.f, 1.f, 0.f), 0.0f)));
    OutIntersect.uv.x = planeIntersect.x + 0.5f;
    OutIntersect.uv.y = planeIntersect.y + 0.5f;
    if (q.direction.y>0.f)
    {
        OutIntersect.surfaceNormal = -OutIntersect.surfaceNormal;
    }
    return glm::length(r.origin - OutIntersect.intersectPoint);
}

__host__ __device__ float triangleIntersectionTest(
	glm::vec3 p1, glm::vec3 p2, glm::vec3 p3,
    Ray r, glm::vec3& Pos, glm::vec3& Nor, glm::vec3& Bary, bool& Front)
{
#if 0
    if (glm::intersectRayTriangle(r.origin, r.direction, p1, p2, p3, Bary))
    {
        Pos = p1 * Bary.x + p2 * Bary.y + p3 * Bary.z;
        Nor = glm::normalize(glm::cross(p2 - p1, p3 - p1));
        Front = glm::dot(Nor, r.direction) < 0.f;
		return glm::length(r.origin - Pos);
    }
	return -1.f;
#else
    glm::vec3 l1 = r.origin - p1;
	glm::vec3 p12 = p2 - p1;
	glm::vec3 p13 = p3 - p1;
    Nor = glm::normalize(glm::cross(p12, p13));
	float t_vert = glm::dot(Nor, l1);
	float directDot = glm::dot(Nor, r.direction);
	float t = t_vert / -directDot;
    Pos = r.origin + r.direction * t;
	glm::vec3 s1 = p1 - Pos;
	glm::vec3 s2 = p2 - Pos;
	glm::vec3 s3 = p3 - Pos;
    float c1, c2, c3;
	c1 = glm::dot(Nor, glm::cross(s1, s2));
    c2 = glm::dot(Nor, glm::cross(s2, s3));
	c3 = glm::dot(Nor, glm::cross(s3, s1));
    if (c1 >= 0.f &&
        c2 >= 0.f &&
        c3 >= 0.f)
    {
		Front = t_vert > 0.f;
		float area = glm::length(glm::cross(p12, p13));
		float area1 = glm::length(glm::cross(p2 - Pos, p3 - Pos));
		float area2 = glm::length(glm::cross(p3 - Pos, p1 - Pos));
		float area3 = glm::length(glm::cross(p1 - Pos, p2 - Pos));
		Bary.x = area1 / area;
		Bary.y = area2 / area;
		Bary.z = area3 / area;
        return t;// glm::length(r.origin - intersectionPoint); // DAMN when use glm::length the ray which from the mesh will hit the mesh itself within a very short distance
	}
	return -1.f;
#endif
}

__host__ __device__ float meshIntersectionTest(
    Geom mesh, StaticMesh::RenderProxy* MeshProxy,
    Ray ray_World,
    ShadeableIntersection& OutIntersect)
{
	int totalObjectCount = MeshProxy->VertexCount / 3;
    Ray ray_Local;
    ray_Local.origin = multiplyMV(mesh.inverseTransform, glm::vec4(ray_World.origin, 1.0f));
    ray_Local.direction = glm::normalize(multiplyMV(mesh.inverseTransform, glm::vec4(ray_World.direction, 0.0f)));
    float minDistance = -1.f;
    glm::vec3 IntersectPos_Local;
	glm::vec3 IntersectNor_Local;
    glm::vec3 Pos_Temp;
	glm::vec3 Nor_Temp;
    glm::vec3 bary;
    glm::vec2 uv1, uv2, uv3;
	bool front;
    float t_min = FLT_MAX;
    float t;
    for (int i = 0;i < totalObjectCount;i++)
    {
	    // test each triangle in the mesh
        t = triangleIntersectionTest(
            MeshProxy->raw.VertexPosition_Device[3 * i + 0],
            MeshProxy->raw.VertexPosition_Device[3 * i + 1],
            MeshProxy->raw.VertexPosition_Device[3 * i + 2],
            ray_Local,
            Pos_Temp,
            Nor_Temp,
            bary,
            front
        );
        if (MeshProxy->raw.VertexTexCoord_Device !=nullptr && t>0.f && t < t_min)
        {
	        t_min = glm::min(t, t_min);
            IntersectPos_Local = Pos_Temp;
            IntersectNor_Local = Nor_Temp;
			uv1 = MeshProxy->raw.VertexTexCoord_Device[3 * i + 0];
            uv2 = MeshProxy->raw.VertexTexCoord_Device[3 * i + 1];
			uv3 = MeshProxy->raw.VertexTexCoord_Device[3 * i + 2];
        }
    }
    if (t_min==FLT_MAX)
    {
		return -1.f;
    }
	OutIntersect.outside = front;
	OutIntersect.uv = uv1 * bary.x + uv2 * bary.y + uv3 * bary.z;
    OutIntersect.intersectPoint = multiplyMV(mesh.transform, glm::vec4(IntersectPos_Local, 1.0f));
    OutIntersect.surfaceNormal = glm::normalize(multiplyMV(mesh.invTranspose, glm::vec4(IntersectNor_Local, 0.0f)));
    minDistance = glm::length(ray_World.origin - OutIntersect.intersectPoint);
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
    Ray InRayWorld, 
    ShadeableIntersection& OutIntersect,
    int layer, 
    StaticMesh::RenderProxy* MeshProxy, bool preCompute, float& tmin_LastBoundSpace)
{
    Ray ray_Local;
    ray_Local.origin = multiplyMV(MeshGeom.inverseTransform, glm::vec4(InRayWorld.origin, 1.0f));
    ray_Local.direction = glm::normalize(multiplyMV(MeshGeom.inverseTransform, glm::vec4(InRayWorld.direction, 0.0f)));

	// calculate the first box intersected by the ray in the given layer

	// step 1: transform ray according to boxMin and boxMax
    Ray BoundSpaceRay = ray_Local;
    BoundSpaceRay.origin -= MeshProxy->boxMin;
    BoundSpaceRay.origin /= (MeshProxy->boxMax - MeshProxy->boxMin);
    BoundSpaceRay.direction /= (MeshProxy->boxMax - MeshProxy->boxMin);
    BoundSpaceRay.direction = glm::normalize(BoundSpaceRay.direction);
    int GridDim = 1 << layer; // 2^layer
	float InvGridDim = 1.f / GridDim;
    glm::vec3 rayStartBoundSpace = BoundSpaceRay.origin;
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
    if (preCompute)
    {
	    tempPos *= (MeshProxy->boxMax - MeshProxy->boxMin);
        tempPos += MeshProxy->boxMin;
        OutIntersect.intersectPoint = multiplyMV(MeshGeom.transform, glm::vec4(tempPos, 1.0f));
        return glm::length(OutIntersect.intersectPoint - InRayWorld.origin);
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
        int startTriangleIndex = MeshProxy->raw.GridIndicesStart_Device[boundIndex + 1];
        int endTriangleIndex = MeshProxy->raw.GridIndicesEnd_Device[boundIndex + 1];
		int triangleCount = endTriangleIndex - startTriangleIndex;
        if (triangleCount > 0 && startTriangleIndex!=-1)
        {
            float minDistance = -1.f;
            glm::vec3 IntersectPos_Local;
            glm::vec3 IntersectNor_Local;
            glm::vec2 IntersectUV;
            glm::vec3 Bary_Temp;
            glm::vec2 uv1, uv2, uv3;
            bool front;
            float t_min = FLT_MAX;
            float t;
            for (int i = 0; i < triangleCount; i++)
            {
                int triangleId = MeshProxy->raw.TriangleIndices_Device[startTriangleIndex + i];
                // test each triangle in the mesh
                t = triangleIntersectionTest(
                    MeshProxy->raw.VertexPosition_Device[3 * triangleId + 0],
                    MeshProxy->raw.VertexPosition_Device[3 * triangleId + 1],
                    MeshProxy->raw.VertexPosition_Device[3 * triangleId + 2],
                    ray_Local,
                    tempPos,
                    tempNor,
                    Bary_Temp,
                    front
                );
                if (MeshProxy->raw.VertexTexCoord_Device!=nullptr && t > 0.f && t < t_min)
                {
					uv1 = MeshProxy->raw.VertexTexCoord_Device[3 * triangleId + 0];
                    uv2 = MeshProxy->raw.VertexTexCoord_Device[3 * triangleId + 1];
                    uv3 = MeshProxy->raw.VertexTexCoord_Device[3 * triangleId + 2];
                    IntersectUV = uv1 * Bary_Temp.x + uv2 * Bary_Temp.y + uv3 * Bary_Temp.z;
                    t_min = glm::min(t, t_min);
                    IntersectPos_Local = tempPos;
                    IntersectNor_Local = tempNor;
                }
            }
            if (t_min != FLT_MAX)
            {
                OutIntersect.intersectPoint = multiplyMV(MeshGeom.transform, glm::vec4(IntersectPos_Local, 1.0f));
                OutIntersect.surfaceNormal = glm::normalize(multiplyMV(MeshGeom.invTranspose, glm::vec4(IntersectNor_Local, 0.0f)));
				OutIntersect.uv = IntersectUV;
                minDistance = glm::length(InRayWorld.origin - OutIntersect.intersectPoint);
                glm::vec3 BoundSpaceIntersectPoint;
                BoundSpaceIntersectPoint = IntersectPos_Local - MeshProxy->boxMax;
                BoundSpaceIntersectPoint /= (MeshProxy->boxMax - MeshProxy->boxMin);
                tmin_LastBoundSpace = glm::length(BoundSpaceIntersectPoint - rayStartBoundSpace);
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
        if (glm::length(rayStartBoundSpace - BoundSpaceRay.origin)>tmin_LastBoundSpace)
        {
            return -1.f;
        }
    }
    return -1.f;
}

__device__ float IntersectBoundingBoxLayerLocal(
    glm::vec3& debug,
    glm::vec3& OutIntersectLocal,
    glm::vec3& OutNormalLocal,
    glm::vec2& OutUV,
    const Geom& MeshGeom,
    const Ray& InRayLocal,
    int layer,
    StaticMesh::RenderProxy* MeshProxy, bool preCompute, float& tmin_LastBoundSpace)
{
    // calculate the first box intersected by the ray in the given layer

    // step 1: transform ray according to boxMin and boxMax
    Ray BoundSpaceRay = InRayLocal;
    BoundSpaceRay.origin -= MeshProxy->boxMin;
    BoundSpaceRay.origin /= (MeshProxy->boxMax - MeshProxy->boxMin);
    BoundSpaceRay.direction /= (MeshProxy->boxMax - MeshProxy->boxMin);
    BoundSpaceRay.direction = glm::normalize(BoundSpaceRay.direction);
    int GridDim = 1 << layer; // 2^layer
    float InvGridDim = 1.f / GridDim;
    glm::vec3 rayStartBoundSpace = BoundSpaceRay.origin;
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
    if (preCompute)
    {
        tempPos *= (MeshProxy->boxMax - MeshProxy->boxMin);
        tempPos += MeshProxy->boxMin;
        OutIntersectLocal = tempPos;
        return glm::length(tempPos - InRayLocal.origin);
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
        int startTriangleIndex = MeshProxy->raw.GridIndicesStart_Device[boundIndex + 1];
        int endTriangleIndex = MeshProxy->raw.GridIndicesEnd_Device[boundIndex + 1];
        int triangleCount = endTriangleIndex - startTriangleIndex;
        if (triangleCount > 0 && startTriangleIndex != -1)
        {
            float minDistance = -1.f;
            glm::vec3 IntersectPos_Local;
            glm::vec3 IntersectNor_Local;
            glm::vec2 IntersectUV;
            glm::vec3 Bary_Temp;
            glm::vec2 uv1, uv2, uv3;
            bool front;
            float t_min = FLT_MAX;
            float t;
            for (int i = 0; i < triangleCount; i++)
            {
                int triangleId = MeshProxy->raw.TriangleIndices_Device[startTriangleIndex + i];
                // test each triangle in the mesh
                t = triangleIntersectionTest(
                    MeshProxy->raw.VertexPosition_Device[3 * triangleId + 0],
                    MeshProxy->raw.VertexPosition_Device[3 * triangleId + 1],
                    MeshProxy->raw.VertexPosition_Device[3 * triangleId + 2],
                    InRayLocal,
                    tempPos,
                    tempNor,
                    Bary_Temp,
                    front
                );
                if (MeshProxy->raw.VertexTexCoord_Device != nullptr && t > 0.f && t < t_min)
                {
                    uv1 = MeshProxy->raw.VertexTexCoord_Device[3 * triangleId + 0];
                    uv2 = MeshProxy->raw.VertexTexCoord_Device[3 * triangleId + 1];
                    uv3 = MeshProxy->raw.VertexTexCoord_Device[3 * triangleId + 2];
                    IntersectUV = uv1 * Bary_Temp.x + uv2 * Bary_Temp.y + uv3 * Bary_Temp.z;
                    t_min = glm::min(t, t_min);
                    IntersectPos_Local = tempPos;
                    IntersectNor_Local = tempNor;
                }
            }
            if (t_min != FLT_MAX)
            {
                OutNormalLocal = IntersectNor_Local;
                OutIntersectLocal = IntersectPos_Local;
                OutUV = IntersectUV;
                glm::vec3 BoundSpaceIntersectPoint;
                BoundSpaceIntersectPoint = IntersectPos_Local - MeshProxy->boxMax;
                BoundSpaceIntersectPoint /= (MeshProxy->boxMax - MeshProxy->boxMin);
                tmin_LastBoundSpace = glm::length(BoundSpaceIntersectPoint - rayStartBoundSpace);
                return glm::length(OutIntersectLocal - InRayLocal.origin);
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
        if (glm::length(rayStartBoundSpace - BoundSpaceRay.origin) > tmin_LastBoundSpace)
        {
            return -1.f;
        }
    }
    return -1.f;
}

__device__ float meshIntersectionTest_Optimized(
    glm::vec3& debug,
    const Geom& mesh, StaticMesh::RenderProxy* dev_staticMesh,
    const Ray& ray_World,
    ShadeableIntersection& OutIntersectionWorld, bool preCompute)
{
    float t_local;
    float t_min_boundspace = FLT_MAX;
    Ray ray_Local;
    ray_Local.origin = multiplyMV(mesh.inverseTransform, glm::vec4(ray_World.origin, 1.0f));
    ray_Local.direction = glm::normalize(multiplyMV(mesh.inverseTransform, glm::vec4(ray_World.direction, 0.0f)));
    ShadeableIntersection TempIntersectLocal;
    t_local = IntersectBoundingBoxLayerLocal(
        debug,
        TempIntersectLocal.intersectPoint,
        TempIntersectLocal.surfaceNormal,
        TempIntersectLocal.uv,
        mesh,
        ray_Local,
        0,
        dev_staticMesh, true, t_min_boundspace
    );

    if (preCompute)
    {
        if (t_local > 0.f)
        {
            OutIntersectionWorld.intersectPoint = glm::vec3(multiplyMV(mesh.transform, glm::vec4(TempIntersectLocal.intersectPoint, 1.f)));
            OutIntersectionWorld.surfaceNormal = glm::normalize(glm::vec3(multiplyMV(mesh.invTranspose, glm::vec4(TempIntersectLocal.surfaceNormal, 0.f))));
            return t_local;
        }
        return -1.0f;
    }

    if (t_local < 0.f)
    {
        return -1.0f;
    }
    ShadeableIntersection IntersectMin;
    float t_min_local = FLT_MAX;
    for (int curLayer = 0; curLayer < GRID_LAYERS; curLayer++)
    {
        t_local = IntersectBoundingBoxLayerLocal(
            debug,
            TempIntersectLocal.intersectPoint,
            TempIntersectLocal.surfaceNormal,
            TempIntersectLocal.uv,
            mesh,
            ray_Local,
            curLayer,
            dev_staticMesh, false, t_min_boundspace
        );
        if (t_local > 0.f && t_local < t_min_local)
        {
            IntersectMin = TempIntersectLocal;
            t_min_local = t_local;
		}
    }
    if (t_min_local !=FLT_MAX)
    {
        OutIntersectionWorld.intersectPoint = glm::vec3(multiplyMV(mesh.transform, glm::vec4(IntersectMin.intersectPoint, 1.f)));
        OutIntersectionWorld.surfaceNormal = glm::normalize(glm::vec3(multiplyMV(mesh.invTranspose, glm::vec4(IntersectMin.surfaceNormal, 0.f))));
        return glm::length(OutIntersectionWorld.intersectPoint - ray_World.origin);
    }
    return -1;
}