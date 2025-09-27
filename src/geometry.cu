#include "geometry.h"
#include <thrust/random.h>

#include "mesh.h"
#include "sceneStructs.h"

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
    glm::vec4 localNormal = glm::vec4(0.f, 1.f, 0.f, 0.f);
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
    if (t_vert<0.f)
    {
        return -1.f;
    }
	float t = t_vert / -glm::dot(normal, r.direction);
    intersectionPoint = r.origin + r.direction * t;
	glm::vec3 s1 = p1 - intersectionPoint;
	glm::vec3 s2 = p2 - intersectionPoint;
	glm::vec3 s3 = p3 - intersectionPoint;
    if (glm::dot(normal, glm::cross(s1, s2)) >= 0.f &&
        glm::dot(normal, glm::cross(s2, s3)) >= 0.f &&
        glm::dot(normal, glm::cross(s3, s1)) >= 0.f)
    {
        return glm::length(r.origin - intersectionPoint);
	}
	return -1.f;
}

__host__ __device__ float meshIntersectionTest(
    Geom mesh, StaticMeshData_Device* dev_staticMeshes,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal)
{
	int totalObjectCount = dev_staticMeshes->VertexCount / 3;
    Ray q;
    q.origin = multiplyMV(mesh.inverseTransform, glm::vec4(r.origin, 1.0f));
    q.direction = glm::normalize(multiplyMV(mesh.inverseTransform, glm::vec4(r.direction, 0.0f)));
    float minDistance = -1.f;
    glm::vec3 tempIntersection;
	glm::vec3 tempNormal;
	bool front;
    float t_min = FLT_MAX;
    float t;
    for (int i = 0;i < totalObjectCount;i++)
    {
	    // test each triangle in the mesh
        t = triangleIntersectionTest(
            dev_staticMeshes->VertexPosition_Device[i * 3 + 0],
            dev_staticMeshes->VertexPosition_Device[i * 3 + 1],
            dev_staticMeshes->VertexPosition_Device[i * 3 + 2],
            q,
            tempIntersection,
            tempNormal,
            front
        );
        if (t>0.f && t < t_min)
        {
	        t_min = glm::min(t, t_min);
            intersectionPoint = tempIntersection;
			normal = tempNormal;
        }
    }
    intersectionPoint = multiplyMV(mesh.transform, glm::vec4(intersectionPoint, 1.0f));
    normal = glm::normalize(multiplyMV(mesh.invTranspose, glm::vec4(normal, 0.0f)));
    minDistance = glm::length(r.origin - intersectionPoint);
	return t_min == FLT_MAX ? -1.f : minDistance;
}