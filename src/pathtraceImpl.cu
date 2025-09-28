#include "pathtraceImpl.h"

#include "common.h"
#include "geometry.h"
#include "interactions.h"
#include "intersections.h"
#include "sceneStructs.h"
#include "utilities.h"
#include "mesh.h"

__device__ int GetTriangleBound(glm::vec3 p1, glm::vec3 p2, glm::vec3 p3, int layer) // p1 p2 p3 within [0,1]
{
	int bound = -1;
    while (layer < 4)
    {
        int boundP1 = GetPointBoundNextLayer(p1);
        int boundP2 = GetPointBoundNextLayer(p2);
        int boundP3 = GetPointBoundNextLayer(p3);
        if (boundP1 != boundP2 || boundP1 != boundP3)
        {
            break;
		}
        bound = 8*(bound+1) + boundP1;
        p1 = glm::fract(p1 * 2.f);
        p2 = glm::fract(p2 * 2.f);
        p3 = glm::fract(p3 * 2.f);
		layer++;
    }
    return bound;
}

__global__ void calculateMeshGridSpeedup(StaticMeshData_Device* InMeshData)
{
	int triangleId = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (triangleId >= InMeshData->VertexCount / 3)
    {
        return;
	}
    glm::vec3 p1 = InMeshData->raw.VertexPosition_Device[triangleId * 3 + 0];
    glm::vec3 p2 = InMeshData->raw.VertexPosition_Device[triangleId * 3 + 1];
    glm::vec3 p3 = InMeshData->raw.VertexPosition_Device[triangleId * 3 + 2];
	glm::vec3 boxMin = InMeshData->boxMin;
	glm::vec3 boxMax = InMeshData->boxMax;
	p1 = (p1 - boxMin) / (boxMax - boxMin);
	p2 = (p2 - boxMin) / (boxMax - boxMin);
    p3 = (p3 - boxMin) / (boxMax - boxMin);
    int bound = GetTriangleBound(p1, p2, p3, 0);
	InMeshData->raw.TraingleToGridIndices_Device[triangleId] = bound;
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments, int* dev_pathAlive)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        dev_pathAlive[index] = index;
        PathSegment& segment = pathSegments[index];
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, traceDepth);
        thrust::uniform_real_distribution<float> u01(0, 1);
		float ramdomNumber1 = u01(rng);
		float ramdomNumber2 = u01(rng);
        segment.ray.origin = cam.position;
        segment.Contribution = glm::vec3(0.0f, 0.0f, 0.0f);
		segment.BSDF = glm::vec3(1.0f, 1.0f, 1.0f);
		segment.PDF = 1.0f;
        segment.Cosine = 1.0f;

        // TODO: implement antialiasing by jittering the ray
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x + ramdomNumber1 - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y + ramdomNumber2 - (float)cam.resolution.y * 0.5f)
        );

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
		segment.debug = glm::vec3(1, 0, 1);
    }
}

__device__ float getIntersectionGeometryIndex(
    glm::vec3& debug,
    int& hit_geom_index,
    glm::vec3& intersect_point,
    glm::vec3& normal,
    Ray& InRayWorld,
    int geoms_size,
	Geom* geoms, 
    StaticMeshData_Device* dev_staticMeshes
)
{
    float t = -1.0f;
    float t_min = FLT_MAX;
    bool outside = true;
    glm::vec3 debug1;
    glm::vec3 IntersectPos_World;
    glm::vec3 IntersectNor_World;

    // naive parse through global geoms

    for (int i = 0; i < geoms_size; i++)
    {
        Geom& geom = geoms[i];

        if (geom.type == CUBE)
        {
            t = boxIntersectionTest(geom, InRayWorld, IntersectPos_World, IntersectNor_World, outside);
        }
        else if (geom.type == SPHERE)
        {
            t = sphereIntersectionTest(geom, InRayWorld, IntersectPos_World, IntersectNor_World, outside);
        }
    	else if (geom.type == PLANE)
        {
            t = planeIntersectionTest(geom, InRayWorld, IntersectPos_World, IntersectNor_World, outside);
        }else if (geom.type==MESH && dev_staticMeshes!=nullptr)
        {
			//t = meshIntersectionTest_Optimized(debug, geom, dev_staticMeshes, InRayWorld, IntersectPos_World, IntersectNor_World);
        }
        // Compute the minimum t from the intersection tests to determine what
        // scene geometry object was hit first.
        if (t > 0.0f && t_min > t)
        {
            t_min = t;
            hit_geom_index = i;
            intersect_point = IntersectPos_World;
            normal = IntersectNor_World;
        }
    }
    return t_min;
}

__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    ShadeableIntersection* intersections,
    int* device_materialIds, int* dev_pathAlive, StaticMeshData_Device* dev_staticMeshes)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int path_index = dev_pathAlive[tid];
    if (path_index < 0 || tid >= num_paths)
    {
        return;
    }
    PathSegment pathSegment = pathSegments[path_index];
	// If there are no remaining bounces, no need to trace
    if (pathSegment.remainingBounces <= 0) {
        intersections[path_index].materialId = -1;
        device_materialIds[path_index] = -1;
        dev_pathAlive[tid] = -1;
        return;
	}
    int hit_geom_index = -1;
    glm::vec3 intersect_point;
    glm::vec3 normal;

    getIntersectionGeometryIndex(pathSegments[path_index].debug,
        hit_geom_index, intersect_point, normal,
        pathSegment.ray, geoms_size, geoms, dev_staticMeshes);

    if (hit_geom_index == -1)
    {
        intersections[path_index].materialId = -1;
        device_materialIds[path_index] = -1;
        dev_pathAlive[tid] = -1;
    }
    else
    {
        // The ray hits something
        int matId = geoms[hit_geom_index].materialid;
		intersections[path_index].intersectPoint = intersect_point;
        intersections[path_index].materialId = matId;
        device_materialIds[path_index] = matId;
        intersections[path_index].surfaceNormal = normal;
    }
}

__host__ __device__ float power_heuristic(float pdf_a, float pdf_b) {
    float a = pdf_a * pdf_a;
    float b = pdf_b * pdf_b;
    return a / (a + b);
}

__device__ void bsdfDiffuse(glm::vec3& outBSDF, float& outPDF, const Ray& in_wi, glm::vec3 surfaceNormal, Material* material)
{
    float absdot = glm::abs(glm::dot(in_wi.direction, surfaceNormal));
    float pdf = absdot * INV_PI;
	outPDF = pdf;
    outBSDF = material->color * INV_PI;
}

__device__ void bsdfDiffuseSample(glm::vec3& outBSDF, float& outPDF, Ray& out_wi, glm::vec3 p,
    glm::vec3 surfaceNormal, Material* material,
    thrust::default_random_engine& rng)
{
    // set segment.ray.origin
    out_wi.origin = p;
    // set segment.ray.direction on random position
    out_wi.direction = calculateRandomDirectionInHemisphere(surfaceNormal, rng);
    bsdfDiffuse(outBSDF, outPDF, out_wi, surfaceNormal, material);
}

__device__ void bsdfSpecular(PathSegment* wi, PathSegment* wo, Material* material)
{

}

__device__ void bsdfEmitting(PathSegment* wo, Material* material)
{
    //wo->remainingBounces = 0;
    //wo->color *= material->emittance;
}

__device__ void getGeomPDF(float& outPdf, Geom& InGeom)
{
    if (InGeom.type == CUBE)
    {
        //pdfCube(outPdf, InGeom);
    }
    else if (InGeom.type == PLANE)
    {
        pdfPlane(outPdf, InGeom);
    }
}

__device__ bool sampleLightFromIntersections(
    glm::vec3 &debug, 
    glm::vec3& outDirectLight,
    float& outPdf,
    Ray& wj,
	glm::vec3 p, // intersection point
    const Material& light_mat,
    Geom& light_geom,
    int geomSize,
	Geom* geoms,
    StaticMeshData_Device* mesh,
    thrust::default_random_engine& rng
)
{
    glm::vec3 lightPosition;
    glm::vec3 lightNormal;
	outPdf = 1.0f;
    sampleGeometry(light_geom, lightPosition, lightNormal, outPdf, rng);
    wj.origin = p;
	wj.direction = glm::normalize(lightPosition - wj.origin);
    float distance = glm::length(lightPosition - wj.origin);
    // convert pdf from area to solid angle
	// pdf_L(direct light) = pdf_A * (distance * distance) / (n . wj)
	float dotProduct = abs(glm::dot(lightNormal, wj.direction));
    outPdf /= dotProduct;
    outPdf *= (distance * distance);
	float t;
	glm::vec3 tmp_intersect;
	glm::vec3 tmp_normal;
	int hit_geom_index = -1;
	t = getIntersectionGeometryIndex(debug, 
		hit_geom_index,
        tmp_intersect, tmp_normal,
        wj, geomSize, geoms, mesh);
    outDirectLight += light_mat.emittance;
    return dotProduct > 0.0001f && hit_geom_index == 0;//&& glm::length(tmp_intersect - lightPosition) < 0.001f;
}

__device__ void SampleDirectLightMIS(glm::vec3& debug, glm::vec3& OutContribution, 
    glm::vec3 In_p, glm::vec3 InSurfaceNormal, Material& InSurfaceMat, 
    Geom& InLightGeom, Material& InLightMat, int GeomSize, Geom* Geoms, StaticMeshData_Device* mesh,
    thrust::default_random_engine& rng)
{
    glm::vec3 directLight;
    float pdf_Ld;
    glm::vec3 bsdf;
    float pdf_bsdf;
    Ray wj;
    bool sampledDirectLight = sampleLightFromIntersections(debug, directLight, pdf_Ld, wj, In_p, InLightMat, InLightGeom, GeomSize, Geoms, mesh, rng);
    if (sampledDirectLight)
    {
        bsdfDiffuse(bsdf, pdf_bsdf, wj, InSurfaceNormal, &InSurfaceMat);
        float weight = power_heuristic(pdf_Ld, pdf_bsdf);
        OutContribution += directLight * bsdf * glm::max(0.f, glm::dot(wj.direction, InSurfaceNormal)) * weight / pdf_Ld;
    }
	else
    {
	    OutContribution = glm::vec3(0.f);
		return;
    }
    bsdfDiffuseSample(bsdf, pdf_bsdf, wj, In_p, InSurfaceNormal, &InSurfaceMat, rng);
    int hit_index = -1;
    glm::vec3 intersect, normal;
    getIntersectionGeometryIndex(debug,hit_index, intersect, normal, wj, GeomSize, Geoms, mesh);
    if (hit_index != 0)
    {
       OutContribution = glm::vec3(0.f);
       return;
    }
    getGeomPDF(pdf_Ld, InLightGeom);
    float weight = power_heuristic(pdf_bsdf, pdf_Ld);
    OutContribution += directLight * bsdf * glm::max(0.f, glm::dot(wj.direction, InSurfaceNormal)) * weight / pdf_bsdf;    
}


__global__ void generateRayFromIntersections(int iter, int numPaths,
    PathSegment* pathSegments, ShadeableIntersection* dev_intersections,
    Material* inMaterial, int geomSize, Geom* geoms, Geom* light_geoms, int* dev_pathAlive, StaticMeshData_Device* mesh)
{
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    int pathIndex = dev_pathAlive[tid];
    if (pathIndex < 0)
    {
        return;
    }
    if (tid >= numPaths)
    {
        return;
    }
    PathSegment path_segment = pathSegments[pathIndex];
    ShadeableIntersection intersection = dev_intersections[pathIndex];
	Material light_mat = inMaterial[light_geoms[0].materialid];
	Geom light_geom = light_geoms[0];
    if (intersection.materialId < 0) {
        path_segment.remainingBounces = 0;
        pathSegments[pathIndex] = path_segment;
        return;
    }
    Material material = inMaterial[intersection.materialId];
	glm::vec3 p = intersection.intersectPoint + EPSILON * intersection.surfaceNormal;
    thrust::default_random_engine rng = makeSeededRandomEngine(iter, pathIndex, path_segment.remainingBounces);
    if (material.emittance > 0.)
    {
		path_segment.Contribution += path_segment.BSDF * material.emittance / path_segment.PDF * path_segment.Cosine;
        path_segment.remainingBounces = 0;
        pathSegments[pathIndex] = path_segment;
        return;
    }
    glm::vec3 contrib;
    glm::vec3 debug;
    SampleDirectLightMIS(debug, contrib, p, intersection.surfaceNormal, material,
        light_geom, light_mat, geomSize, geoms, mesh, rng);
	path_segment.Contribution += path_segment.BSDF * contrib / path_segment.PDF * path_segment.Cosine;
    Ray wi;
    glm::vec3 bsdf_at_p;
	float pdf_bsdf;
    bsdfDiffuseSample(bsdf_at_p, pdf_bsdf, wi, p, intersection.surfaceNormal, &material, rng);
	path_segment.BSDF *= bsdf_at_p;
    path_segment.PDF *= pdf_bsdf;
	path_segment.Cosine *= glm::max(0.f, glm::dot(wi.direction, intersection.surfaceNormal));

    path_segment.ray = wi;
	path_segment.remainingBounces--;
    pathSegments[pathIndex] = path_segment;
}