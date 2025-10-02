#include "pathtraceImpl.h"

#include "bsdf.h"
#include "common.h"
#include "geometry.h"
#include "interactions.h"
#include "intersections.h"
#include "sceneStructs.h"
#include "utilities.h"
#include "mesh.h"
#include "material.h"
#include "renderproxy.h"
#include "scene.h"

#define USE_MESH_GRID_ACCELERATION 1

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
    ShadeableIntersection& OutIntersect,
    Ray& InRayWorld,
    int geoms_size,
	Geom* geoms
)
{
    float t = -1.0f;
    float t_min = FLT_MAX;
    bool outside = true;
    glm::vec3 debug1;
    ShadeableIntersection TempIntersect;
    // naive parse through global geoms

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
        }else if (geom.type==MESH && geom.MeshProxy_Device!=nullptr)
        {
#if USE_MESH_GRID_ACCELERATION
			t = meshIntersectionTest_Optimized(debug, geom, geom.MeshProxy_Device, InRayWorld, TempIntersect);
#else
            t = meshIntersectionTest(geom, geom.MeshProxy_Device, InRayWorld, OutIntersect);
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

__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Scene::RenderProxy* scene,
    ShadeableIntersection* intersections,
    int* device_materialIds, int* dev_pathAlive)
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
    ShadeableIntersection Intersect = intersections[path_index];
    glm::vec3 debug;
    getIntersectionGeometryIndex(debug,
        hit_geom_index, Intersect,
        pathSegment.ray, scene->geoms_size, scene->geoms_Device);

    if (hit_geom_index == -1)
    {
        Intersect.materialId = -1;
        device_materialIds[path_index] = -1;
        dev_pathAlive[tid] = -1;
    }
    else
    {
        // The ray hits something
        int matId = scene->geoms_Device[hit_geom_index].materialid;
        device_materialIds[path_index] = matId;
        Intersect.materialId = matId;
        pathSegment.debug = glm::vec3(Intersect.uv.x, Intersect.uv.y, 0.f);
		pathSegments[path_index] = pathSegment;
    }
	intersections[path_index] = Intersect;
}

__host__ __device__ float power_heuristic(float pdf_a, float pdf_b) {
    float a = pdf_a * pdf_a;
    float b = pdf_b * pdf_b;
    return a / (a + b);
}

//__device__ void bsdfDiffuse(glm::vec3& outBSDF, float& outPDF, const Ray& in_wi, glm::vec3 surfaceNormal, Material* material)
//{
//    float absdot = glm::abs(glm::dot(in_wi.direction, surfaceNormal));
//    float pdf = absdot * INV_PI;
//	outPDF = pdf;
//    outBSDF = material->color * INV_PI;
//}
//
//__device__ void bsdfDiffuseSample(glm::vec3& outBSDF, float& outPDF, Ray& out_wi, glm::vec3 p,
//    glm::vec3 surfaceNormal, Material* material,
//    thrust::default_random_engine& rng)
//{
//    // set segment.ray.origin
//    out_wi.origin = p;
//    // set segment.ray.direction on random position
//    out_wi.direction = calculateRandomDirectionInHemisphere(surfaceNormal, rng);
//    bsdfDiffuse(outBSDF, outPDF, out_wi, surfaceNormal, material);
//}

__device__ void bsdfPBR(glm::vec3& debug, glm::vec3& outBSDF, 
    float& outPDF, glm::vec2 uv,
    const Material& material, 
    glm::vec3 L /*Light direction*/, 
    glm::vec3 V /*View direction*/, 
    glm::vec3 N /*Normal*/)
{
    BRDF_Params params;
    if (material.BaseColorTextureProxy_Device != nullptr)
    {
        params.baseColor = GetColorDevice(*material.BaseColorTextureProxy_Device, uv);
    }
	else
	{
        params.baseColor = material.color;
	}
    debug = params.baseColor;
    params.roughness = material.roughness;
    float absdot = glm::abs(glm::dot(L, N));
    float pdf = absdot * INV_PI;

    glm::vec3 tangentX, tangentY;
    // calculate tangentX and tangentY
    glm::vec3 v(0.f, 1.f, 0.f);
    if (glm::abs(glm::dot(v, N)) > 0.999f)
    {
        v = glm::vec3(1.f, 0.f, 0.f);
    }
    tangentX = glm::normalize(glm::cross(v, N));
    tangentY = glm::normalize(glm::cross(N, tangentX));
    outPDF = pdf;
    outBSDF = BRDF(params, L, V, N, tangentX, tangentY);
}

__device__ void bsdfPBRSample(glm::vec3& debug, glm::vec3& outBSDF, float& outPDF, Ray& out_wi, glm::vec3 ViewDir, glm::vec3 p,
    glm::vec3 surfaceNormal, glm::vec2 uv, const Material& material, 
    thrust::default_random_engine& rng)
{
    // set segment.ray.origin
    out_wi.origin = p;
    // set segment.ray.direction on random position
    out_wi.direction = calculateRandomDirectionInHemisphere(surfaceNormal, rng);
	bsdfPBR(debug, outBSDF, outPDF, uv, material, out_wi.direction, ViewDir, surfaceNormal);
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
	int hit_geom_index = -1;
    ShadeableIntersection _;
	getIntersectionGeometryIndex(debug, 
		hit_geom_index,
        _,
        wj, geomSize, geoms);
    outDirectLight += light_mat.emittance;
    return dotProduct > 0.0001f && hit_geom_index == 0;//&& glm::length(tmp_intersect - lightPosition) < 0.001f;
}

__device__ void SampleDirectLightMIS(glm::vec3& debug, glm::vec3& OutContribution, 
    glm::vec3 In_p, glm::vec3 InViewDir, glm::vec3 InSurfaceNormal, Material& InSurfaceMat, 
    Geom& InLightGeom, Material& InLightMat, int GeomSize, Geom* Geoms, 
    thrust::default_random_engine& rng)
{
    glm::vec3 directLight;
    float pdf_Ld;
    glm::vec3 bsdf;
    glm::vec2 uv;
    float pdf_bsdf;
    Ray wj;
    bool sampledDirectLight = sampleLightFromIntersections(debug, directLight, pdf_Ld, wj, In_p, InLightMat, InLightGeom, GeomSize, Geoms, rng);
    if (sampledDirectLight)
    {
        //bsdfDiffuse(bsdf, pdf_bsdf, wj, InSurfaceNormal, &InSurfaceMat);
        bsdfPBR(debug, bsdf, pdf_bsdf, uv, InSurfaceMat, wj.direction, InViewDir, InSurfaceNormal);
        float weight = power_heuristic(pdf_Ld, pdf_bsdf);
        OutContribution += directLight * bsdf * glm::max(0.f, glm::dot(wj.direction, InSurfaceNormal)) * weight / pdf_Ld;
    }
	else
    {
	    OutContribution = glm::vec3(0.f);
		return;
    }
    bsdfPBRSample(debug, bsdf, pdf_bsdf, wj, InViewDir, In_p, InSurfaceNormal, uv, InSurfaceMat, rng);
    //bsdfDiffuseSample(bsdf, pdf_bsdf, wj, In_p, InSurfaceNormal, &InSurfaceMat, rng);
    int hit_index = -1;
    ShadeableIntersection _;
    getIntersectionGeometryIndex(debug,hit_index, _, wj, GeomSize, Geoms);
    if (hit_index != 0)
    {
       OutContribution = glm::vec3(0.f);
       return;
    }
    getGeomPDF(pdf_Ld, InLightGeom);
    float weight = power_heuristic(pdf_bsdf, pdf_Ld);
    OutContribution += directLight * bsdf * glm::max(0.f, glm::dot(wj.direction, InSurfaceNormal)) * weight / pdf_bsdf;    
}


__global__ void generateRayFromIntersections(int iter, int frame, int numPaths,
    PathSegment* pathSegments, ShadeableIntersection* dev_intersections,
    Scene::RenderProxy* scene,
    int* dev_pathAlive)
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
    Geom light_geom = scene->geoms_Device[scene->light_index_Device[0]];
	Material light_mat = scene->materials_Device[light_geom.materialid];
    if (intersection.materialId < 0) {
        path_segment.remainingBounces = 0;
        pathSegments[pathIndex] = path_segment;
        return;
    }
    Material material = scene->materials_Device[intersection.materialId];
	glm::vec3 p = intersection.intersectPoint + EPSILON * intersection.surfaceNormal;
    thrust::default_random_engine rng = makeSeededRandomEngine(frame, pathIndex, iter * path_segment.remainingBounces);
    if (material.emittance > 0.)
    {
		path_segment.Contribution += path_segment.BSDF * material.emittance / path_segment.PDF * path_segment.Cosine;
        path_segment.remainingBounces = 0;
        pathSegments[pathIndex] = path_segment;
        return;
    }
    glm::vec3 contrib;
    glm::vec3 debug;
	glm::vec3 ViewDir = -path_segment.ray.direction;
    SampleDirectLightMIS(debug, contrib, p, ViewDir, intersection.surfaceNormal, material,
        light_geom, light_mat, scene->geoms_size, scene->geoms_Device, rng);
	path_segment.Contribution += path_segment.BSDF * contrib / path_segment.PDF * path_segment.Cosine;
    Ray wi;
    glm::vec3 bsdf_at_p;
	float pdf_bsdf;
    //bsdfDiffuseSample(bsdf_at_p, pdf_bsdf, wi, p, intersection.surfaceNormal, &material, rng);
	bsdfPBRSample(debug, bsdf_at_p, pdf_bsdf, wi, ViewDir, p, intersection.surfaceNormal, intersection.uv, material, rng);
	path_segment.BSDF *= bsdf_at_p;
    path_segment.PDF *= pdf_bsdf;
    if (path_segment.PDF<EPSILON)
    {
        path_segment.remainingBounces = 0;
        pathSegments[pathIndex] = path_segment;
		return;
    }
	path_segment.Cosine *= glm::max(0.f, glm::dot(wi.direction, intersection.surfaceNormal));

    path_segment.ray = wi;
	path_segment.remainingBounces--;
    pathSegments[pathIndex] = path_segment;
}