#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <iostream>
#include <thrust/random.h>
#include <thrust/sort.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/remove.h>

#include "bsdf.h"
#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"
#include "common.h"
#include "pathtraceImpl.h"
#include "mesh.h"

#define SORT_RAYS 1

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y)
	{
		int index = x + (y * resolution.x);
		glm::vec3 pix = image[index];

		glm::ivec3 color;
		color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
		color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
		color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

		// Each thread writes one pixel location in the texture (textel)
		pbo[index].w = 0;
		pbo[index].x = color.x;
		pbo[index].y = color.y;
		pbo[index].z = color.z;
	}
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
int lightCount = 0;
static Geom* device_light_geoms = nullptr;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_path_intersections = NULL;
static int* device_path_matIds = nullptr;
static int* device_pathAlive = nullptr;
static StaticMeshData_Device host_object_staticMeshData;
static StaticMeshData_Device* device_staticMeshData;
static StaticMeshData_Host staticMeshData;
std::string MeshPath = "../models/cube.obj";


void InitDataContainer(GuiDataContainer* imGuiData)
{
	guiData = imGuiData;
}

void pathtraceNewFrame(Scene* scene)
{
	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;
	cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_path_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(device_path_matIds, -1, pixelcount * sizeof(int));
	cudaMemset(device_pathAlive, -1, pixelcount * sizeof(int));
}

void pathtraceCreate(Scene* scene)
{
	hst_scene = scene;

	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

	{
		// populate device_light_geoms
		std::vector<Geom> light_geoms;
		for (int i = 0; i < scene->geoms.size(); i++)
		{
			if (scene->materials[scene->geoms[i].materialid].emittance > 0.)
			{
				light_geoms.push_back(scene->geoms[i]);
			}
		}
		lightCount = light_geoms.size();
		std::cout << "number of light geoms: " << lightCount << std::endl;
		cudaMalloc(&device_light_geoms, lightCount * sizeof(Geom));
		cudaMemcpy(device_light_geoms, light_geoms.data(), lightCount * sizeof(Geom), cudaMemcpyHostToDevice);
	}

	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material),
	           cudaMemcpyHostToDevice);

	cudaMalloc(&dev_path_intersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_path_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
	// allocate device memory for materialIds
	cudaMalloc(&device_path_matIds, pixelcount * sizeof(int));
	cudaMemset(device_path_matIds, -1, pixelcount * sizeof(int));

	cudaMalloc(&device_pathAlive, pixelcount * sizeof(int));
	cudaMemset(device_pathAlive, -1, pixelcount * sizeof(int));
	checkCUDAError("pathtraceInit");

	ReadObjMesh(staticMeshData, MeshPath);
	CreateDeviceObject(&device_staticMeshData, host_object_staticMeshData, staticMeshData);
}

void pathtraceFree()
{
	cudaFree(dev_image); // no-op if dev_image is null
	cudaFree(dev_paths);
	cudaFree(dev_geoms);
	cudaFree(dev_materials);
	cudaFree(dev_path_intersections);
	checkCUDAError("pathtraceFree");
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
		image[iterationPath.pixelIndex] += iterationPath.Contribution;
	}
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */

void pathtrace(uchar4* pbo, int frame, int iter)
{
	static int* materialIdStart = new int[hst_scene->materials.size()];
	static int* materialIdEnd = new int[hst_scene->materials.size()];
	const int traceDepth = 10; // = hst_scene->state.traceDepth;
	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	// 2D block for generating ray from camera
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// 1D block for path tracing
	const int blockSize1d = 128;
	generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths, device_pathAlive);
	checkCUDAError("generate camera ray");
	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	PathSegment* dev_path_begin = dev_paths;
	int total_paths = pixelcount;
	int num_paths = pixelcount;
	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks
	std::cout << "New Frame" << std::endl;
	for (int i = 0; i < 2; i++)
	{
		// clean shading chunks
		cudaMemset(dev_path_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

		dim3 numblocksPathSegmentTracing = (total_paths + blockSize1d - 1) / blockSize1d;
		{
			// tracing
			computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> >(
				depth,
				num_paths,
				dev_path_begin,
				dev_geoms,
				hst_scene->geoms.size(),
				dev_path_intersections,
				device_path_matIds, device_pathAlive, device_staticMeshData
			);
		}
		{
			struct isPathAlive
			{
				__host__ __device__
				bool operator()(const int x)
				{
					return (x % 2) == 0;
				}
			};
			auto end_iter = thrust::remove(thrust::device, device_pathAlive, device_pathAlive + num_paths, -1);
			num_paths = end_iter - device_pathAlive;
//#if SORT_RAYS
//TODO sorting rays by material id
//			thrust::sort_by_key(
//				thrust::device,
//				device_path_matIds,
//				device_path_matIds + num_paths,
//				thrust::make_zip_iterator(
//					thrust::make_tuple(dev_path_intersections, dev_paths, device_pathAlive)
//				));
//#endif
		}
		std::cout << "number of paths: " << num_paths << std::endl;
		numblocksPathSegmentTracing = (total_paths + blockSize1d - 1) / blockSize1d;
		generateRayFromIntersections << <numblocksPathSegmentTracing, blockSize1d >> >(
			iter, num_paths, dev_path_begin,
			dev_path_intersections, dev_materials,
			hst_scene->geoms.size(), dev_geoms, device_light_geoms, device_pathAlive);
		depth++;
		if (guiData != NULL)
		{
			guiData->TracedDepth = depth;
		}
	}

	// Assemble this iteration and apply it to the image
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather<<<numBlocksPixels, blockSize1d>>>(total_paths, dev_image, dev_paths);

	// Send results to OpenGL buffer for rendering
	sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image,
	           pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	checkCUDAError("pathtrace");
}
