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
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"
#include "common.h"
#include "pathtraceImpl.h"
#include "mesh.h"
#include "material.h"

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

static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_path_intersections = NULL;
static int* device_path_matIds = nullptr;
static int* device_pathAlive = nullptr;

void InitDataContainer(GuiDataContainer* imGuiData)
{
	guiData = imGuiData;
}

void pathtraceNewFrame(Scene* scene)
{
	const Camera& cam = scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;
	cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_path_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(device_path_matIds, -1, pixelcount * sizeof(int));
	cudaMemset(device_pathAlive, -1, pixelcount * sizeof(int));
}


void pathtraceCreate(Scene* scene)
{
	const Camera& cam = scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	checkCUDAError("before pathtraceInit");
	cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

	cudaMalloc(&dev_path_intersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_path_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
	// allocate device memory for materialIds
	cudaMalloc(&device_path_matIds, pixelcount * sizeof(int));
	cudaMemset(device_path_matIds, -1, pixelcount * sizeof(int));

	cudaMalloc(&device_pathAlive, pixelcount * sizeof(int));
	cudaMemset(device_pathAlive, -1, pixelcount * sizeof(int));
	checkCUDAError("pathtraceInit");

	scene->CreateRenderProxyForAll();
	scene->CenterCamera();
	size_t stack_size = 8192; // 8KB
	cudaDeviceSetLimit(cudaLimitStackSize, stack_size);
}


void pathtraceFree()
{
	cudaFree(dev_image); // no-op if dev_image is null
	cudaFree(dev_paths);
	cudaFree(dev_path_intersections);
	checkCUDAError("pathtraceFree");
}

__host__ __device__
bool is_nan(glm::vec3 v)
{
	return cuda::std::isnan(v.x) || cuda::std::isnan(v.y) || cuda::std::isnan(v.z);
}

__host__ __device__
bool is_inf(glm::vec3 v)
{
	return cuda::std::isinf(v.x) || cuda::std::isinf(v.y) || cuda::std::isinf(v.z);
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths, bool debug = false)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
		bool bInvalidVal = false;
		bInvalidVal = is_nan(iterationPath.Contribution);
		bInvalidVal = bInvalidVal || is_inf(iterationPath.Contribution);
		if (bInvalidVal)
		{
			iterationPath.debug = glm::vec3(0, 1, 0);
		}
		if (debug)
		{
			iterationPath.Contribution = iterationPath.debug;
		}
		image[iterationPath.pixelIndex] += iterationPath.Contribution;
	}
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */

void pathtrace(Scene* scene, uchar4* pbo, int frame, int iter)
{
	const int traceDepth = 10; // = hst_scene->state.traceDepth;
	const Camera& cam = scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	// 2D block for generating ray from camera
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// 1D block for path tracing
	const int blockSize1d = 128;
	generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, frame, iter, dev_paths, device_pathAlive);
	checkCUDAError("generate camera ray");
	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	PathSegment* dev_path_begin = dev_paths;
	int total_paths = pixelcount;
	int num_paths = pixelcount;
	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks
	std::cout << "New Frame" << std::endl;
	for (int i = 0; i < 1; i++)
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
				scene->Proxy_Device,
				dev_path_intersections,
				device_path_matIds, device_pathAlive
			);
		}
		{
			//struct isPathAlive
			//{
			//	__host__ __device__
			//	bool operator()(const int x)
			//	{
			//		return (x % 2) == 0;
			//	}
			//};
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
			iter, frame, num_paths, dev_path_begin,
			dev_path_intersections,
			scene->Proxy_Device, device_pathAlive);
		depth++;
		if (guiData != NULL)
		{
			guiData->TracedDepth = depth;
		}
	}

	// Assemble this iteration and apply it to the image
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather<<<numBlocksPixels, blockSize1d>>>(total_paths, dev_image, dev_paths, guiData->isDebug);

	// Send results to OpenGL buffer for rendering
	sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

	// Retrieve image from GPU
	cudaMemcpy(scene->state.image.data(), dev_image,
	           pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	checkCUDAError("pathtrace");
}
