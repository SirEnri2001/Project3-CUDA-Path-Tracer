#include "pathtrace.h"

#include <iostream>
#include <thrust/sort.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/remove.h>

#include "defs.h"
#include "bsdf.h"
#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "common.h"
#include "pathtraceImpl.h"

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

struct Pred {
	template<typename Tuple>
	__host__ __device__
	bool operator()(Tuple x)
	{
		return thrust::get<0>(x) == -1;
	}
};

void pathtrace(PTEngine* Engine, Scene* scene)
{
	const Camera& cam = scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	// 2D block for generating ray from camera
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// 1D block for path tracing
	const int blockSize1d = 64;
	auto& Resource = Engine->RenderResource;
	auto& State = scene->state;
	generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, State.frames, Engine->Info.depths, 
		Resource.dev_paths, Resource.device_pathAlive, Resource.dev_path_intersections);
	checkCUDAError("generate camera ray");
	PathSegment* dev_path_begin = Resource.dev_paths;
	int total_paths = pixelcount;
	int num_paths = pixelcount;
	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks
	for (int depth = 0; depth < Engine->Info.depths; depth++)
	{
#if USE_SORT_BY_BOUNDING
		dim3 numblocksPathSegmentTracing = (total_paths + blockSize1d - 1) / blockSize1d;
		{
			// tracing
			PreIntersect << <numblocksPathSegmentTracing, blockSize1d >> >(
				num_paths,
				dev_path_begin,
				scene->Proxy_Device,
				Resource.dev_path_intersections,
				Resource.dev_geom_ids, Resource.device_pathAlive
			);
			auto t = thrust::make_tuple(Resource.device_pathAlive, Resource.dev_geom_ids);
			auto zip_iter = thrust::make_zip_iterator(thrust::make_tuple(Resource.device_pathAlive, Resource.dev_geom_ids));
			auto end_iter = thrust::remove_if(thrust::device, zip_iter, zip_iter + num_paths, Pred());
			num_paths = end_iter - zip_iter;
			thrust::sort_by_key(
				thrust::device,
				Resource.dev_geom_ids,
				Resource.dev_geom_ids + num_paths,
				zip_iter);
			Intersect << <numblocksPathSegmentTracing, blockSize1d >> > (
				num_paths,
				dev_path_begin,
				scene->Proxy_Device,
				Resource.dev_path_intersections,
				Resource.dev_geom_ids, Resource.device_pathAlive
			);
		}
#else
		dim3 numblocksPathSegmentTracing = (total_paths + blockSize1d - 1) / blockSize1d;
		{
			// tracing
			computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
				depth,
				num_paths,
				dev_path_begin,
				scene->Proxy_Device,
				dev_path_intersections,
				dev_geom_ids, device_pathAlive, false
				);
			auto end_iter = thrust::remove(thrust::device, device_pathAlive, device_pathAlive + num_paths, -1);
			num_paths = end_iter - device_pathAlive;
		}
#endif

		std::cout << "number of paths: " << num_paths << std::endl;
		numblocksPathSegmentTracing = (total_paths + blockSize1d - 1) / blockSize1d;
		DirectLightingShadingPathSegments << <numblocksPathSegmentTracing, blockSize1d >> > (
			depth, State.frames, num_paths, dev_path_begin,
			Resource.dev_path_intersections,
			scene->Proxy_Device, Resource.device_pathAlive);
		SamplingShadingPathSegments << <numblocksPathSegmentTracing, blockSize1d >> >(
			depth, State.frames, num_paths, dev_path_begin,
			Resource.dev_path_intersections,
			scene->Proxy_Device, Resource.device_pathAlive);
		Engine->GuiData.TracedDepth = depth;
	}

	// Assemble this iteration and apply it to the image
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather<<<numBlocksPixels, blockSize1d>>>(total_paths, Resource.dev_image, Resource.dev_paths, Engine->GuiData.isDebug);
	if (Resource.pbo)
	{
		// Send results to OpenGL buffer for rendering
		sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (Resource.pbo, cam.resolution, State.frames+1, Resource.dev_image);
	}
	// Retrieve image from GPU
	cudaMemcpy(scene->state.image.data(), Resource.dev_image,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
	checkCUDAError("pathtrace");
}
