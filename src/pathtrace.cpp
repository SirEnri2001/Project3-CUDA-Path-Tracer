#include "pathtrace.h"
#include "common.h"

PTEngine* PTEngine::GInstance = nullptr;

PTEngine::PTEngine(PathTraceInfo InitInfo):Info(InitInfo)
{
	assert(GInstance == nullptr);
	GInstance = this;
}

PTEngine* PTEngine::Get()
{
	return GInstance;
}


void PTEngine::ClearBuffers()
{
	const int pixelcount = Info.x * Info.y;
	cudaDeviceSynchronize();
	cudaMemset(RenderResource.dev_image, 0, pixelcount * sizeof(glm::vec3));
	cudaMemset(RenderResource.dev_path_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(RenderResource.dev_geom_ids, -1, pixelcount * sizeof(int));
	cudaMemset(RenderResource.device_pathAlive, -1, pixelcount * sizeof(int));
}


void PTEngine::Init()
{
	const int pixelcount = Info.x * Info.y;

	checkCUDAError("before pathtraceInit");
	cudaMalloc(&RenderResource.dev_image, pixelcount * sizeof(glm::vec3));
	cudaMemset(RenderResource.dev_image, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&RenderResource.dev_paths, pixelcount * sizeof(PathSegment));

	cudaMalloc(&RenderResource.dev_path_intersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(RenderResource.dev_path_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
	// allocate device memory for materialIds
	cudaMalloc(&RenderResource.dev_geom_ids, pixelcount * sizeof(int));
	cudaMemset(RenderResource.dev_geom_ids, -1, pixelcount * sizeof(int));

	cudaMalloc(&RenderResource.device_pathAlive, pixelcount * sizeof(int));
	cudaMemset(RenderResource.device_pathAlive, -1, pixelcount * sizeof(int));
	checkCUDAError("pathtraceInit");
}


void PTEngine::Destroy()
{
	cudaFree(RenderResource.dev_image); // no-op if dev_image is null
	cudaFree(RenderResource.dev_paths);
	cudaFree(RenderResource.dev_path_intersections);
	checkCUDAError("pathtraceFree");
}

void PTEngine::Tick(Scene* scene)
{
	pathtrace(this, scene);
}
