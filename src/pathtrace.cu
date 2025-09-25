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
static int* device_materialIdStart = nullptr;
static int* device_materialIdEnd = nullptr;
static int* device_pathAlive = nullptr;

// TODO: static variables for device memory, any extra info you need, etc
// ...

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
}

void pathtraceNewFrame(Scene* scene)
{
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_path_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(device_path_matIds, -1, pixelcount * sizeof(int));
    cudaMemset(device_pathAlive, 1, pixelcount * sizeof(int));
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
			if (scene->materials[scene->geoms[i].materialid].emittance>0.)
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
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_path_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_path_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // TODO: initialize any extra device memeory you need

	// allocate device memory for materialIds
	cudaMalloc(&device_path_matIds, pixelcount * sizeof(int));
	cudaMemset(device_path_matIds, -1, pixelcount * sizeof(int));

    cudaMalloc(&device_pathAlive, pixelcount * sizeof(int));
	cudaMemset(device_pathAlive, 1, pixelcount * sizeof(int));

	// allocate device memory for materialIdStart
	cudaMalloc(&device_materialIdStart, (scene->materials.size()+1) * sizeof(int));
	cudaMemset(device_materialIdStart, -1, (scene->materials.size() + 1) * sizeof(int));
	// allocate device memory for materialIdEnd
	cudaMalloc(&device_materialIdEnd, (scene->materials.size() + 1) * sizeof(int));
	cudaMemset(device_materialIdEnd, -1, (scene->materials.size() + 1) * sizeof(int));

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_path_intersections);
    // TODO: clean up any extra device memory you created

    checkCUDAError("pathtraceFree");
}


__global__ void calculateMaterialStartEndId(int numPaths, int* device_path_matIds, int* device_matStart, int* device_matEnd)
{
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid >= numPaths)
    {
        return;
	}

	int matId = device_path_matIds[tid]+1;
    if (tid > 0)
    {
        int prevMatId = device_path_matIds[tid - 1];
        if (matId != prevMatId)
        {
            device_matStart[matId] = tid;
        }
	}
    if (tid < numPaths-1)
    {
        int nextMatId = device_path_matIds[tid + 1];
        if (matId != nextMatId)
        {
            device_matEnd[matId] = tid+1;
        }
	}

    if (tid==0)
    {
		device_matStart[matId] = 0;
    }
    if (tid == numPaths)
    {
        device_matEnd[matId] = tid+1;
    }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        if (iterationPath.remainingBounces==0)
        {
            image[iterationPath.pixelIndex] += iterationPath.Contribution;
        }
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
    const int traceDepth = 10;// = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

    // TODO: perform one iteration of path tracing

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
	PathSegment* dev_path_begin = dev_paths;
    int num_paths = dev_path_end - dev_paths;
    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    int deadPaths = 0;
    for (int i = 0; i < iter; i++)
    {
        // clean shading chunks
        cudaMemset(dev_path_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
#if SORT_RAYS
        dim3 numblocksPathSegmentTracing = (num_paths - deadPaths + blockSize1d - 1) / blockSize1d;
        {
            // tracing
            computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
                depth,
                num_paths - deadPaths,
                dev_path_begin + deadPaths,
                dev_geoms,
                hst_scene->geoms.size(),
                dev_path_intersections + deadPaths,
                device_path_matIds + deadPaths
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
            int newDeadPaths = 0;
            //auto end_iter = thrust::remove_if(thrust::device, )
            thrust::sort_by_key(
                thrust::device, 
                device_path_matIds + deadPaths, 
                device_path_matIds + num_paths,
                thrust::make_zip_iterator(
                    thrust::make_tuple(dev_path_intersections + deadPaths, dev_paths + deadPaths)
                ));

            newDeadPaths = thrust::count(thrust::device, device_path_matIds + deadPaths, device_path_matIds + num_paths, -1);
			deadPaths += newDeadPaths;
		}
		std::cout << "number of dead paths: " << deadPaths << std::endl;
        if (deadPaths==num_paths)
        {
            std::cout << "no valid paths, shown black screen" << deadPaths << std::endl;
            break;
        }
        numblocksPathSegmentTracing = (num_paths - deadPaths + blockSize1d - 1) / blockSize1d;
        generateRayFromIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
            iter, num_paths - deadPaths, dev_path_begin + deadPaths, 
            dev_path_intersections + deadPaths, dev_materials,
            hst_scene->geoms.size(), dev_geoms, device_light_geoms);
#else
	    // tracing
	    dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
	    computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
	        depth,
	        num_paths,
	        dev_path_begin,
	        dev_geoms,
	        hst_scene->geoms.size(),
	        dev_path_intersections,
	        device_path_matIds
	        );
        generateRayFromIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter, num_paths, dev_path_begin, dev_path_intersections, dev_materials);
#endif
        // TODO:
        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.

        //shadeFakeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
        //    iter,
        //    num_paths,
        //    dev_path_intersections,
        //    dev_paths,
        //    dev_materials
        //);

        depth++;
        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(num_paths, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
