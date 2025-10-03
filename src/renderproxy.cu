#include "renderproxy.h"

#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include "common.h"
#include "geometry.h"
#include "mesh.h"

__device__ glm::vec3 GetColorDevice(const Texture::RenderProxy& InTexture, glm::vec2 uv)
{
	uv.y = 1.f - uv.y; // Flip V to match image coordinate system
	glm::vec2 TexCoord = uv * glm::vec2(InTexture.Extent.x, InTexture.Extent.y);
	int x = static_cast<int>(TexCoord.x) % InTexture.Extent.x;
	int y = static_cast<int>(TexCoord.y) % InTexture.Extent.y;
	return InTexture.Image_Device[x + y * InTexture.Extent.x];
}

#if USE_OPTIMIZED_GRID
__device__ void GetTriangleBound(int& bound1, int& bound2, glm::vec3 p1, glm::vec3 p2, glm::vec3 p3, int layer, int optimizedLayer) // p1 p2 p3 within [0,1]
{
    int bound = -1;
    while (layer < GRID_LAYERS - 1)
    {
        int boundP1 = GetPointBoundNextLayer(p1);
        int boundP2 = GetPointBoundNextLayer(p2);
        int boundP3 = GetPointBoundNextLayer(p3);
        if (boundP1 != boundP2 && boundP1 != boundP3 && boundP1 != boundP3)
        {
	        // this triangle cross three bounds
            break;
        }

        if (boundP1 != boundP2 || boundP1 != boundP3)
        {
            if (layer>optimizedLayer)
            {
                break;
            }
            if (boundP1!=boundP2)
            {
	            bound1 = 8 * (bound + 1) + boundP1;
                bound2 = 8 * (bound + 1) + boundP2;
            }else
            {
                bound1 = 8 * (bound + 1) + boundP1;
                bound2 = 8 * (bound + 1) + boundP3;
            }
            return;
        }
        bound = 8 * (bound + 1) + boundP1;
        p1 = glm::fract(p1 * 2.f);
        p2 = glm::fract(p2 * 2.f);
        p3 = glm::fract(p3 * 2.f);
        layer++;
    }
    bound1 = bound;
    bound2 = GRID_SIZE-1;
}
#else
__device__ void GetTriangleBound(int& bound1, int& bound2, glm::vec3 p1, glm::vec3 p2, glm::vec3 p3, int layer, int optimizedLayer) // p1 p2 p3 within [0,1]
{
    int bound = -1;
    while (layer < GRID_LAYERS - 1)
    {
        int boundP1 = GetPointBoundNextLayer(p1);
        int boundP2 = GetPointBoundNextLayer(p2);
        int boundP3 = GetPointBoundNextLayer(p3);

        if (boundP1 != boundP2 || boundP1 != boundP3)
        {
            break;
        }
        bound = 8 * (bound + 1) + boundP1;
        p1 = glm::fract(p1 * 2.f);
        p2 = glm::fract(p2 * 2.f);
        p3 = glm::fract(p3 * 2.f);
        layer++;
    }
    bound1 = bound;
    bound2 = GRID_SIZE;
}
#endif

__global__ void calculateMeshGridSpeedup(StaticMesh::RenderProxy* InMeshData)
{
    int triangleId = (blockIdx.x * blockDim.x) + threadIdx.x;
    int triangleCount = InMeshData->VertexCount / 3;
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
    int b1, b2;
    GetTriangleBound(b1, b2, p1, p2, p3, 0, 1);

#if USE_OPTIMIZED_GRID
    InMeshData->raw.TriangleToGridIndices_Device[triangleId] = b1;
    InMeshData->raw.TriangleToGridIndices_Device[triangleId + triangleCount] = b2;
#else
    InMeshData->raw.TriangleToGridIndices_Device[triangleId] = b1;
#endif
}

__global__ void calculateGridStartEndIndices(int triangleSize, int* InSortedGridIndices, int* OutGridIndicesStart, int* OutGridIndicesEnd)
{
    int triangleId = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (triangleId > triangleSize - 1)
    {
        return;
    }
    int index1 = InSortedGridIndices[triangleId];

    if (triangleId == triangleSize - 1)
    {
        OutGridIndicesEnd[index1 + 1] = triangleSize;
        return;
    }
    int index2 = InSortedGridIndices[triangleId + 1];
    if (triangleId == 0)
    {
        OutGridIndicesStart[0] = 0;
    }
    if (index1 != index2)
    {
        // plus one because grid index start with -1
        OutGridIndicesStart[index2 + 1] = triangleId + 1;
        OutGridIndicesEnd[index1 + 1] = triangleId + 1;
    }
}

__host__ void OutputDebug(StaticMesh::RenderProxy* Proxy_Host)
{
    int* indicesStart = new int[GRID_SIZE];
    int* indicesEnd = new int[GRID_SIZE];
    int emptyGrids = 0;
    int usedGrids = 0;
    int maxObjects = 0;
    int totalObjects = 0;

    cudaMemcpy(indicesStart,Proxy_Host->raw.GridIndicesStart_Device, sizeof(int) * GRID_SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(indicesEnd, Proxy_Host->raw.GridIndicesEnd_Device, sizeof(int) * GRID_SIZE, cudaMemcpyDeviceToHost);
    for (int i = 0; i < GRID_SIZE; i++)
    {
        if (indicesStart[i]==-1)
        {
            emptyGrids++;
            continue;
        }
        usedGrids++;
        maxObjects = glm::max(maxObjects, indicesEnd[i] - indicesStart[i]);
        totalObjects += indicesEnd[i] - indicesStart[i];
    }
    std::cout << "====<<< Used Grid: " << usedGrids << " Empty Grid: " << emptyGrids << " Max Objects: " << maxObjects << " Object / Grid: " << (float)totalObjects / usedGrids << " >>>===" << std::endl;
    delete[] indicesStart;
    delete[] indicesEnd;
}

__host__ void CalculateSpeedUpOctreeForStaticMesh(StaticMesh* Mesh)
{
    dim3 Grid;
    dim3 BlockSize;
    BlockSize.x = 128;
    Grid.x = (Mesh->Data.VertexCount / 3 + BlockSize.x - 1) / BlockSize.x;
    calculateMeshGridSpeedup << <Grid, BlockSize >> > (Mesh->Proxy_Device);
    checkCUDAError("calculateMeshGridSpeedup");
    int useOptimizedGrid = 1;
#if USE_OPTIMIZED_GRID
    useOptimizedGrid = 2;
#endif
    Grid.x = (Mesh->Data.VertexCount / 3 * useOptimizedGrid + BlockSize.x - 1) / BlockSize.x;
    auto tuple = thrust::make_tuple(Mesh->Proxy_Host->raw.TriangleToGridIndices_Device, Mesh->Proxy_Host->raw.TriangleIndices_Device);
    thrust::sort_by_key(
        thrust::device,
        Mesh->Proxy_Host->raw.TriangleToGridIndices_Device,
        Mesh->Proxy_Host->raw.TriangleToGridIndices_Device + Mesh->Data.VertexCount / 3 * useOptimizedGrid,
        thrust::make_zip_iterator(tuple));
    calculateGridStartEndIndices << <Grid, BlockSize >> > (
        Mesh->Data.VertexCount / 3 * useOptimizedGrid,
        Mesh->Proxy_Host->raw.TriangleToGridIndices_Device,
        Mesh->Proxy_Host->raw.GridIndicesStart_Device,
        Mesh->Proxy_Host->raw.GridIndicesEnd_Device);
    checkCUDAError("calculateGridStartEndIndices");
    OutputDebug(Mesh->Proxy_Host);
}

void StaticMeshManager::CalculateOctreeStructureCUDA()
{
    // Create Proxy for all meshes
    for (auto& NameMesh : Meshes)
    {
        auto* Mesh = NameMesh.second.get();
        CalculateSpeedUpOctreeForStaticMesh(Mesh);
    }
}
