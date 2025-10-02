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

__device__ int GetTriangleBound(glm::vec3 p1, glm::vec3 p2, glm::vec3 p3, int layer) // p1 p2 p3 within [0,1]
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
    return bound;
}

__global__ void calculateMeshGridSpeedup(StaticMesh::RenderProxy* InMeshData)
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
    InMeshData->raw.TriangleToGridIndices_Device[triangleId] = bound;
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

__host__ void CalculateSpeedUpOctreeForStaticMesh(StaticMesh* Mesh)
{
    dim3 Grid;
    dim3 BlockSize;
    BlockSize.x = 128;
    Grid.x = (Mesh->Data.VertexCount / 3 + BlockSize.x - 1) / BlockSize.x;
    calculateMeshGridSpeedup << <Grid, BlockSize >> > (Mesh->Proxy_Device);
    checkCUDAError("calculateMeshGridSpeedup");

    auto tuple = thrust::make_tuple(Mesh->Proxy_Host->raw.TriangleToGridIndices_Device, Mesh->Proxy_Host->raw.TriangleIndices_Device);
    thrust::sort_by_key(
        thrust::device,
        Mesh->Proxy_Host->raw.TriangleToGridIndices_Device,
        Mesh->Proxy_Host->raw.TriangleToGridIndices_Device + Mesh->Data.VertexCount / 3,
        thrust::make_zip_iterator(tuple));
    calculateGridStartEndIndices << <Grid, BlockSize >> > (
        Mesh->Data.VertexCount / 3,
        Mesh->Proxy_Host->raw.TriangleToGridIndices_Device,
        Mesh->Proxy_Host->raw.GridIndicesStart_Device,
        Mesh->Proxy_Host->raw.GridIndicesEnd_Device);
    checkCUDAError("calculateGridStartEndIndices");
    int* indices = new int[GRID_SIZE];
    cudaMemcpy(indices, Mesh->Proxy_Host->raw.GridIndicesStart_Device, sizeof(int) * GRID_SIZE, cudaMemcpyDeviceToHost);
    for (int i = 0; i < GRID_SIZE; i++)
    {
        std::cout << indices[i] << ", ";
        if (i % 20 == 0)
        {
            std::cout << std::endl;
        }
    }
    std::cout << "============" << std::endl;
    cudaMemcpy(indices, Mesh->Proxy_Host->raw.GridIndicesEnd_Device, sizeof(int) * GRID_SIZE, cudaMemcpyDeviceToHost);
    for (int i = 0; i < GRID_SIZE; i++)
    {
        std::cout << indices[i] << ", ";
        if (i % 20 == 0)
        {
            std::cout << std::endl;
        }
    }
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
