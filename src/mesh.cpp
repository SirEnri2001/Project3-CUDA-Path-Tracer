#include "mesh.h"

#include <iostream>
#define TINYOBJLOADER_IMPLEMENTATION
#include "common.h"
#include "MeshLoader/tiny_gltf.h"
#include "MeshLoader/tiny_obj_loader.h"

void LoadObjImpl(StaticMesh& Mesh, std::string FilePath)
{
    tinyobj::ObjReaderConfig reader_config;
    reader_config.mtl_search_path = "./"; // Path to material files

    tinyobj::ObjReader reader;

    if (!reader.ParseFromFile(FilePath, reader_config)) {
        if (!reader.Error().empty()) {
            std::cerr << "TinyObjReader: " << reader.Error();
        }
        exit(1);
    }

    if (!reader.Warning().empty()) {
        std::cout << "TinyObjReader: " << reader.Warning();
    }

    auto& attrib = reader.GetAttrib();
    auto& shapes = reader.GetShapes();
    auto& materials = reader.GetMaterials();

    // Loop over shapes
    for (size_t s = 0; s < 1/*shapes.size()*/; s++) {
        // Loop over faces(polygon)
        size_t index_offset = 0;

        // allocate memory
        Mesh.Data.VertexCount = shapes[s].mesh.num_face_vertices.size() * 3; // assume triangulated
        Mesh.Data.VertexPosition_Host.resize(Mesh.Data.VertexCount);
        Mesh.Data.VertexNormal_Host.resize(Mesh.Data.VertexCount);
        Mesh.Data.VertexColor_Host.resize(Mesh.Data.VertexCount);
        Mesh.Data.VertexTexCoord_Host.resize(Mesh.Data.VertexCount);

        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);

            // Loop over vertices in the face.
            for (size_t v = 0; v < 3/*fv*/; v++) {
                // access to vertex
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                tinyobj::real_t vx = attrib.vertices[3 * size_t(idx.vertex_index) + 0];
                tinyobj::real_t vy = attrib.vertices[3 * size_t(idx.vertex_index) + 1];
                tinyobj::real_t vz = attrib.vertices[3 * size_t(idx.vertex_index) + 2];

                Mesh.Data.boxMin = glm::min(Mesh.Data.boxMin, glm::vec3(vx, vy, vz));
                Mesh.Data.boxMax = glm::max(Mesh.Data.boxMax, glm::vec3(vx, vy, vz));

                Mesh.Data.VertexPosition_Host[3 * f + v] = glm::vec3(vx, vy, vz);
                // Check if `normal_index` is zero or positive. negative = no normal data
                if (idx.normal_index >= 0) {
                    tinyobj::real_t nx = attrib.normals[3 * size_t(idx.normal_index) + 0];
                    tinyobj::real_t ny = attrib.normals[3 * size_t(idx.normal_index) + 1];
                    tinyobj::real_t nz = attrib.normals[3 * size_t(idx.normal_index) + 2];
                    Mesh.Data.VertexNormal_Host[3 * f + v] = glm::vec3(nx, ny, nz);
                }

                // Check if `texcoord_index` is zero or positive. negative = no texcoord data
                if (idx.texcoord_index >= 0) {
                    tinyobj::real_t tx = attrib.texcoords[2 * size_t(idx.texcoord_index) + 0];
                    tinyobj::real_t ty = attrib.texcoords[2 * size_t(idx.texcoord_index) + 1];
                    Mesh.Data.VertexTexCoord_Host[3 * f + v] = glm::vec2(tx, ty);
                }
            }
            index_offset += fv;

            // per-face material
            shapes[s].mesh.material_ids[f];
        }
    }
}

void LoadGLTFImpl(StaticMesh& Mesh, const tinygltf::Mesh& gltfMesh, const tinygltf::Model& model)
{
    for (size_t j = 0; j < 1 /*gltfMesh.primitives.size()*/; ++j) {
        const tinygltf::Primitive& primitive = gltfMesh.primitives[j];
        if (primitive.mode != TINYGLTF_MODE_TRIANGLES) {
            std::cout << "Only triangle mode is supported!" << std::endl;
            continue;
        }
        const tinygltf::Accessor& posAccessor = model.accessors[primitive.attributes.find("POSITION")->second];
        const tinygltf::BufferView& posView = model.bufferViews[posAccessor.bufferView];
        const tinygltf::Buffer& posBuffer = model.buffers[posView.buffer];
        const tinygltf::Accessor& normAccessor = model.accessors[primitive.attributes.find("NORMAL")->second];
        const tinygltf::BufferView& normView = model.bufferViews[normAccessor.bufferView];
        const tinygltf::Buffer& normBuffer = model.buffers[normView.buffer];
        const tinygltf::Accessor& texAccessor = model.accessors[primitive.attributes.find("TEXCOORD_0")->second];
        const tinygltf::BufferView& texView = model.bufferViews[texAccessor.bufferView];
        const tinygltf::Buffer& texBuffer = model.buffers[texView.buffer];
        const tinygltf::Accessor& indexAccessor = model.accessors[primitive.indices];
        const tinygltf::BufferView& indexView = model.bufferViews[indexAccessor.bufferView];
        const tinygltf::Buffer& indexBuffer = model.buffers[indexView.buffer];
        Mesh.Data.VertexCount = static_cast<unsigned int>(indexAccessor.count);
        Mesh.Data.VertexPosition_Host.resize(Mesh.Data.VertexCount);
        Mesh.Data.VertexNormal_Host.resize(Mesh.Data.VertexCount);
        Mesh.Data.VertexTexCoord_Host.resize(Mesh.Data.VertexCount);

        // read triangles
        size_t posStride = posView.byteStride ? posView.byteStride / sizeof(float) : 3;
        size_t normStride = normView.byteStride ? normView.byteStride / sizeof(float) : 3;
        size_t texStride = texView.byteStride ? texView.byteStride / sizeof(float) : 2;
        const float* positionData = reinterpret_cast<const float*>(&posBuffer.data[posView.byteOffset + posAccessor.byteOffset]);
        const float* normalData = reinterpret_cast<const float*>(&normBuffer.data[normView.byteOffset + normAccessor.byteOffset]);
        const float* texCoordData = reinterpret_cast<const float*>(&texBuffer.data[texView.byteOffset + texAccessor.byteOffset]);

        std::vector<size_t> indices(indexAccessor.count);

        if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT)
        {
            const uint16_t* InIndices = reinterpret_cast<const uint16_t*>(&indexBuffer.data[indexView.byteOffset + indexAccessor.byteOffset]);
            for (size_t i = 0; i < indexAccessor.count; ++i) {
                indices[i] = InIndices[i];
            }
        }
        else if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT)
        {
            const uint32_t* InIndices = reinterpret_cast<const uint32_t*>(&indexBuffer.data[indexView.byteOffset + indexAccessor.byteOffset]);
            for (size_t i = 0; i < indexAccessor.count; ++i) {
                indices[i] = InIndices[i];
            }
        }
        else if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE)
        {
            const uint8_t* InIndices = reinterpret_cast<const uint8_t*>(&indexBuffer.data[indexView.byteOffset + indexAccessor.byteOffset]);
            for (size_t i = 0; i < indexAccessor.count; ++i) {
                indices[i] = InIndices[i];
            }
        }
        else {
            std::cout << "Unsupported index component type!" << std::endl;
            continue;
        }
        assert(indices.size() % 3 == 0);
        // store data to staticmesh object
        for (size_t tid = 0; tid < indices.size(); tid += 3) {
            for (size_t indId = tid; indId < tid + 3; indId++)
            {
                Mesh.Data.VertexPosition_Host[3 * tid + indId] = {
                    positionData[indices[tid + indId] * posStride],
                    positionData[indices[tid + indId] * posStride + 1],
                    positionData[indices[tid + indId] * posStride + 2]
                };

                Mesh.Data.VertexNormal_Host[3 * tid + indId] = {
                    normalData[indices[tid + indId] * normStride],
                    normalData[indices[tid + indId] * normStride + 1],
                    normalData[indices[tid + indId] * normStride + 2]
                };

                Mesh.Data.VertexTexCoord_Host[3 * tid + indId] = {
                texCoordData[indices[tid + indId] * texStride],
                texCoordData[indices[tid + indId] * texStride + 1]
                };
            }
        }

        for (size_t v = 0; v < Mesh.Data.VertexCount; ++v) {
            Mesh.Data.boxMin = glm::min(Mesh.Data.boxMin, Mesh.Data.VertexPosition_Host[v]);
            Mesh.Data.boxMax = glm::max(Mesh.Data.boxMax, Mesh.Data.VertexPosition_Host[v]);
        }

    }
}

StaticMesh::StaticMesh(unsigned int VCount, glm::vec3 InBoxMin, glm::vec3 InBoxMax):
Data()
{
    Data.VertexCount = VCount;
    Data.boxMin = InBoxMin;
    Data.boxMax = InBoxMax;
    Data.VertexPosition_Host.resize(Data.VertexCount);
    Data.VertexNormal_Host.resize(Data.VertexCount);
    Data.VertexColor_Host.resize(Data.VertexCount);
    Data.VertexTexCoord_Host.resize(Data.VertexCount);
}

StaticMesh::StaticMesh()
{
	Proxy_Host = nullptr;
}


StaticMesh::~StaticMesh()
{
	if (Proxy_Host!=nullptr)
	{
        DestroyProxy();
	}
}

void StaticMesh::CreateProxy()
{
    Proxy_Host = new RenderProxy();
    checkCUDAError("StaticMesh::CreateProxy Start");
    Proxy_Host->VertexCount = Data.VertexCount;
    Proxy_Host->boxMin = Data.boxMin;
    Proxy_Host->boxMax = Data.boxMax;

    // Set mesh props
    cudaMalloc((void**)&Proxy_Host->raw.VertexPosition_Device, sizeof(glm::vec3) * Data.VertexCount);
    cudaMemcpy(Proxy_Host->raw.VertexPosition_Device, Data.VertexPosition_Host.data(), sizeof(glm::vec3) * Data.VertexCount, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&Proxy_Host->raw.VertexTexCoord_Device, sizeof(glm::vec3) * Data.VertexCount);
    cudaMemcpy(Proxy_Host->raw.VertexTexCoord_Device, Data.VertexTexCoord_Host.data(), sizeof(glm::vec2) * Data.VertexCount, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&Proxy_Host->raw.VertexNormal_Device, sizeof(glm::vec3) * Data.VertexCount);
    cudaMemcpy(Proxy_Host->raw.VertexNormal_Device, Data.VertexNormal_Host.data(), sizeof(glm::vec3) * Data.VertexCount, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&Proxy_Host->raw.VertexColor_Device, sizeof(glm::vec3) * Data.VertexCount);
    cudaMemcpy(Proxy_Host->raw.VertexColor_Device, Data.VertexColor_Host.data(), sizeof(glm::vec3) * Data.VertexCount, cudaMemcpyHostToDevice);

    // Create memory for accelerate structure
    cudaMalloc((void**)&Proxy_Host->raw.TriangleToGridIndices_Device, sizeof(int) * Data.VertexCount);
    cudaMemset(Proxy_Host->raw.TriangleToGridIndices_Device, -1, sizeof(int) * Data.VertexCount / 3);

    cudaMalloc((void**)&Proxy_Host->raw.GridIndicesStart_Device, sizeof(int) * GRID_SIZE);
    cudaMemset(Proxy_Host->raw.GridIndicesStart_Device, -1, sizeof(int) * GRID_SIZE);

	cudaMalloc((void**)&Proxy_Host->raw.GridIndicesEnd_Device, sizeof(int) * GRID_SIZE);
    cudaMemset(Proxy_Host->raw.GridIndicesEnd_Device, -1, sizeof(int) * GRID_SIZE);

    cudaMalloc((void**)&Proxy_Host->raw.TriangleIndices_Device, sizeof(int) * Proxy_Host->VertexCount / 3);
    std::vector<int> InitTriangleIndices(Proxy_Host->VertexCount / 3);
    for (int i = 0; i < Proxy_Host->VertexCount / 3; i++)
    {
        InitTriangleIndices[i] = i;
    }
    cudaMemcpy(Proxy_Host->raw.TriangleIndices_Device, InitTriangleIndices.data(), sizeof(int) * Proxy_Host->VertexCount / 3, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&Proxy_Device, sizeof(RenderProxy));
    cudaMemcpy(Proxy_Device, Proxy_Host, sizeof(RenderProxy), cudaMemcpyHostToDevice);
    checkCUDAError("StaticMesh::CreateProxy End");
}

void StaticMesh::DestroyProxy()
{
    cudaFree(Proxy_Device);
    cudaFree(Proxy_Host->raw.VertexPosition_Device);
    cudaFree(Proxy_Host->raw.VertexNormal_Device);
    cudaFree(Proxy_Host->raw.VertexColor_Device);
    cudaFree(Proxy_Host->raw.VertexTexCoord_Device);
    cudaFree(Proxy_Host->raw.TriangleToGridIndices_Device);
    cudaFree(Proxy_Host->raw.TriangleIndices_Device);
    cudaFree(Proxy_Host->raw.GridIndicesStart_Device);
    cudaFree(Proxy_Host->raw.GridIndicesEnd_Device);
    delete Proxy_Host;
    Proxy_Host = nullptr;
}

StaticMeshManager* StaticMeshManager::Get()
{
    static StaticMeshManager* GInstance = new StaticMeshManager();
    return GInstance;
}

StaticMesh* StaticMeshManager::LoadObj(std::string MeshName, std::string FilePath)
{
    StaticMesh* Mesh = CreateAndGetMesh(MeshName);
    LoadObjImpl(*Mesh, FilePath);
    return Mesh;
}

StaticMesh* StaticMeshManager::LoadGLTF(std::string MeshName, const tinygltf::Mesh& gltfMesh, const tinygltf::Model& model)
{
    StaticMesh* Mesh = CreateAndGetMesh(MeshName);
    LoadGLTFImpl(*Mesh, gltfMesh, model);
    return Mesh;
}

StaticMesh* StaticMeshManager::CreateAndGetMesh(std::string MeshName)
{
    if (Meshes.find(MeshName) == Meshes.end())
    {
		Meshes[MeshName] = std::make_unique<StaticMesh>();
	}
    return Meshes[MeshName].get();
}

StaticMesh* StaticMeshManager::GetMesh(std::string MeshName)
{
    if (Meshes.find(MeshName) == Meshes.end())
    {
        std::cerr << "Error: Mesh " << MeshName << " not found!" << std::endl;
		exit(1);
    }
    return Meshes[MeshName].get();
}

void StaticMeshManager::CreateRenderProxyForAll()
{
	// Create Proxy for all meshes
	for (auto& NameMesh : Meshes)
	{
		auto* Mesh = NameMesh.second.get();
        if (Mesh->Proxy_Host==nullptr)
        {
            Mesh->CreateProxy();
        }
	}
}
