#include "mesh.h"

#include <iostream>
#define TINYOBJLOADER_IMPLEMENTATION
#include "common.h"
#include "MeshLoader/tiny_obj_loader.h"
// Define these only in *one* .cc file.
#define TINYGLTF_IMPLEMENTATION

#include <stb_image.h>
#include <stb_image_write.h>
// #define TINYGLTF_NOEXCEPTION // optional. disable exception handling.
#include "MeshLoader/tiny_gltf.h"


//void ReadObjMesh(StaticMeshData_Host& OutData_Host, std::string	FilePath)
//{
//    tinyobj::ObjReaderConfig reader_config;
//    reader_config.mtl_search_path = "./"; // Path to material files
//
//    tinyobj::ObjReader reader;
//
//    if (!reader.ParseFromFile(FilePath, reader_config)) {
//        if (!reader.Error().empty()) {
//            std::cerr << "TinyObjReader: " << reader.Error();
//        }
//        exit(1);
//    }
//
//    if (!reader.Warning().empty()) {
//        std::cout << "TinyObjReader: " << reader.Warning();
//    }
//
//    auto& attrib = reader.GetAttrib();
//    auto& shapes = reader.GetShapes();
//    auto& materials = reader.GetMaterials();
//
//    // Loop over shapes
//    for (size_t s = 0; s < 1/*shapes.size()*/; s++) {
//        // Loop over faces(polygon)
//        size_t index_offset = 0;
//
//        // allocate memory
//		OutData_Host.VertexCount = shapes[s].mesh.num_face_vertices.size() * 3; // assume triangulated
//		OutData_Host.VertexPosition_Host = new glm::vec3[OutData_Host.VertexCount];
//		OutData_Host.VertexNormal_Host = new glm::vec3[OutData_Host.VertexCount];
//		OutData_Host.VertexColor_Host = new glm::vec3[OutData_Host.VertexCount];
//        OutData_Host.VertexTexCoord_Host = new glm::vec2[OutData_Host.VertexCount];
//		OutData_Host.Indices_Host = new unsigned int[OutData_Host.VertexCount];
//
//        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
//            size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);
//
//            // Loop over vertices in the face.
//            for (size_t v = 0; v < 3/*fv*/; v++) {
//                // access to vertex
//                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
//                tinyobj::real_t vx = attrib.vertices[3 * size_t(idx.vertex_index) + 0];
//                tinyobj::real_t vy = attrib.vertices[3 * size_t(idx.vertex_index) + 1];
//                tinyobj::real_t vz = attrib.vertices[3 * size_t(idx.vertex_index) + 2];
//
//				OutData_Host.boxMin = glm::min(OutData_Host.boxMin, glm::vec3(vx, vy, vz));
//				OutData_Host.boxMax = glm::max(OutData_Host.boxMax, glm::vec3(vx, vy, vz));
//
//				OutData_Host.VertexPosition_Host[3 * f + v] = glm::vec3(vx, vy, vz);
//				OutData_Host.Indices_Host[3 * f + v] = 3 * f + v;
//
//                // Check if `normal_index` is zero or positive. negative = no normal data
//                if (idx.normal_index >= 0) {
//                    tinyobj::real_t nx = attrib.normals[3 * size_t(idx.normal_index) + 0];
//                    tinyobj::real_t ny = attrib.normals[3 * size_t(idx.normal_index) + 1];
//                    tinyobj::real_t nz = attrib.normals[3 * size_t(idx.normal_index) + 2];
//                    OutData_Host.VertexNormal_Host[3 * f + v] = glm::vec3(nx, ny, nz);
//                }
//
//                // Check if `texcoord_index` is zero or positive. negative = no texcoord data
//                if (idx.texcoord_index >= 0) {
//                    tinyobj::real_t tx = attrib.texcoords[2 * size_t(idx.texcoord_index) + 0];
//                    tinyobj::real_t ty = attrib.texcoords[2 * size_t(idx.texcoord_index) + 1];
//					OutData_Host.VertexTexCoord_Host[3 * f + v] = glm::vec2(tx, ty);
//                }
//                
//                // Optional: vertex colors
//                // tinyobj::real_t red   = attrib.colors[3*size_t(idx.vertex_index)+0];
//                // tinyobj::real_t green = attrib.colors[3*size_t(idx.vertex_index)+1];
//                // tinyobj::real_t blue  = attrib.colors[3*size_t(idx.vertex_index)+2];
//            }
//            index_offset += fv;
//
//            // per-face material
//            shapes[s].mesh.material_ids[f];
//        }
//    }
//}

//void ReadGLTFMesh(StaticMeshData_Host& OutData_Host, std::string FilePath)
//{
//    using namespace tinygltf;
//    Model model;
//    TinyGLTF loader;
//    std::string err;
//    std::string warn;
//
//    bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, FilePath);
//    //bool ret = loader.LoadBinaryFromFile(&model, &err, &warn, argv[1]); // for binary glTF(.glb)
//
//    if (!warn.empty()) {
//        printf("Warn: %s\n", warn.c_str());
//    }
//
//    if (!err.empty()) {
//        printf("Err: %s\n", err.c_str());
//    }
//
//    if (!ret) {
//        printf("Failed to parse glTF\n");
//        exit(1);
//    }
//
//
//}

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

                // Optional: vertex colors
                // tinyobj::real_t red   = attrib.colors[3*size_t(idx.vertex_index)+0];
                // tinyobj::real_t green = attrib.colors[3*size_t(idx.vertex_index)+1];
                // tinyobj::real_t blue  = attrib.colors[3*size_t(idx.vertex_index)+2];
            }
            index_offset += fv;

            // per-face material
            shapes[s].mesh.material_ids[f];
        }
    }
}

//void CreateDeviceObject(StaticMeshData_Device** OutData_Device, StaticMeshData_Device& DeviceData_OnHost, StaticMeshData_Host& InData_Host)
//{
//    checkCUDAError("before CreateDeviceObject");
//    DeviceData_OnHost.VertexCount = InData_Host.VertexCount;
//	DeviceData_OnHost.boxMin = InData_Host.boxMin;
//	DeviceData_OnHost.boxMax = InData_Host.boxMax;
//
//    cudaMalloc((void**)&DeviceData_OnHost.raw.VertexPosition_Device, sizeof(glm::vec3) * DeviceData_OnHost.VertexCount);
//    cudaMemcpy(DeviceData_OnHost.raw.VertexPosition_Device, InData_Host.VertexPosition_Host, sizeof(glm::vec3) * DeviceData_OnHost.VertexCount, cudaMemcpyHostToDevice);
//
//	cudaMalloc((void**)&DeviceData_OnHost.raw.VertexTexCoord_Device, sizeof(glm::vec3) * DeviceData_OnHost.VertexCount);
//    cudaMemcpy(DeviceData_OnHost.raw.VertexTexCoord_Device, InData_Host.VertexTexCoord_Host, sizeof(glm::vec2) * DeviceData_OnHost.VertexCount, cudaMemcpyHostToDevice);
//
//	cudaMalloc((void**)&DeviceData_OnHost.raw.VertexNormal_Device, sizeof(glm::vec3) * DeviceData_OnHost.VertexCount);
//    cudaMemcpy(DeviceData_OnHost.raw.VertexNormal_Device, InData_Host.VertexNormal_Host, sizeof(glm::vec3) * DeviceData_OnHost.VertexCount, cudaMemcpyHostToDevice);
//
//	cudaMalloc((void**)&DeviceData_OnHost.raw.VertexColor_Device, sizeof(glm::vec3) * DeviceData_OnHost.VertexCount);
//    cudaMemcpy(DeviceData_OnHost.raw.VertexColor_Device, InData_Host.VertexColor_Host, sizeof(glm::vec3) * DeviceData_OnHost.VertexCount, cudaMemcpyHostToDevice);
//
//	cudaMalloc((void**)&DeviceData_OnHost.raw.Indices_Device, sizeof(unsigned int) * DeviceData_OnHost.VertexCount);
//    cudaMemcpy(DeviceData_OnHost.raw.Indices_Device, InData_Host.Indices_Host, sizeof(unsigned int) * DeviceData_OnHost.VertexCount, cudaMemcpyHostToDevice);
//
//	cudaMalloc((void**)&DeviceData_OnHost.raw.TraingleToGridIndices_Device, sizeof(int) * DeviceData_OnHost.VertexCount);
//    cudaMemset(DeviceData_OnHost.raw.TraingleToGridIndices_Device, -1, sizeof(int) * DeviceData_OnHost.VertexCount / 3);
//
//	cudaMalloc((void**)&DeviceData_OnHost.raw.GridIndicesStart_Device, sizeof(int) * GRID_SIZE);
//    cudaMemset(DeviceData_OnHost.raw.GridIndicesStart_Device, -1, sizeof(int) * GRID_SIZE);
//    cudaMalloc((void**)&DeviceData_OnHost.raw.GridIndicesEnd_Device, sizeof(int) * GRID_SIZE);
//    cudaMemset(DeviceData_OnHost.raw.GridIndicesEnd_Device, -1, sizeof(int) * GRID_SIZE);
//
//    cudaMalloc((void**)&DeviceData_OnHost.raw.TriangleIndices_Device, sizeof(int) * DeviceData_OnHost.VertexCount / 3);
//	std::vector<int> InitTriangleIndices(DeviceData_OnHost.VertexCount / 3);
//    for (int i = 0; i < DeviceData_OnHost.VertexCount / 3;i++)
//    {
//        InitTriangleIndices[i] = i;
//    }
//	cudaMemcpy(DeviceData_OnHost.raw.TriangleIndices_Device, InitTriangleIndices.data(), sizeof(int) * DeviceData_OnHost.VertexCount / 3, cudaMemcpyHostToDevice);
//
//	cudaMalloc((void**)OutData_Device, sizeof(StaticMeshData_Device));
//    cudaMemcpy(*OutData_Device, &DeviceData_OnHost, sizeof(StaticMeshData_Device), cudaMemcpyHostToDevice);
//    checkCUDAError("CreateDeviceObject");
//}
//
//StaticMeshData_Device::StaticMeshData_Device(unsigned int VCount, glm::vec3 InBoxMin, glm::vec3 InBoxMax)
//    :VertexCount(VCount), boxMin(InBoxMin), boxMax(InBoxMax), raw()
//{
//}

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
    Meshes[MeshName] = std::make_unique<StaticMesh>();
    StaticMesh* Mesh = Meshes[MeshName].get();
    LoadObjImpl(*Mesh, FilePath);
    return Mesh;
}

StaticMesh* StaticMeshManager::GetMesh(std::string MeshName)
{
    return Meshes[MeshName].get();
}
