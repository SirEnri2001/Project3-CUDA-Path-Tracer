#include "mesh.h"

#include <iostream>
#define TINYOBJLOADER_IMPLEMENTATION
#include "common.h"
#include "MeshLoader/tiny_obj_loader.h"

void ReadObjMesh(StaticMeshData_Host& OutData_Host, std::string	FilePath)
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
		OutData_Host.VertexCount = shapes[s].mesh.num_face_vertices.size() * 3; // assume triangulated
		OutData_Host.VertexPosition_Host = new glm::vec3[OutData_Host.VertexCount];
		OutData_Host.VertexNormal_Host = new glm::vec3[OutData_Host.VertexCount];
		OutData_Host.VertexColor_Host = new glm::vec3[OutData_Host.VertexCount];
        OutData_Host.VertexTexCoord_Host = new glm::vec2[OutData_Host.VertexCount];
		OutData_Host.Indices_Host = new unsigned int[OutData_Host.VertexCount];

        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);

            // Loop over vertices in the face.
            for (size_t v = 0; v < 3/*fv*/; v++) {
                // access to vertex
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                tinyobj::real_t vx = attrib.vertices[3 * size_t(idx.vertex_index) + 0];
                tinyobj::real_t vy = attrib.vertices[3 * size_t(idx.vertex_index) + 1];
                tinyobj::real_t vz = attrib.vertices[3 * size_t(idx.vertex_index) + 2];

				OutData_Host.VertexPosition_Host[3 * f + v] = glm::vec3(vx, vy, vz);
				OutData_Host.Indices_Host[3 * f + v] = 3 * f + v;

                // Check if `normal_index` is zero or positive. negative = no normal data
                if (idx.normal_index >= 0) {
                    tinyobj::real_t nx = attrib.normals[3 * size_t(idx.normal_index) + 0];
                    tinyobj::real_t ny = attrib.normals[3 * size_t(idx.normal_index) + 1];
                    tinyobj::real_t nz = attrib.normals[3 * size_t(idx.normal_index) + 2];
                    OutData_Host.VertexNormal_Host[3 * f + v] = glm::vec3(nx, ny, nz);
                }

                // Check if `texcoord_index` is zero or positive. negative = no texcoord data
                if (idx.texcoord_index >= 0) {
                    tinyobj::real_t tx = attrib.texcoords[2 * size_t(idx.texcoord_index) + 0];
                    tinyobj::real_t ty = attrib.texcoords[2 * size_t(idx.texcoord_index) + 1];
					OutData_Host.VertexTexCoord_Host[3 * f + v] = glm::vec2(tx, ty);
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


void CreateDeviceObject(StaticMeshData_Device** OutData_Device, StaticMeshData_Device& DeviceData_OnHost, StaticMeshData_Host& InData_Host)
{
    DeviceData_OnHost.VertexCount = InData_Host.VertexCount;
    cudaMalloc((void**)&DeviceData_OnHost.VertexPosition_Device, sizeof(glm::vec3) * DeviceData_OnHost.VertexCount);
    cudaMemcpy(DeviceData_OnHost.VertexPosition_Device, InData_Host.VertexPosition_Host, sizeof(glm::vec3) * DeviceData_OnHost.VertexCount, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&DeviceData_OnHost.VertexTexCoord_Device, sizeof(glm::vec3) * DeviceData_OnHost.VertexCount);
    cudaMemcpy(DeviceData_OnHost.VertexTexCoord_Device, InData_Host.VertexTexCoord_Host, sizeof(glm::vec2) * DeviceData_OnHost.VertexCount, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&DeviceData_OnHost.VertexNormal_Device, sizeof(glm::vec3) * DeviceData_OnHost.VertexCount);
    cudaMemcpy(DeviceData_OnHost.VertexNormal_Device, InData_Host.VertexNormal_Host, sizeof(glm::vec3) * DeviceData_OnHost.VertexCount, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&DeviceData_OnHost.VertexColor_Device, sizeof(glm::vec3) * DeviceData_OnHost.VertexCount);
    cudaMemcpy(DeviceData_OnHost.VertexColor_Device, InData_Host.VertexColor_Host, sizeof(glm::vec3) * DeviceData_OnHost.VertexCount, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&DeviceData_OnHost.Indices_Device, sizeof(unsigned int) * DeviceData_OnHost.VertexCount);
    cudaMemcpy(DeviceData_OnHost.Indices_Device, InData_Host.Indices_Host, sizeof(unsigned int) * DeviceData_OnHost.VertexCount, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&DeviceData_OnHost.VertexGridIndices_Device, sizeof(int) * DeviceData_OnHost.VertexCount);
    cudaMemset(DeviceData_OnHost.VertexGridIndices_Device, -1, sizeof(int) * DeviceData_OnHost.VertexCount);
    cudaMalloc((void**)&DeviceData_OnHost.GridIndicesStart_Device, sizeof(int) * GRID_SIZE);
    
	cudaMalloc((void**)&DeviceData_OnHost.GridIndicesEnd_Device, sizeof(int) * GRID_SIZE);

    cudaMalloc((void**)OutData_Device, sizeof(StaticMeshData_Device));
	cudaMemcpy(*OutData_Device, &DeviceData_OnHost, sizeof(StaticMeshData_Device), cudaMemcpyHostToDevice);
    checkCUDAError("CreateDeviceObject");
}