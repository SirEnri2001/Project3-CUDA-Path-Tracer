#pragma once

#include <cuda_runtime_api.h>
#include <string>
#include <glm/glm.hpp>
#include <thrust/device_vector.h>

#define GRID_SIZE 4681
#define GRID_WIDTH 16
#define GRID_LAYERS 4
// 2^4==16, 8^0+8^1+8^2+8^3+8^4=4681
struct StaticMeshData_Host
{
	glm::vec3 boxMin = glm::vec3(FLT_MAX, FLT_MAX, FLT_MAX);
	glm::vec3 boxMax = glm::vec3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
	glm::vec3* VertexPosition_Host;
	glm::vec3* VertexNormal_Host;
	glm::vec3* VertexColor_Host;
	glm::vec2* VertexTexCoord_Host;
	unsigned int* Indices_Host;
	unsigned int VertexCount;
	StaticMeshData_Host() : VertexPosition_Host(nullptr), VertexNormal_Host(nullptr), VertexColor_Host(nullptr), VertexCount(0), VertexTexCoord_Host(nullptr), Indices_Host(nullptr) {}
	~StaticMeshData_Host()
	{
		if (VertexPosition_Host)
		{
			delete[] VertexPosition_Host;
			VertexPosition_Host = nullptr;
		}
		if (VertexNormal_Host)
		{
			delete[] VertexNormal_Host;
			VertexNormal_Host = nullptr;
		}
		if (VertexColor_Host)
		{
			delete[] VertexColor_Host;
			VertexColor_Host = nullptr;
		}
	}
};


class StaticMeshData_Device
{
public:
	unsigned int VertexCount;
	glm::vec3 boxMin;
	glm::vec3 boxMax;
	struct Raw
	{
		glm::vec3* VertexPosition_Device;
		glm::vec3* VertexNormal_Device;
		glm::vec3* VertexColor_Device;
		glm::vec2* VertexTexCoord_Device;
		int* TraingleToGridIndices_Device;
		int* TriangleIndices_Device;
		int* GridIndicesStart_Device;
		int* GridIndicesEnd_Device;
		unsigned int* Indices_Device;
	} raw;

	StaticMeshData_Device(unsigned int VCount, glm::vec3 InBoxMin, glm::vec3 InBoxMax);
	~StaticMeshData_Device()
	{
	}
};

void ReadObjMesh(StaticMeshData_Host& OutData_Host, std::string	FilePath);
void CreateDeviceObject(StaticMeshData_Device** OutData_Device, StaticMeshData_Device& DeviceData_OnHost, StaticMeshData_Host& InData_Host);