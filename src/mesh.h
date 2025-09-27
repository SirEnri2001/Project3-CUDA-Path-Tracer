#pragma once

#include <cuda_runtime_api.h>
#include <string>
#include <glm/glm.hpp>

#define GRID_SIZE 4096
#define GRID_WIDTH 16

struct StaticMeshData_Host
{
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

struct StaticMeshData_Device
{
	glm::vec3* VertexPosition_Device;
	glm::vec3* VertexNormal_Device;
	glm::vec3* VertexColor_Device;
	glm::vec2* VertexTexCoord_Device;
	int* VertexGridIndices_Device;
	int* GridIndicesStart_Device;
	int* GridIndicesEnd_Device;
	unsigned int* Indices_Device;
	unsigned int VertexCount;
	StaticMeshData_Device() : VertexPosition_Device(nullptr), VertexNormal_Device(nullptr), VertexColor_Device(nullptr), VertexCount(0), Indices_Device(nullptr), VertexTexCoord_Device(nullptr) {}
	~StaticMeshData_Device()
	{
		if (VertexPosition_Device)
		{
			cudaFree(VertexPosition_Device);
			VertexPosition_Device = nullptr;
		}
		if (VertexNormal_Device)
		{
			cudaFree(VertexNormal_Device);
			VertexNormal_Device = nullptr;
		}
		if (VertexColor_Device)
		{
			cudaFree(VertexColor_Device);
			VertexColor_Device = nullptr;
		}
		if (Indices_Device)
		{
			cudaFree(Indices_Device);
			Indices_Device = nullptr;
		}
	}
};

void ReadObjMesh(StaticMeshData_Host& OutData_Host, std::string	FilePath);
void CreateDeviceObject(StaticMeshData_Device** OutData_Device, StaticMeshData_Device& DeviceData_OnHost, StaticMeshData_Host& InData_Host);