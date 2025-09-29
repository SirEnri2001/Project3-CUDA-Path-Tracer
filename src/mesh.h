#pragma once
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <glm/glm.hpp>

#define GRID_SIZE 4681
#define GRID_WIDTH 16
#define GRID_LAYERS 4
// 2^4==16, 8^0+8^1+8^2+8^3+8^4=4681
//struct StaticMeshData_Host
//{
//	glm::vec3 boxMin = glm::vec3(FLT_MAX, FLT_MAX, FLT_MAX);
//	glm::vec3 boxMax = glm::vec3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
//	glm::vec3* VertexPosition_Host;
//	glm::vec3* VertexNormal_Host;
//	glm::vec3* VertexColor_Host;
//	glm::vec2* VertexTexCoord_Host;
//	unsigned int* Indices_Host;
//	unsigned int VertexCount;
//	StaticMeshData_Host() : VertexPosition_Host(nullptr), VertexNormal_Host(nullptr), VertexColor_Host(nullptr), VertexCount(0), VertexTexCoord_Host(nullptr), Indices_Host(nullptr) {}
//	~StaticMeshData_Host()
//	{
//		if (VertexPosition_Host)
//		{
//			delete[] VertexPosition_Host;
//			VertexPosition_Host = nullptr;
//		}
//		if (VertexNormal_Host)
//		{
//			delete[] VertexNormal_Host;
//			VertexNormal_Host = nullptr;
//		}
//		if (VertexColor_Host)
//		{
//			delete[] VertexColor_Host;
//			VertexColor_Host = nullptr;
//		}
//	}
//};
//
//
//class StaticMeshData_Device
//{
//public:
//	unsigned int VertexCount;
//	glm::vec3 boxMin;
//	glm::vec3 boxMax;
//	struct Raw
//	{
//		glm::vec3* VertexPosition_Device;
//		glm::vec3* VertexNormal_Device;
//		glm::vec3* VertexColor_Device;
//		glm::vec2* VertexTexCoord_Device;
//		int* TraingleToGridIndices_Device;
//		int* TriangleIndices_Device;
//		int* GridIndicesStart_Device;
//		int* GridIndicesEnd_Device;
//		unsigned int* Indices_Device;
//	} raw;
//
//	StaticMeshData_Device(unsigned int VCount, glm::vec3 InBoxMin, glm::vec3 InBoxMax);
//	~StaticMeshData_Device()
//	{
//	}
//};

//void ReadObjMesh(StaticMeshData_Host& OutData_Host, std::string	FilePath);
//void CreateDeviceObject(StaticMeshData_Device** OutData_Device, StaticMeshData_Device& DeviceData_OnHost, StaticMeshData_Host& InData_Host);

class StaticMesh
{
public:
	struct RenderProxy
	{
		unsigned int VertexCount;
		glm::vec3 boxMin;
		glm::vec3 boxMax;
		struct RawPtr
		{
			glm::vec3* VertexPosition_Device = nullptr;
			glm::vec3* VertexNormal_Device = nullptr;
			glm::vec3* VertexColor_Device = nullptr;
			glm::vec2* VertexTexCoord_Device = nullptr;
			int* TriangleToGridIndices_Device = nullptr;
			int* TriangleIndices_Device = nullptr;
			int* GridIndicesStart_Device = nullptr;
			int* GridIndicesEnd_Device = nullptr;
		} raw;
	};

	struct HostData
	{
		unsigned int VertexCount;
		glm::vec3 boxMin;
		glm::vec3 boxMax;
		std::vector<glm::vec3> VertexPosition_Host;
		std::vector<glm::vec3> VertexNormal_Host;
		std::vector<glm::vec3> VertexColor_Host;
		std::vector<glm::vec2> VertexTexCoord_Host;
	} Data;

	RenderProxy* Proxy_Host = nullptr; // Host readable address
	RenderProxy* Proxy_Device = nullptr; // Device readable address

	StaticMesh(unsigned int VCount, glm::vec3 InBoxMin, glm::vec3 InBoxMax);
	StaticMesh();
	~StaticMesh();
	void CreateProxy(); // create Proxy_Host and Proxy_Device
	void DestroyProxy(); // destroy Proxy_Host and Proxy_Device
};

class StaticMeshManager
{
	std::unordered_map<std::string, std::unique_ptr<StaticMesh>> Meshes;
public:
	static StaticMeshManager* Get();
	StaticMesh* LoadObj(std::string MeshName, std::string FilePath);
	StaticMesh* GetMesh(std::string MeshName);
};