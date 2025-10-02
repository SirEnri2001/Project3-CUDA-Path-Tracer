#pragma once
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <glm/glm.hpp>

#define GRID_SIZE 4681
#define GRID_WIDTH 16
#define GRID_LAYERS 4

namespace tinygltf
{
	class Model;
	struct Mesh;
}

enum StaticMeshAttrib
{
	POSITION = 1 << 0,
	NORMAL = 1 << 1,
	TEXCOORD = 1 << 2,
	COLOR = 1 << 3
};

class StaticMesh
{
public:
	struct RenderProxy
	{
		unsigned int VertexCount;
		glm::vec3 boxMin;
		glm::vec3 boxMax;
		unsigned int AttribFlags;
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

	unsigned int AttribFlags = 0;
	int MaterialId;
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
	StaticMesh* LoadGLTF(std::string MeshName, const tinygltf::Mesh& gltfMesh, const tinygltf::Model& model);
	StaticMesh* GetMesh(std::string MeshName);
	StaticMesh* CreateAndGetMesh(std::string MeshName);
	void CreateRenderProxyForAll();
	void CalculateOctreeStructureCUDA();
};
