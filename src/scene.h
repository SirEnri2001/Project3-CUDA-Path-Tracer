#pragma once

#include "sceneStructs.h"
#include <vector>

#include "pathtrace.h"

struct Material;

class Scene
{
    glm::vec3 boxMin;
    glm::vec3 boxMax;
public:
    Scene(PathTraceInfo Info);
    Scene() = delete;
    ~Scene();
    void ReadJSON(const std::string& jsonName);
    void ReadGLTF(std::string filename);
    void PostLoad();
    void CreateRenderProxyForAll();
    void DestroySceneRenderProxy();
    void CenterCamera();
    void CreateDefaultLight();
    void CreateDefaultFloor();
    std::vector<Geom> geoms;
    std::vector<int> lights;
    std::vector<Material> materials;
    RenderState state;

    struct RenderProxy
    {
		Geom* geoms_Device = nullptr;
		int geoms_size = 0;
		int* light_index_Device = nullptr;
		int lights_size = 0;
		Material* materials_Device = nullptr;
		int materials_size = 0;
    };

	RenderProxy* Proxy_Host = nullptr;
    RenderProxy* Proxy_Device = nullptr;
};
