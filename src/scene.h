#pragma once

#include "sceneStructs.h"
#include <vector>

struct Material;

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;
    std::string mesh;
};
void ReadGLTF(Scene* InOutScene, std::string filename);
