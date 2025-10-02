#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

#include "material.h"

#define TINYGLTF_IMPLEMENTATION

#include <stb_image.h>
#include <stb_image_write.h>
// #define TINYGLTF_NOEXCEPTION // optional. disable exception handling.
#include "MeshLoader/tiny_gltf.h"

using namespace std;
using json = nlohmann::json;

Scene::Scene()
{
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = 800;
    camera.resolution.y = 800;
    float fovy = 45.f;
    state.iterations = 5000;
    state.traceDepth = 8;
    state.imageName = "PT_Result";

    camera.position = glm::vec3(0.f, 5.0f, 10.5f);
    camera.lookAt = glm::vec3(0.f, 5.0f, 0.f);
    camera.up = glm::vec3(0.f, 1.0f, 0.f);

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = yscaled * camera.resolution.x / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}

void Scene::CenterCamera()
{
	if (geoms.size()==0)
	{
        return;
	}
    glm::vec3 boxMin(FLT_MAX);
    glm::vec3 boxMax(-FLT_MAX);
    for (const auto& g : geoms)
    {
        if (g.type == MESH && g.Mesh_Host)
        {
            boxMin = glm::min(boxMin, glm::vec3(g.transform * glm::vec4(g.Mesh_Host->Data.boxMin, 1.0f)));
            boxMax = glm::max(boxMax, glm::vec3(g.transform * glm::vec4(g.Mesh_Host->Data.boxMax, 1.0f)));
        }
        else
        {
            //other primitive types
            //assume they are unit size centered at origin before transform
            boxMin = glm::min(boxMin, glm::vec3(-0.5f));
            boxMax = glm::max(boxMax, glm::vec3(0.5f));
        }
    }
    glm::vec3 center = (boxMin + boxMax) * 0.5f;
    Camera& camera = state.camera;
    camera.resetLookAt = center;
    camera.radius = glm::length(boxMax - boxMin) * 0.5f;
    camera.lookAt = center;
    camera.view = glm::vec3(0.f,-1.f,0.f);
	camera.position = -camera.view * 3.0f * camera.radius + camera.lookAt; // 3.0 is default zoom factor
    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
	camera.up = glm::normalize(glm::cross(camera.right, camera.view));
}


void Scene::DestroySceneRenderProxy()
{
    if (Proxy_Host)
    {
        cudaFree(Proxy_Device);
        cudaFree(Proxy_Host->geoms_Device);
        cudaFree(Proxy_Host->materials_Device);
        delete Proxy_Host;
        Proxy_Host = nullptr;
    }
}


Scene::~Scene()
{
    DestroySceneRenderProxy();
}


void Scene::ReadJSON(const std::string& jsonName)
{
    cout << "Reading scene from " << jsonName << " ..." << endl;
    cout << " " << endl;
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        if (p["TYPE"] == "Diffuse")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
			newMaterial.isLight = true;
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.roughness = p["ROUGHNESS"];
        }
        else if (p["TYPE"] == "DefaultLit")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.roughness = p["ROUGHNESS"];
            if (p.contains("TEX"))
            {
				const auto& col = p["TEX"]["BASECOLOR"];
                newMaterial.BaseColorTexture = TextureManager::Get()->PreloadTexture("../texture/" + col.get<std::string>());
            }
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom;
        if (type == "cube")
        {
            newGeom.type = CUBE;
        }else if (type=="plane")
        {
            newGeom.type = PLANE;
        }
        else if (type=="sphere")
        {
            newGeom.type = SPHERE;
        }else if (type=="mesh")
        {
            newGeom.type = MESH;
            std::string meshName = p["MESH"];
            newGeom.Mesh_Host = StaticMeshManager::Get()->LoadObj(meshName, std::string("../models/") + meshName + ".obj");
        }else
        {
            newGeom.type = CUBE;
        }
        newGeom.materialid = MatNameToID[p["MATERIAL"]];
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geoms.push_back(newGeom);
    }
    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
    int index = 0;
    for (auto& g : geoms)
    {
	    if (materials[g.materialid].isLight)
	    {
            lights.push_back(index);
	    }
        index++;
    }
}

void LoadGeomFromModelNodes(
    std::vector<Geom>& geoms, 
    const std::vector<tinygltf::Node>& nodes,
    const std::vector<tinygltf::Mesh>& meshes
)
{
	struct SceneNode
    {
        glm::mat4 LocalTransfrom;
		glm::mat4 GlobalTransform;
        SceneNode* parent = nullptr;
        std::string MeshName;
	};

    std::vector<SceneNode> sceneNodes;
    sceneNodes.resize(nodes.size());
    for (int i = 0; i < nodes.size();i++)
    {
        const tinygltf::Node& n = nodes[i];
		SceneNode& sn = sceneNodes[i];
        if (n.mesh!=-1)
        {
			sn.MeshName = meshes[n.mesh].name;
        }
		sn.LocalTransfrom = glm::mat4(1.0f);
        if (n.matrix.size()==16)
        {
            for (int d = 0; d < 16; d++)
            {
                sn.LocalTransfrom[d / 4][d % 4] = n.matrix[d];
            }
        }
        else
        {
            glm::vec3 translation(0.0f);
            if (n.translation.size() == 3)
            {
                translation = glm::vec3(n.translation[0], n.translation[1], n.translation[2]);
            }
            glm::vec3 rotation(0.0f);
            if (n.rotation.size() == 4)
            {
                // Convert quaternion to Euler angles
                glm::quat q(n.rotation[3], n.rotation[0], n.rotation[1], n.rotation[2]);
                rotation = glm::eulerAngles(q);
                rotation = glm::degrees(rotation);
            }
            glm::vec3 scale(1.0f);
            if (n.scale.size() == 3)
            {
                scale = glm::vec3(n.scale[0], n.scale[1], n.scale[2]);
            }
			sn.LocalTransfrom = utilityCore::buildTransformationMatrix(translation, rotation, scale);
        }

        for (auto& child : n.children)
        {
			sceneNodes[child].parent = &sn;
        }
    }

    for (int i = 0; i < sceneNodes.size(); i++)
	{
		// Calculate GlobalTransform
		SceneNode* current = &sceneNodes[i];
        SceneNode* sn = current;
        current->GlobalTransform = glm::mat4(1.0f);
		while (current)
		{
			sn->GlobalTransform = current->LocalTransfrom * sn->GlobalTransform;
			current = current->parent;
		}
    }

    for (int i = 0; i < sceneNodes.size(); i++)
    {
        SceneNode& sn = sceneNodes[i];
        if (sn.MeshName.size() > 0)
        {
            Geom geom;
            geom.type = GeomType::MESH;
            geom.transform = sn.GlobalTransform;
            geom.inverseTransform = glm::inverse(geom.transform);
            geom.invTranspose = glm::transpose(geom.inverseTransform);
            geom.Mesh_Host = StaticMeshManager::Get()->GetMesh(sn.MeshName);
            geom.materialid = geom.Mesh_Host->MaterialId;
            geoms.push_back(geom);
        }
    }
}

void Scene::ReadGLTF(std::string filename)
{
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;

    bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, filename);
    //bool ret = loader.LoadBinaryFromFile(&model, &err, &warn, argv[1]); // for binary glTF(.glb)

    if (!warn.empty()) {
        printf("Warn: %s\n", warn.c_str());
    }

    if (!err.empty()) {
        printf("Err: %s\n", err.c_str());
    }

    if (!ret) {
        printf("Failed to parse glTF\n");
        exit(1);
    }

    const tinygltf::Scene& scene = model.scenes[model.defaultScene > -1 ? model.defaultScene : 0];

	// Load Textures
    for (size_t i = 0; i < model.textures.size(); i++)
    {
        const tinygltf::Texture& baseColorTex = model.textures[i];
        const tinygltf::Image& image = model.images[baseColorTex.source];
        TextureManager::Get()->PreloadTexture(image.name, image.width, image.height, image.image);
        TextureManager::Get()->RegisterTextureName(baseColorTex.name, image.name);
    }

	// Load Materials
    for (size_t i = 0; i < model.materials.size(); i++)
    {
        const tinygltf::Material& material = model.materials[i];
        const tinygltf::PbrMetallicRoughness& pbr = material.pbrMetallicRoughness;

        std::vector<double> baseColorFactor = pbr.baseColorFactor;

        double metallicFactor = pbr.metallicFactor;
        double roughnessFactor = pbr.roughnessFactor;
        Material mat;
        mat.color = glm::vec3(baseColorFactor[0], baseColorFactor[1], baseColorFactor[2]);
        mat.roughness = roughnessFactor;

        if (pbr.baseColorTexture.index >= 0) {
            mat.BaseColorTexture = TextureManager::Get()->GetByTextureName(model.textures[pbr.baseColorTexture.index].name);
        }

        if (material.normalTexture.index >= 0) {

        }
        if (material.emissiveTexture.index >= 0) {
            mat.isLight = true;
            mat.emittance = 1.0f;
		}
        materials.push_back(mat);
    }

	// Load Meshes
    for (int i = 0; i < model.meshes.size(); i++)
    {
        const tinygltf::Mesh& gltfMesh = model.meshes[i];
        StaticMeshManager::Get()->LoadGLTF(gltfMesh.name, gltfMesh, model);
    }
    LoadGeomFromModelNodes(geoms, model.nodes, model.meshes);
    int index = 0;
    for (auto& g : geoms)
    {
        if (g.materialid >=0 && materials[g.materialid].isLight)
        {
            lights.push_back(index);
        }
        index++;
    }
}

void Scene::CreateRenderProxyForAll()
{
	StaticMeshManager::Get()->CreateRenderProxyForAll();
	StaticMeshManager::Get()->CalculateOctreeStructureCUDA();
    TextureManager::Get()->LoadAllTexturesToDevice();

	// Link Material texture pointers to device proxies
    for (auto& Mat : materials)
    {
        Mat.BaseColorTextureProxy_Device = nullptr;
        if (Mat.BaseColorTexture)
        {
            Mat.BaseColorTextureProxy_Device = Mat.BaseColorTexture->Proxy_Device;
        }
    }

	// Link Geom mesh pointers to device proxies
    for (auto& g : geoms)
    {
        g.MeshProxy_Device = nullptr;
        if (g.Mesh_Host)
        {
			g.MeshProxy_Device = g.Mesh_Host->Proxy_Device;
        }
    }

    if (Proxy_Host)
    {
        DestroySceneRenderProxy();
    }

	Proxy_Host = new RenderProxy();
	Proxy_Host->geoms_size = geoms.size();
	Proxy_Host->lights_size = lights.size();
	Proxy_Host->materials_size = materials.size();

    cudaMalloc((void**)&Proxy_Host->geoms_Device, sizeof(Geom) * geoms.size());
	cudaMemcpy(Proxy_Host->geoms_Device, geoms.data(), sizeof(Geom) * geoms.size(), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&Proxy_Host->materials_Device, sizeof(Material) * materials.size());
    cudaMemcpy(Proxy_Host->materials_Device, materials.data(), sizeof(Material) * materials.size(), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&Proxy_Host->light_index_Device, sizeof(int) * lights.size());
	cudaMemcpy(Proxy_Host->light_index_Device, lights.data(), sizeof(int) * lights.size(), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&Proxy_Device, sizeof(Scene::RenderProxy));
    cudaMemcpy(Proxy_Device, Proxy_Host, sizeof(Scene::RenderProxy), cudaMemcpyHostToDevice);
}
