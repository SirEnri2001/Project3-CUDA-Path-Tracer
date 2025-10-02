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

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        // TODO: handle materials loading differently
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
            mesh = p["MESH"];
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
}

void ReadGLTF(Scene* InOutScene, std::string filename)
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

    // 假设我们处理默认场景
    const tinygltf::Scene& scene = model.scenes[model.defaultScene > -1 ? model.defaultScene : 0];

    for (size_t i = 0; i < model.textures.size(); i++)
    {
        const tinygltf::Texture& baseColorTex = model.textures[i];
        const tinygltf::Image& image = model.images[baseColorTex.source];
        TextureManager::Get()->PreloadTexture(image.name, image.width, image.height, image.image);
        TextureManager::Get()->RegisterTextureName(baseColorTex.name, image.name);
    }

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
        InOutScene->materials.push_back(mat);
    }

    for (int i = 0; i < model.meshes.size(); i++)
    {
        const tinygltf::Mesh& gltfMesh = model.meshes[i];
        for (size_t j = 0; j < gltfMesh.primitives.size(); ++j) {
            const tinygltf::Primitive& primitive = gltfMesh.primitives[j];
            if (primitive.mode != TINYGLTF_MODE_TRIANGLES) {
                std::cout << "Only triangle mode is supported!" << std::endl;
                continue;
            }
            StaticMesh& Mesh = *StaticMeshManager::Get()->CreateAndGetMesh(gltfMesh.name);
            // 读取顶点属性
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

                    Mesh.Data.VertexNormal_Host[i] = {
                        normalData[indices[tid + indId] * normStride],
                        normalData[indices[tid + indId] * normStride + 1],
                        normalData[indices[tid + indId] * normStride + 2]
                    };

                    Mesh.Data.VertexTexCoord_Host[i] = {
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

    for (size_t i = 0; i < scene.nodes.size(); ++i) {
        const tinygltf::Node& node = model.nodes[scene.nodes[i]];

        glm::mat4 mat = glm::mat4(1.0f);
        for (int d = 0; d < 16; d++)
        {
            mat[d / 4][d % 4] = node.matrix[d];
        }
        Geom geom;
        geom.type = GeomType::MESH;
        geom.transform = mat;
        geom.inverseTransform = glm::inverse(mat);
        geom.invTranspose = glm::transpose(geom.inverseTransform);
        // 处理节点变换（矩阵，或 TRS 信息）
        // 如果节点包含网格
        if (node.mesh >= 0) {
			geom.Mesh_Host = StaticMeshManager::Get()->GetMesh(model.meshes[node.mesh].name);
        }
    }
}