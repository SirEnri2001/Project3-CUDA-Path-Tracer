#pragma once

#include <string>
#include <glm/glm.hpp>
#include <unordered_map>

struct Texture
{
    glm::ivec3 Extent;
	glm::vec3* Image_Host;
    glm::vec3* Image_Device;
    struct RenderProxy
    {
        glm::ivec3 Extent;
        glm::vec3* Image_Device;
    };
	RenderProxy* Proxy_Host = nullptr; // Host readable address
	RenderProxy* Proxy_Device = nullptr; // Device readable address
};
struct Material
{
    glm::vec3 color = glm::vec3(1.f,0.f,1.f);
	Texture* BaseColorTexture = nullptr;
	Texture::RenderProxy* BaseColorTextureProxy_Device = nullptr;
    float roughness = 0.f;
    float metallicness = 0.f;
    Texture* MetallicnessTexture = nullptr;
    Texture::RenderProxy* MetallicnessTextureProxy_Device = nullptr;
    float emittance = 0.f;
    Texture* EmittanceTexture = nullptr;
    Texture::RenderProxy* EmittanceTextureProxy_Device = nullptr;
    bool isLight = false;
    bool isTransmissive = false;
};

class TextureManager
{
    std::unordered_map<std::string, Texture> Textures;
    std::unordered_map<std::string, std::string> TextureNameToImageName;
public:
    void LoadAllTexturesToDevice();
    Texture* PreloadTexture(std::string Filename);
    Texture* PreloadTexture(std::string ImageName, size_t width, size_t height, const std::vector<unsigned char>& Data);
    Texture* GetByTextureName(std::string TexName);
    void RegisterTextureName(std::string TexName, std::string ImageName);
    TextureManager();
    static TextureManager* Get();

    ~TextureManager();
};