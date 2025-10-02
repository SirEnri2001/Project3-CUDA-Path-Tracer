#include "material.h"

#include <cuda_runtime_api.h>
#include <stb_image.h>
std::vector<Material> Materials;
void CreateTextureFromFile(Texture& InOutTexture, std::string Filename)
{
    // Load an image file with 4 desired channels (RGBA)
    int width, height, channels_in_file;
    unsigned char* image = stbi_load(
        Filename.c_str(),      // filename
        &width,           // output for width
        &height,          // output for height
        &channels_in_file, // output for original channels in file
        4                 // desired number of channels
    );

    // Use the image data...
	InOutTexture.Extent = glm::ivec3(width, height, 1);
	InOutTexture.Image_Host = new glm::vec3[width * height];
	for (int y = 0; y < height; ++y)
	{
		for (int x = 0; x<width;++x)
		{
			InOutTexture.Image_Host[x+y*width] = glm::vec3(
                image[(x + y * width) * 4 + 0] / 255.0f,
                image[(x + y * width) * 4 + 1] / 255.0f,
                image[(x + y * width) * 4 + 2] / 255.0f
			);
		}
	}

    // Don't forget to free the memory when done
    stbi_image_free(image);
	InOutTexture.Image_Device = nullptr; // Will be allocated and copied later
}

void FreeTexture(Texture& InOutTexture)
{
    if (InOutTexture.Image_Device)
    {
        cudaFree(InOutTexture.Image_Device);
    }
    if (InOutTexture.Image_Host)
    {
        delete[] InOutTexture.Image_Host;
    }
}

TextureManager* TextureManager::Get()
{
	static TextureManager* Instance = new TextureManager();
	return Instance;
}

TextureManager::TextureManager()
{
}

TextureManager::~TextureManager()
{
    for (auto& tex : Textures)
        FreeTexture(tex.second);
    Textures.clear();
}

void TextureManager::LoadAllTexturesToDevice()
{
    for (auto& name_tex : Textures)
    {
		auto& tex = name_tex.second;
        if (tex.Image_Device!=nullptr)
        {
            continue;
        }
        cudaMalloc((void**)&tex.Image_Device, sizeof(glm::vec3) * tex.Extent.x * tex.Extent.y);
        cudaMemcpy(tex.Image_Device, tex.Image_Host,
            sizeof(glm::vec3) * tex.Extent.x * tex.Extent.y, cudaMemcpyHostToDevice);
        tex.Proxy_Host = new Texture::RenderProxy();
		tex.Proxy_Host->Extent = tex.Extent;
		tex.Proxy_Host->Image_Device = tex.Image_Device;
        cudaMalloc((void**)&tex.Proxy_Device, sizeof(Texture::RenderProxy));
        cudaMemcpy(tex.Proxy_Device, tex.Proxy_Host, sizeof(Texture::RenderProxy), cudaMemcpyHostToDevice);
    }
}

Texture* TextureManager::PreloadTexture(std::string Filename)
{
    if (Textures.find(Filename) != Textures.end())
    {
        return &Textures[Filename];
	}
	Textures[Filename] = Texture();
    CreateTextureFromFile(Textures[Filename], Filename);
	return &Textures[Filename];
}

Texture* TextureManager::PreloadTexture(std::string name, size_t width, size_t height, const std::vector<unsigned char>& Data)
{
    if (Textures.find(name) != Textures.end())
    {
        return &Textures[name];
    }
    Textures[name] = Texture();
	Texture& tex = Textures[name];
    // Use the image data...
    tex.Extent = glm::ivec3(width, height, 1);
    tex.Image_Host = new glm::vec3[width * height];
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            tex.Image_Host[x + y * width] = glm::vec3(
                Data[(x + y * width) * 4 + 0] / 255.0f,
                Data[(x + y * width) * 4 + 1] / 255.0f,
                Data[(x + y * width) * 4 + 2] / 255.0f
            );
        }
    }
	tex.Image_Device = nullptr; // Will be allocated and copied later
    tex.Proxy_Host = nullptr;
    tex.Proxy_Device = nullptr;
	return &Textures[name];
}

Texture* TextureManager::GetByTextureName(std::string TexName)
{
    return &Textures[TextureNameToImageName[TexName]];
}

void TextureManager::RegisterTextureName(std::string TexName, std::string ImageFilename)
{
    TextureNameToImageName[TexName] = ImageFilename;
}