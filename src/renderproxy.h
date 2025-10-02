#pragma once
#include <cuda_runtime_api.h>
#include <glm/glm.hpp>

#include "material.h"
__device__ glm::vec3 GetColorDevice(const Texture::RenderProxy& InTexture, glm::vec2 uv);