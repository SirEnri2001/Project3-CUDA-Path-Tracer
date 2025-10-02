#pragma once

#include <cuda_runtime_api.h>
#include <glm/glm.hpp>
struct ShadeableIntersection;
struct PathSegment;
struct Material;

struct BRDF_Params
{
    glm::vec3 baseColor = glm::vec3(.82f, .67f, .16f);
    float metallic = 0.f;
    float subsurface = 0.f;
    float specular = .5f;
    float roughness = .5f;
    float specularTint = 0.f;
    float anisotropic = 0.f;
    float sheen = 0.f;
    float sheenTint = .5f;
    float clearcoat = 0.f;
    float clearcoatGloss = 1.f;
};

__host__ __device__
glm::vec3 BRDF(BRDF_Params Params, glm::vec3 L, glm::vec3 V, glm::vec3 N, glm::vec3 X, glm::vec3 Y);