#pragma once

#include <cuda_runtime.h>

#include "glm/glm.hpp"

#include <string>
#include <vector>

#include "mesh.h"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType
{
    SPHERE,
    CUBE,
    PLANE,
    MESH
};

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Geom
{
    enum GeomType type;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;

	StaticMesh* Mesh_Host = nullptr;
	StaticMesh::RenderProxy* MeshProxy_Device = nullptr;
};

struct Camera
{
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
    float radius = 3.f;
    glm::vec3 resetLookAt = glm::vec3(0.f);
    void Reset()
    {
        lookAt = resetLookAt;
        position = resetLookAt + glm::vec3(0, 0, radius * 2.5f);
        view = glm::normalize(lookAt - position);
        right = glm::normalize(glm::cross(view, up));
        up = glm::cross(right, view);
    }
    void SetCamera(float phi, float theta, float zoom, float dX, float dY)
    {
        view.x = -sin(phi) * sin(theta);
        view.y = -cos(theta);
        view.z = -cos(phi) * sin(theta);

        view = glm::normalize(view);
        glm::vec3 v = view;
        glm::vec3 u = glm::vec3(0, 1, 0);//glm::normalize(cam.up);
        glm::vec3 r = glm::cross(v, u);
        up = glm::normalize(glm::cross(r, v));
        right = glm::normalize(r);
        position = -view * zoom * radius + lookAt;

        lookAt -= dX * right * radius * pixelLength.x * zoom * 0.5f;
        lookAt += dY * up * radius * pixelLength.y * zoom * 0.5f;
    }
};

struct RenderState
{
    Camera camera;
    unsigned int frames;
    int depths;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment
{
    Ray ray;
    glm::vec3 Contribution;
    glm::vec3 BSDF;
    glm::vec3 debug;
    float PDF;
    float Cosine;
    int pixelIndex;
    int remainingBounces;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
    glm::vec3 intersectPoint;
    glm::vec3 surfaceNormal;
	glm::vec2 uv;
    bool outside;
    int materialId;
    float t_min;
};
