#pragma once

#include <vector_types.h>

#include "utilities.h"

class Scene;
struct ShadeableIntersection;
struct PathSegment;

struct PathTraceInfo
{
	int x;
	int y;
	int frames;
	int depths;
	int ShowGUI; // 0: commandlet, 1: Show window but no imgui, 2: Show window with imgui
};

struct PathTraceRenderResource
{
	glm::vec3* dev_image = nullptr;
	PathSegment* dev_paths = nullptr;
	ShadeableIntersection* dev_path_intersections = nullptr;
	int* dev_geom_ids = nullptr;
	int* device_pathAlive = nullptr;
	uchar4* pbo = nullptr;
};

class PTEngine
{
	static PTEngine* GInstance;
public:
	PathTraceInfo Info;
	PathTraceRenderResource RenderResource;
	GuiDataContainer GuiData;
	PTEngine(PathTraceInfo InitInfo);
	PTEngine() = delete;
	void Init();
	void ClearBuffers();
	void Tick(Scene* scene);
	void Destroy();
	static PTEngine* Get();
};

void pathtrace(PTEngine* Engine, Scene* scene);