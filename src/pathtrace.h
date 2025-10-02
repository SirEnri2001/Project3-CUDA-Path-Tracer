#pragma once

#include "scene.h"
#include "utilities.h"

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceCreate(Scene* scene);
void pathtraceNewFrame(Scene* scene);
void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(Scene* scene, uchar4 *pbo, int frame, int iteration);
