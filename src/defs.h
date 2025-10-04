#pragma once

// use for not showing glfw window, so NSight compute can run
#define COMMANDLET 1

// Enable BVH acceleration
#define USE_MESH_GRID_ACCELERATION 1

// BVH and Octree structure defs. Grid size should be 1+2^3+4^3+8^3+...
#define GRID_SIZE 73
#define GRID_WIDTH 4
#define GRID_LAYERS 3

// Split triangle into bounds so that no big root bound
#define USE_OPTIMIZED_GRID 1

// Whether use presort by object's bvh so that minimum warp divergent
#define USE_SORT_BY_BOUNDING 1

// Use fallback uniform sampling instead of GGX importance sampling
#define USE_UNIFORM_SAMPLING 0

// Math constant defs
#define PI                3.1415926535897932384626422832795028841971f
#define INV_PI                0.31830988618379067153776752674503f
#define TWO_PI            6.2831853071795864769252867665590057683943f
#define SQRT_OF_ONE_THIRD 0.5773502691896257645091487805019574556476f
#define EPSILON           0.00001f