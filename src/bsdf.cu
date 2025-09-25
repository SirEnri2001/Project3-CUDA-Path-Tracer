#include "bsdf.h"
#include "sceneStructs.h"
#include "common.h"
#include <thrust/random.h>
// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
//__global__ void shadeFakeMaterial(
//    int iter,
//    int num_paths,
//    ShadeableIntersection* shadeableIntersections,
//    PathSegment* pathSegments,
//    Material* materials)
//{
//    int idx = blockIdx.x * blockDim.x + threadIdx.x;
//    if (idx < num_paths)
//    {
//        ShadeableIntersection intersection = shadeableIntersections[idx];
//        if (intersection.t > 0.0f) // if the intersection exists...
//        {
//            // Set up the RNG
//            // LOOK: this is how you use thrust's RNG! Please look at
//            // makeSeededRandomEngine as well.
//            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
//            thrust::uniform_real_distribution<float> u01(0, 1);
//
//            Material material = materials[intersection.materialId];
//            glm::vec3 materialColor = material.color;
//
//            // If the material indicates that the object was a light, "light" the ray
//            if (material.emittance > 0.0f) {
//                pathSegments[idx].color *= (materialColor * material.emittance);
//            }
//            // Otherwise, do some pseudo-lighting computation. This is actually more
//            // like what you would expect from shading in a rasterizer like OpenGL.
//            // TODO: replace this! you should be able to start with basically a one-liner
//            else {
//                float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(1.0f, 0.0f, 0.0f));
//                pathSegments[idx].color *= (materialColor * lightTerm) *0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
//                //pathSegments[idx].color *= u01(rng); // apply some noise because why not
//            }
//            // If there was no intersection, color the ray black.
//            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
//            // used for opacity, in which case they can indicate "no opacity".
//            // This can be useful for post-processing and image compositing.
//        }
//        else {
//            pathSegments[idx].color = glm::vec3(0.0f);
//        }
//    }
//}