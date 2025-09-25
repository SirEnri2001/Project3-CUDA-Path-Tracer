#include "intersections.h"



//__host__ __device__ float triangleIntersectionTest(
//    Geom triangle,
//    Ray r,
//    glm::vec3 &intersectionPoint,
//    glm::vec3 &normal,
//    bool &outside)
//{
//    Ray q;
//    q.origin    =                multiplyMV(triangle.inverseTransform, glm::vec4(r.origin   , 1.0f));
//    q.direction = glm::normalize(multiplyMV(triangle.inverseTransform, glm::vec4(r.direction, 0.0f)));
//    glm::vec3 v0 = triangle.vertices[1] - triangle.vertices[0];
//    glm::vec3 v1 = triangle.vertices[2] - triangle.vertices[0];
//    glm::vec3 v2 = q.origin - triangle.vertices[0];
//    float d00 = glm::dot(v0, v0);
//    float d01 = glm::dot(v0, v1);
//    float d11 = glm::dot(v1, v1);
//    float d20 = glm::dot(v2, v0);
//    float d21 = glm::dot(v2, v1);
//    float denom = d00 * d11 - d01 * d01;
//    float a = (d11 * d20 - d01 * d21) / denom;
//    if (a < 0.f)
//        return -1;
//    float b = (d00 * d21 - d01 * d20) / denom;
//    if (b < 0.f || a + b > 1.f)
//        return -1;
//    glm::vec3 pvec = glm::cross(q.direction, v1);
//    float det = glm::dot(v0, pvec);
//    if (fabs(det) < 1e-8f)
//        return -1;
//    float invDet = 1.f / det;
//    glm::vec3 tvec = q.origin - triangle.vertices[0];
//    float u = glm::dot(tvec, pvec) * invDet;
//    if (u < 0.f || u > 1.f)
//        return -1;
//    glm::vec3 qvec = glm::cross(tvec, v0);
//    float v = glm::dot(q.direction, qvec) * invDet;
//    if (v < 0.f || u + v > 1.f)
//        return -1;
//    float t = glm::dot(v1, qvec);
//	t *= invDet;
//    if (t < 0)
//        return -1;
//    intersectionPoint = multiplyMV(triangle.transform, glm::vec4(getPointOnRay(q, t), 1.0f));
//    normal = glm::normalize(multiplyMV(triangle.invTranspose, glm::vec4(glm::cross(v0, v1), 0.0f)));
//    outside = true;
//	return glm::length(r.origin - intersectionPoint);
//}