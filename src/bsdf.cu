#include "bsdf.h"
#include "sceneStructs.h"
#include "common.h"
#include <thrust/random.h>

#include "utilities.h"

using namespace glm;

// code below is from https://github.com/wdas/brdf/blob/main/src/brdfs/disney.brdf

__host__ __device__
vec3 SchlickFresnel(float thetaO, vec3 R, float rough)
{
    float m = clamp(1.f - thetaO, 0.f, 1.f);
    float m2 = m * m;
    return R + (1.f - R)*m2 * m2 * m; // pow(m,5)
}

__host__ __device__
float sqr(float x) { return x * x; }

__host__ __device__
float GTR1(float NdotH, float a)
{
    if (a >= 1) return 1 / PI;
    float a2 = a * a;
    float t = 1 + (a2 - 1) * NdotH * NdotH;
    return (a2 - 1) / (PI * log(a2) * t);
}

__host__ __device__
float GTR2(float NdotH, float a)
{
    float a2 = a * a;
    float t = 1.f + (a2 - 1.f) * NdotH * NdotH;
    return a2 / (PI * t * t);
}

__host__ __device__
float GTR2_aniso(float NdotH, float HdotX, float HdotY, float ax, float ay)
{
    return 1 / (PI * ax * ay * sqr(sqr(HdotX / ax) + sqr(HdotY / ay) + NdotH * NdotH));
}

__host__ __device__
float smithG_GGX(float NdotV, float alphaG)
{
    float a = alphaG * alphaG;
    float b = NdotV * NdotV;
    return 1 / (NdotV + sqrt(a + b - a * b));
}

__host__ __device__
float schlick_G (float cos_w, float a)
{
    float k = a / 8.f;
    return cos_w / (cos_w * (1.f - k) + k);
}

__host__ __device__
float D_GGX(float a2, float NoH)
{
    float d = (NoH * a2 - NoH) * NoH + 1; // 2 mad
    return a2 / (PI * d * d);         // 4 mul, 1 rcp
}


__host__ __device__
void ImportanceSampleGGX_TangentSpace(vec3& OutH, float& pdf, vec2 E, float a2)
{
    float Phi = 2 * PI * E.x;
    float CosTheta = sqrt((1.f - E.y) / (1.f + (a2 - 1) * E.y));
    float SinTheta = sqrt(1.f - CosTheta * CosTheta);

    OutH.x = SinTheta * cos(Phi);
    OutH.y = CosTheta;
    OutH.z = SinTheta * sin(Phi);

    float d = (CosTheta * a2 - CosTheta) * CosTheta + 1.f;
    float D = a2 / (PI * d * d);
    pdf = D * CosTheta;
}

__host__ __device__
float smithG_GGX_aniso(float NdotV, float VdotX, float VdotY, float ax, float ay)
{
    return 1 / (NdotV + sqrt(sqr(VdotX * ax) + sqr(VdotY * ay) + sqr(NdotV)));
}

__host__ __device__
vec3 mon2lin(vec3 x)
{
    return vec3(pow(x[0], 2.2), pow(x[1], 2.2), pow(x[2], 2.2));
}

__host__ __device__ float Cos2Theta(vec3 w) { return sqr(w.z); }
__host__ __device__ float Sin2Theta(vec3 w) { return glm::max(0.f, 1.f - Cos2Theta(w)); }
__host__ __device__ float SinTheta(vec3 w) { return std::sqrt(Sin2Theta(w)); }
__host__ __device__ float CosTheta(vec3 w) { return std::sqrt(Cos2Theta(w)); }
__host__ __device__ float TanTheta(vec3 w) { return SinTheta(w) / CosTheta(w); }
__host__ __device__ float Tan2Theta(vec3 w) { return Sin2Theta(w) / Cos2Theta(w); }
__host__ __device__ float CosPhi(vec3 w) {
    float sinTheta = SinTheta(w);
    return (sinTheta == 0) ? 1 : glm::clamp(w.x / sinTheta, -1.f, 1.f);
}
__host__ __device__ float AbsCosTheta(vec3 w) { return glm::abs(CosTheta(w)); }
__host__ __device__
float SinPhi(vec3 w) {
    float sinTheta = SinTheta(w);
    return (sinTheta == 0) ? 0 : glm::clamp(w.y / sinTheta, -1.f, 1.f);
}
//__host__ __device__
//float D(vec3 wm) {
//    float tan2Theta = Tan2Theta(wm);
//    if (cuda::std::isinf(tan2Theta)) return 0;
//    float cos4Theta = sqr(Cos2Theta(wm));
//    float e = tan2Theta * (sqr(CosPhi(wm) / alpha_x) +
//        sqr(SinPhi(wm) / alpha_y));
//    return 1 / (Pi * alpha_x * alpha_y * cos4Theta * sqr(1 + e));
//}

//__host__ __device__
//float3 F_Schlick(float HdotV, vec3 F0)
//{
//    return F0 + (1.f - F0) * pow(1 - HdotV, 5.0f);
//}

__host__ __device__
vec3 BRDF(BRDF_Params Params, vec3 L, vec3 V, vec3 N)
{
    vec3 H = normalize(L + V);
    float NdotH = dot(N, H);
    float NdotV = dot(N, V);
    float NdotL = dot(N, L);
    float HdotV = dot(H, V);
    float a = Params.roughness * Params.roughness;
    float D = D_GGX(a*a, NdotH);

    // Diffuse fresnel - go from 1 at normal incidence to .5 at grazing
    // and mix in diffuse retro-reflection based on roughness
    float Gs = schlick_G(NdotV, a) * schlick_G(NdotL, a);
    vec3 F0 = mix(vec3(0.04f), Params.baseColor, Params.metallic);
    vec3 Fs = SchlickFresnel(max(HdotV, 0.0f), F0, Params.roughness);
    vec3 f_CookTorrance = D * Fs * Gs / NdotV / NdotL / 4.f;
    vec3 ks = Fs;
    vec3 kd = vec3(1.0f) - ks;
    kd *= (1.f - Params.metallic);
	vec3 f_lambert = Params.baseColor * INV_PI;
    return kd * f_lambert + f_CookTorrance;
}
