#include "bsdf.h"
#include "sceneStructs.h"
#include "common.h"
#include <thrust/random.h>

#include "utilities.h"

using namespace glm;

// code below is from https://github.com/wdas/brdf/blob/main/src/brdfs/disney.brdf

__host__ __device__
float SchlickFresnel(float u)
{
    float m = clamp(1.f - u, 0.f, 1.f);
    float m2 = m * m;
    return m2 * m2 * m; // pow(m,5)
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
    float t = 1 + (a2 - 1) * NdotH * NdotH;
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
float smithG_GGX_aniso(float NdotV, float VdotX, float VdotY, float ax, float ay)
{
    return 1 / (NdotV + sqrt(sqr(VdotX * ax) + sqr(VdotY * ay) + sqr(NdotV)));
}

__host__ __device__
vec3 mon2lin(vec3 x)
{
    return vec3(pow(x[0], 2.2), pow(x[1], 2.2), pow(x[2], 2.2));
}

__host__ __device__
vec3 BRDF(BRDF_Params Params, vec3 L, vec3 V, vec3 N, vec3 X, vec3 Y)
{
    float NdotL = dot(N, L);
    float NdotV = dot(N, V);
    if (NdotL < 0 || NdotV < 0) return vec3(0);

    vec3 H = normalize(L + V);
    float NdotH = dot(N, H);
    float LdotH = dot(L, H);

    vec3 Cdlin = mon2lin(Params.baseColor);
    float Cdlum = .3f * Cdlin[0] + .6f * Cdlin[1] + .1f * Cdlin[2]; // luminance approx.

    vec3 Ctint = Cdlum > 0 ? Cdlin / Cdlum : vec3(1); // normalize lum. to isolate hue+sat
    vec3 Cspec0 = mix(Params.specular * .08f * mix(vec3(1), Ctint, Params.specularTint), Cdlin, Params.metallic);
    vec3 Csheen = mix(vec3(1), Ctint, Params.sheenTint);

    // Diffuse fresnel - go from 1 at normal incidence to .5 at grazing
    // and mix in diffuse retro-reflection based on roughness
    float FL = SchlickFresnel(NdotL), FV = SchlickFresnel(NdotV);
    float Fd90 = 0.5f + 2 * LdotH * LdotH * Params.roughness;
    float Fd = mix(1.0f, Fd90, FL) * mix(1.0f, Fd90, FV);

    // Based on Hanrahan-Krueger brdf approximation of isotropic bssrdf
    // 1.25 scale is used to (roughly) preserve albedo
    // Fss90 used to "flatten" retroreflection based on roughness
    float Fss90 = LdotH * LdotH * Params.roughness;
    float Fss = mix(1.0f, Fss90, FL) * mix(1.0f, Fss90, FV);
    float ss = 1.25f * (Fss * (1 / (NdotL + NdotV) - .5f) + .5f);

    // specular
    float aspect = sqrt(1 - Params.anisotropic * .9);
    float ax = max(.001f, sqr(Params.roughness) / aspect);
    float ay = max(.001f, sqr(Params.roughness) * aspect);
    float Ds = GTR2_aniso(NdotH, dot(H, X), dot(H, Y), ax, ay);
    float FH = SchlickFresnel(LdotH);
    vec3 Fs = mix(Cspec0, vec3(1), FH);
    float Gs;
    Gs = smithG_GGX_aniso(NdotL, dot(L, X), dot(L, Y), ax, ay);
    Gs *= smithG_GGX_aniso(NdotV, dot(V, X), dot(V, Y), ax, ay);

    // sheen
    vec3 Fsheen = FH * Params.sheen * Csheen;

    // clearcoat (ior = 1.5 -> F0 = 0.04)
    float Dr = GTR1(NdotH, mix(.1f, .001f, Params.clearcoatGloss));
    float Fr = mix(.04f, 1.0f, FH);
    float Gr = smithG_GGX(NdotL, .25f) * smithG_GGX(NdotV, .25f);

    return ((1 / PI) * mix(Fd, ss, Params.subsurface) * Cdlin + Fsheen)
        * (1 - Params.metallic)
        + Gs * Fs * Ds + .25f * Params.clearcoat * Gr * Fr * Dr;
}