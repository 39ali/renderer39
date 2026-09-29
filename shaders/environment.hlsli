#ifndef ENVIRONMENT_HLSLI
#define ENVIRONMENT_HLSLI

#include "brdf.hlsli"
#include "scene_bindings.hlsli"

static const float kInvPi = 0.31830988618;
static const float kInvTwoPi = 0.15915494309;
// used when no environment map is loaded
static const float3 kFallbackAmbient = float3(0.03, 0.035, 0.045);

// +Y is up; u wraps around the horizon, v runs from the zenith down.
float2 directionToEquirect(float3 direction) {
  return float2(atan2(direction.z, direction.x) * kInvTwoPi + 0.5,
                acos(clamp(direction.y, -1.0, 1.0)) * kInvPi);
}

// inverse of directionToEquirect
float3 equirectToDirection(float2 uv) {
  const float phi = (uv.x - 0.5) * 2.0 * kPi;
  const float theta = uv.y * kPi;
  const float sinTheta = sin(theta);
  return float3(sinTheta * cos(phi), cos(theta), sinTheta * sin(phi));
}

bool hasEnvironmentSampling() {
  return pc.environmentTexture != kInvalidTexture &&
         pc.environmentSampling != 0u && pc.environmentTableWidth > 0u;
}

// solid-angle density of sampleEnvironmentDirection producing `direction`
float environmentPdf(float3 direction) {
  const uint width = pc.environmentTableWidth;
  const uint height = pc.environmentTableHeight;
  const float2 uv = directionToEquirect(direction);
  const uint x = min(uint(frac(uv.x) * width), width - 1u);
  const uint y = min(uint(uv.y * height), height - 1u);
  const float pdfUv = environmentAlias[y * width + x].pdf;

  // du dv -> solid angle: d(omega) = sin(theta) * pi * 2pi * du dv
  const float sinTheta = sqrt(max(1.0 - direction.y * direction.y, 0.0));
  return sinTheta > 0.0 ? pdfUv / (2.0 * kPi * kPi * sinTheta) : 0.0;
}

// picks a direction with probability proportional to the environment's
// luminance (times sin(theta)); u holds four uniform random numbers
float3 sampleEnvironmentDirection(float4 u, out float pdfSolidAngle) {
  const uint width = pc.environmentTableWidth;
  const uint height = pc.environmentTableHeight;
  uint index = min(uint(u.x * float(width * height)), width * height - 1u);
  const EnvironmentAliasEntry entry = environmentAlias[index];
  if (u.y >= entry.probability) {
    index = entry.alias;
  }

  // uniform position inside the chosen texel
  const float2 uv =
      (float2(index % width, index / width) + u.zw) / float2(width, height);
  const float3 direction = equirectToDirection(uv);
  pdfSolidAngle = environmentPdf(direction);
  return direction;
}

float3 sampleEnvironment(float3 direction, float mip) {
  if (pc.environmentTexture == kInvalidTexture) {
    return kFallbackAmbient;
  }
  return textures[NonUniformResourceIndex(pc.environmentTexture)]
             .SampleLevel(materialSampler, directionToEquirect(direction), mip)
             .rgb *
         pc.environmentIntensity;
}

#endif
