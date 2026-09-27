#ifndef ENVIRONMENT_HLSLI
#define ENVIRONMENT_HLSLI

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
