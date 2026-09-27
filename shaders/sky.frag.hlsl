#include "environment.hlsli"
#include "tonemap.hlsli"

float4 main([[vk::location(0)]] float2 ndc : NDC) : SV_Target0 {
  // rows of the view matrix are the camera axes in world space; the view
  // looks down -Z and Vulkan NDC has +Y pointing down.
  const float tanHalfFovX = pc.frustum.y / pc.frustum.x;
  const float tanHalfFovY = pc.frustum.w / pc.frustum.z;
  const float3 direction =
      normalize(ndc.x * tanHalfFovX * pc.view[0].xyz -
                ndc.y * tanHalfFovY * pc.view[1].xyz - pc.view[2].xyz);
  return float4(tonemapAces(sampleEnvironment(direction, 0.0)), 1.0);
}
