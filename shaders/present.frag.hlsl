#include "scene_bindings.hlsli"
#include "tonemap.hlsli"

// shows the path tracer's accumulated radiance
float4 main(float4 position : SV_Position) : SV_Target0 {
  return float4(tonemapAces(accumulation[uint2(position.xy)].rgb), 1.0);
}
