#include "shared_structs.h"

struct PushConstants {
  float4x4 view;
  float4 frustum;
  float2 zNearFar;
  uint candidateCount;
  uint mode; // 0 = SingleDraw, 1 = MultiDraw
  uint meshletDrawCount;
};
[[vk::push_constant]] PushConstants pc;

struct Counters {
  uint visibleMeshletCount;
  uint visibleTriangleCount;
};

[[vk::binding(0, 0)]] StructuredBuffer<InstanceData> instances;
[[vk::binding(1, 0)]] StructuredBuffer<Meshlet> meshlets;
[[vk::binding(4, 0)]] StructuredBuffer<MeshletInstance> candidateMeshlets;
[[vk::binding(5, 0)]] RWStructuredBuffer<MeshletInstance> visibleMeshlets;
[[vk::binding(6, 0)]] RWStructuredBuffer<DrawIndirectCommand> drawCommands;
[[vk::binding(7, 0)]] RWStructuredBuffer<Counters> counters;
[[vk::binding(8, 0)]] StructuredBuffer<MeshletDrawMeta> meshletDrawMetas;

bool frustumCull(float3 viewCenter, float radius) {
  const float depth = -viewCenter.z;

  // checks the left/right frustum planes
  bool visible =
      depth * pc.frustum.y - abs(viewCenter.x) * pc.frustum.x > -radius;
  // top/bottom
  visible = visible &&
            depth * pc.frustum.w - abs(viewCenter.y) * pc.frustum.z > -radius;
  // near/far
  visible = visible && depth + radius > pc.zNearFar.x &&
            depth - radius < pc.zNearFar.y;
  return visible;
}

bool coneCull(float3 center, float radius, float3 coneAxis, float coneCutoff,
              float3 cameraPosition) {
  float3 view = center - cameraPosition;
  return dot(view, coneAxis) >= coneCutoff * length(view) + radius;
}

bool coneBackfacing(float3 viewCenter, float worldRadius, uint boundingCone,
                    float4 rotation) {
  float4 cone = unpackSnorm4x8(boundingCone);
  float3 worldAxis = normalize(rotateByQuat(cone.xyz, rotation));
  float3 viewAxis = normalize(mul((float3x3)pc.view, worldAxis));
  return coneCull(viewCenter, worldRadius, viewAxis, cone.w,
                  float3(0.0, 0.0, 0.0));
}

[numthreads(64, 1, 1)]
void main(uint3 dispatchId : SV_DispatchThreadID) {
  const uint candidateId = dispatchId.x;

  if (pc.mode == 0u) {
    if (candidateId == 0) {
      drawCommands[0].vertexCount = kSyntheticIndicesPerMeshlet;
      drawCommands[0].firstVertex = 0u;
      drawCommands[0].firstInstance = 0u;
    }
  } else if (candidateId < pc.meshletDrawCount) {
    MeshletDrawMeta meta = meshletDrawMetas[candidateId];
    Meshlet drawMeshlet = meshlets[meta.meshletId];
    drawCommands[candidateId].vertexCount =
        meshletTriangleCount(drawMeshlet) * 3u;
    drawCommands[candidateId].firstVertex = 0u;
    drawCommands[candidateId].firstInstance = meta.visibleInstanceOffset;
  }

  if (candidateId >= pc.candidateCount) {
    return;
  }

  MeshletInstance candidate = candidateMeshlets[candidateId];
  InstanceData instance = instances[candidate.instanceId];
  Meshlet meshlet = meshlets[candidate.meshletId];
  float4 boundingSphere = unpackMeshletSphere(meshlet);

  float3 scale = float3(instance.translationScaleX.w, instance.scaleYZ.x,
                        instance.scaleYZ.y);
  float3 worldCenter =
      instance.translationScaleX.xyz +
      rotateByQuat(boundingSphere.xyz * scale, instance.rotation);
  float worldRadius =
      boundingSphere.w * max(max(abs(scale.x), abs(scale.y)), abs(scale.z));
  float3 viewCenter = mul(pc.view, float4(worldCenter, 1.0)).xyz;

  if (!frustumCull(viewCenter, worldRadius)) {
    return;
  }
  if (coneBackfacing(viewCenter, worldRadius, meshlet.boundingCone,
                     instance.rotation)) {
    return;
  }

  uint ignored;
  InterlockedAdd(counters[0].visibleMeshletCount, 1u, ignored);
  InterlockedAdd(counters[0].visibleTriangleCount,
                 meshletTriangleCount(meshlet), ignored);

  uint visibleId = 0u;
  if (pc.mode == 0u) {
    InterlockedAdd(drawCommands[0].instanceCount, 1u, visibleId);
  } else {
    MeshletDrawMeta meta = meshletDrawMetas[candidate.meshletId];
    uint meshletInstanceId;
    InterlockedAdd(drawCommands[candidate.meshletId].instanceCount, 1u,
                   meshletInstanceId);
    visibleId = meta.visibleInstanceOffset + meshletInstanceId;
  }

  visibleMeshlets[visibleId].instanceId = candidate.instanceId;
  visibleMeshlets[visibleId].meshletId = candidate.meshletId;
}
