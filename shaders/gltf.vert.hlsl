#include "shared_structs.h"

struct PushConstants {
  float4x4 mvp;
};
[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] StructuredBuffer<InstanceData> instances;
[[vk::binding(1, 0)]] StructuredBuffer<Meshlet> meshlets;
[[vk::binding(2, 0)]] StructuredBuffer<MeshletVertex> vertices;
[[vk::binding(3, 0)]] StructuredBuffer<uint> clusterTriangles;
[[vk::binding(5, 0)]] StructuredBuffer<MeshletInstance> visibleMeshlets;
[[vk::binding(9, 0)]] StructuredBuffer<uint> meshletVertexRefs;

struct VSOutput {
  float4 position : SV_Position;
  [[vk::location(0)]] float3 normal : NORMAL;
  [[vk::location(1)]] float4 color : COLOR;
};

float4 unpackColor(uint color) {
  return float4(float(color & 0xffu), float((color >> 8) & 0xffu),
                float((color >> 16) & 0xffu), float((color >> 24) & 0xffu)) /
         255.0;
}

// SV_InstanceID maps to gl_InstanceIndex (which includes firstInstance) as long
// as -fvk-support-nonzero-base-instance is not passed; multi-draw relies on it.
VSOutput main(uint localVertexIndex : SV_VertexID,
              uint instanceIndex : SV_InstanceID) {
  VSOutput output;
  MeshletInstance visibleMeshlet = visibleMeshlets[instanceIndex];
  InstanceData instance = instances[visibleMeshlet.instanceId];
  Meshlet meshlet = meshlets[visibleMeshlet.meshletId];

  uint triangleIndex = localVertexIndex / 3u;
  uint triangleCorner = localVertexIndex % 3u;
  if (triangleIndex >= meshletTriangleCount(meshlet)) {
    output.position = float4(2.0, 2.0, 2.0, 1.0);
    output.normal = float3(0.0, 0.0, 1.0);
    output.color = float4(0.0, 0.0, 0.0, 0.0);
    return output;
  }

  uint packedTriangle =
      clusterTriangles[meshlet.triangleOffset + triangleIndex];
  // extracts one of the three 10-bit local vertex indices from the packed
  // triangle.
  uint clusterVertexIndex = (packedTriangle >> (triangleCorner * 10u)) & 0x3ffu;
  uint vertexIndex =
      meshletVertexRefs[meshlet.vertexOffset + clusterVertexIndex];
  MeshletVertex vertex = vertices[vertexIndex];

  float2 positionXY = unpackHalf2x16(vertex.positionXY);
  float2 positionZNormalX = unpackHalf2x16(vertex.positionZNormalX);
  float2 normalYZ = unpackHalf2x16(vertex.normalYZ);
  float3 position = float3(positionXY.x, positionXY.y, positionZNormalX.x);
  float3 normal = normalize(float3(positionZNormalX.y, normalYZ.x, normalYZ.y));

  float3 translation = instance.translationScaleX.xyz;
  float3 scale = float3(instance.translationScaleX.w, instance.scaleYZ.x,
                        instance.scaleYZ.y);
  float3 worldPosition =
      translation + rotateByQuat(position * scale, instance.rotation);

  output.position = mul(pc.mvp, float4(worldPosition, 1.0));
  output.normal = normalize(rotateByQuat(normal, instance.rotation));
  output.color = unpackColor(instance.packedColor);
  return output;
}
