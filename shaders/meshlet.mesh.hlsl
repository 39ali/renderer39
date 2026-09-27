#include "shared_structs.h"

[[vk::push_constant]] DrawPushConstants pc;

[[vk::binding(0, 0)]] StructuredBuffer<InstanceData> instances;
[[vk::binding(1, 0)]] StructuredBuffer<Meshlet> meshlets;
[[vk::binding(2, 0)]] StructuredBuffer<MeshletVertex> vertices;
[[vk::binding(3, 0)]] StructuredBuffer<uint> clusterTriangles;
[[vk::binding(4, 0)]] StructuredBuffer<uint> meshletVertexRefs;

struct VertexOutput {
  float4 position : SV_Position;
  [[vk::location(0)]] float3 normal : NORMAL;
  [[vk::location(1)]] float4 color : COLOR;
};

float4 unpackColor(uint color) {
  return float4(float(color & 0xffu), float((color >> 8) & 0xffu),
                float((color >> 16) & 0xffu), float((color >> 24) & 0xffu)) /
         255.0;
}

// one workgroup per visible meshlet; each thread emits at most one vertex and
// one triangle.
[outputtopology("triangle")]
[numthreads(128, 1, 1)]
void main(uint threadIndex : SV_GroupThreadID, uint3 groupId : SV_GroupID,
          in payload TaskPayload payload,
          out vertices VertexOutput outVertices[kMaxMeshletVertices],
          out indices uint3 outTriangles[kMaxMeshletTriangles]) {
  const MeshletInstance visibleMeshlet = payload.meshlets[groupId.x];
  const InstanceData instance = instances[visibleMeshlet.instanceId];
  const Meshlet meshlet = meshlets[visibleMeshlet.meshletId];
  const uint vertexCount = meshletVertexCount(meshlet);
  const uint triangleCount = meshletTriangleCount(meshlet);

  SetMeshOutputCounts(vertexCount, triangleCount);

  if (threadIndex < vertexCount) {
    const uint vertexIndex = meshletVertexRefs[meshlet.vertexOffset + threadIndex];
    const MeshletVertex vertex = vertices[vertexIndex];

    float2 positionXY = unpackHalf2x16(vertex.positionXY);
    float2 positionZNormalX = unpackHalf2x16(vertex.positionZNormalX);
    float2 normalYZ = unpackHalf2x16(vertex.normalYZ);
    float3 position = float3(positionXY.x, positionXY.y, positionZNormalX.x);
    float3 normal =
        normalize(float3(positionZNormalX.y, normalYZ.x, normalYZ.y));

    float3 scale = float3(instance.translationScaleX.w, instance.scaleYZ.x,
                          instance.scaleYZ.y);
    float3 worldPosition = instance.translationScaleX.xyz +
                           rotateByQuat(position * scale, instance.rotation);

    VertexOutput output;
    output.position = mul(pc.viewProjection, float4(worldPosition, 1.0));
    output.normal = normalize(rotateByQuat(normal, instance.rotation));
    output.color = unpackColor(instance.packedColor);
    outVertices[threadIndex] = output;
  }

  if (threadIndex < triangleCount) {
    // three 10-bit meshlet-local vertex indices
    const uint packedTriangle =
        clusterTriangles[meshlet.triangleOffset + threadIndex];
    outTriangles[threadIndex] =
        uint3(packedTriangle & 0x3ffu, (packedTriangle >> 10u) & 0x3ffu,
              (packedTriangle >> 20u) & 0x3ffu);
  }
}
