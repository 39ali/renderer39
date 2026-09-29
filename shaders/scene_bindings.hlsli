#ifndef SCENE_BINDINGS_HLSLI
#define SCENE_BINDINGS_HLSLI

#include "shared_structs.h"

// must match Renderer::createDescriptorSetLayout
[[vk::push_constant]] DrawPushConstants pc;

[[vk::binding(0, 0)]] StructuredBuffer<InstanceData> instances;
[[vk::binding(1, 0)]] StructuredBuffer<Meshlet> meshlets;
[[vk::binding(2, 0)]] StructuredBuffer<VertexPosition> vertexPositions;
[[vk::binding(3, 0)]] StructuredBuffer<VertexAttributes> vertexAttributes;
[[vk::binding(4, 0)]] StructuredBuffer<uint> meshletVertexRefs;
[[vk::binding(5, 0)]] StructuredBuffer<uint> meshletTriangles;
[[vk::binding(6, 0)]] StructuredBuffer<MeshletInstance> candidateMeshlets;
[[vk::binding(7, 0)]] RWStructuredBuffer<RenderCounters> counters;
[[vk::binding(8, 0)]] StructuredBuffer<Material> materials;
[[vk::binding(9, 0)]] SamplerState materialSampler;
[[vk::binding(10, 0)]] Texture2D textures[];
[[vk::binding(11, 0)]] RaytracingAccelerationStructure sceneTlas;
// path tracer output: running average of radiance
[[vk::binding(12, 0)]] RWTexture2D<float4> accumulation;
// meshlet id of every triangle, in meshletTriangles order
[[vk::binding(13, 0)]] StructuredBuffer<uint> triangleMeshlets;
[[vk::binding(14, 0)]] StructuredBuffer<EnvironmentAliasEntry> environmentAlias;

// the mesh shader output and fragment shader input
struct VertexOutput {
  float4 position : SV_Position;
  [[vk::location(0)]] float3 worldPosition : WORLD_POSITION;
  [[vk::location(1)]] float3 normal : NORMAL;
  [[vk::location(2)]] float4 tangent : TANGENT;
  [[vk::location(3)]] float2 uv : TEXCOORD0;
  [[vk::location(4)]] nointerpolation uint materialIndex : MATERIAL;
};

#endif
