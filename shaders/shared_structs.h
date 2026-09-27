#ifndef SHARED_STRUCTS_H
#define SHARED_STRUCTS_H

#ifdef __cplusplus
#include <cstddef>
#include <cstdint>

#include <glm/glm.hpp>

using uint = std::uint32_t;
using vec2 = glm::vec2;
using vec4 = glm::vec4;
#define SHARED_INLINE inline
#else
// HLSL
typedef float2 vec2;
typedef float4 vec4;
#define SHARED_INLINE
#endif

// 48 bytes
struct InstanceData {
  vec4 translationScaleX;
  vec4 rotation;
  vec2 scaleYZ;
  uint packedColor;
  uint padding;
};

// 24 bytes
struct Meshlet {
  uint packedSphere0;
  uint packedSphere1;
  // Offset into the meshlet vertex reference buffer.
  uint vertexOffset;
  uint triangleOffset;
  uint packedCountsAndFlags;
  uint boundingCone;
};

// 12 bytes
struct MeshletVertex {
  uint positionXY;
  uint positionZNormalX;
  uint normalYZ;
};

struct MeshletInstance {
  uint instanceId;
  uint meshletId;
};

struct MeshletDrawMeta {
  uint visibleInstanceOffset;
  uint meshletId;
};

struct DrawIndirectCommand {
  uint vertexCount;
  uint instanceCount;
  uint firstVertex;
  uint firstInstance;
};

SHARED_INLINE uint meshletTriangleCount(
#ifdef __cplusplus
    const Meshlet &meshlet
#else
    Meshlet meshlet
#endif
) {
  return (meshlet.packedCountsAndFlags >> 8u) & 0xffu;
}

#ifndef __cplusplus
static const uint kMaxMeshletTriangles = 124u;
static const uint kSyntheticIndicesPerMeshlet = kMaxMeshletTriangles * 3u;

float2 unpackHalf2x16(uint packed) {
  return float2(f16tof32(packed), f16tof32(packed >> 16u));
}

float4 unpackMeshletSphere(Meshlet meshlet) {
  float2 centerXY = unpackHalf2x16(meshlet.packedSphere0);
  float2 centerZRadius = unpackHalf2x16(meshlet.packedSphere1);
  return float4(centerXY, centerZRadius);
}

float unpackSnorm8(uint value) {
  int signedValue = int(value & 0xffu);
  if (signedValue >= 128) {
    signedValue -= 256;
  }
  return clamp(float(signedValue) / 127.0, -1.0, 1.0);
}

float4 unpackSnorm4x8(uint packed) {
  return float4(unpackSnorm8(packed), unpackSnorm8(packed >> 8u),
                unpackSnorm8(packed >> 16u), unpackSnorm8(packed >> 24u));
}

float3 rotateByQuat(float3 v, float4 q) {
  return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}
#endif

#ifdef __cplusplus
static_assert(sizeof(InstanceData) == 48);
static_assert(offsetof(InstanceData, translationScaleX) == 0);
static_assert(offsetof(InstanceData, rotation) == 16);
static_assert(offsetof(InstanceData, scaleYZ) == 32);
static_assert(offsetof(InstanceData, packedColor) == 40);
static_assert(offsetof(InstanceData, padding) == 44);

static_assert(sizeof(Meshlet) == 24);
static_assert(offsetof(Meshlet, packedSphere0) == 0);
static_assert(offsetof(Meshlet, packedSphere1) == 4);
static_assert(offsetof(Meshlet, vertexOffset) == 8);
static_assert(offsetof(Meshlet, triangleOffset) == 12);
static_assert(offsetof(Meshlet, packedCountsAndFlags) == 16);
static_assert(offsetof(Meshlet, boundingCone) == 20);

static_assert(sizeof(MeshletVertex) == 12);
static_assert(sizeof(MeshletInstance) == 8);
static_assert(sizeof(MeshletDrawMeta) == 8);
static_assert(sizeof(DrawIndirectCommand) == 16);
#endif

#undef SHARED_INLINE

#endif
