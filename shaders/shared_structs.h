#ifndef SHARED_STRUCTS_H
#define SHARED_STRUCTS_H

#ifdef __cplusplus
#include <cstddef>
#include <cstdint>

#include <glm/glm.hpp>

using uint = std::uint32_t;
using vec2 = glm::vec2;
using vec3 = glm::vec3;
using vec4 = glm::vec4;
using mat4 = glm::mat4;
#define SHARED_INLINE inline
#else
// HLSL
typedef float2 vec2;
typedef float3 vec3;
typedef float4 vec4;
typedef float4x4 mat4;
#define SHARED_INLINE
#endif

static const uint kMaxMeshletVertices = 64u;
static const uint kMaxMeshletTriangles = 124u;
// one task shader invocation per candidate meshlet; must equal the wave size
// (32 on NVIDIA) because the task shader compacts with wave intrinsics.
static const uint kTaskGroupSize = 32u;
// task workgroups are laid out as a 2D grid so large scenes stay under
// maxTaskWorkGroupCount[0].
static const uint kTaskGroupsPerRow = 65535u;

static const uint kMaxTextures = 4096u;
static const uint kInvalidTexture = 0xffffffffu;

static const uint kAlphaModeOpaque = 0u;
static const uint kAlphaModeMask = 1u;
// blended materials are drawn like masked ones until there is a
// transparency pass.
static const uint kAlphaModeBlend = 2u;

static const uint kDebugViewLit = 0u;
static const uint kDebugViewBaseColor = 1u;
static const uint kDebugViewNormal = 2u;
static const uint kDebugViewMetallic = 3u;
static const uint kDebugViewRoughness = 4u;
static const uint kDebugViewOcclusion = 5u;
static const uint kDebugViewEmissive = 6u;
static const uint kDebugViewUv = 7u;

// 48 bytes
struct InstanceData {
  vec4 translationScaleX;
  vec4 rotation;
  vec2 scaleYZ;
  // index of the mesh's first triangle in meshletTriangles/triangleMeshlets;
  // ray tracing primitive indices are relative to it.
  uint firstTriangle;
  uint padding0;
};

// 32 bytes
struct Meshlet {
  uint packedSphere0;
  uint packedSphere1;
  // offset into the meshlet vertex reference buffer.
  uint vertexOffset;
  uint triangleOffset;
  uint packedCountsAndFlags;
  uint boundingCone;
  uint materialIndex;
  uint padding;
};

// full precision so the same stream can feed ray tracing acceleration
// structures.
struct VertexPosition {
  float x;
  float y;
  float z;
};

// 16 bytes
struct VertexAttributes {
  vec2 uv;
  // octahedral snorm16x2
  uint normal;
  // octahedral snorm16x2; the lowest bit of x holds the bitangent sign
  // (set = -1).
  uint tangent;
};

// 80 bytes. Texture indices index the bindless texture array, or are
// kInvalidTexture.
struct Material {
  vec4 baseColorFactor;
  vec3 emissiveFactor;
  float metallicFactor;
  float roughnessFactor;
  float normalScale;
  float occlusionStrength;
  float alphaCutoff;
  uint baseColorTexture;
  uint metallicRoughnessTexture;
  uint normalTexture;
  uint occlusionTexture;
  uint emissiveTexture;
  uint alphaMode;
  uint padding0;
  uint padding1;
};

// one entry of the environment map's alias table (Vose's method); the table
// is a downsampled copy of the HDR, one entry per texel, row-major.
struct EnvironmentAliasEntry {
  // chance of keeping this texel instead of `alias`
  float probability;
  uint alias;
  // density of this texel over the [0,1]^2 equirect square
  float pdf;
  uint padding;
};

struct MeshletInstance {
  uint instanceId;
  uint meshletId;
};

struct RenderCounters {
  uint visibleMeshletCount;
  uint visibleTriangleCount;
};

// shared by the task, mesh and fragment stages.
struct DrawPushConstants {
  mat4 viewProjection;
  mat4 view;
  vec4 frustum;
  vec2 zNearFar;
  uint candidateCount;
  uint debugView;
  vec4 cameraPosition;
  // xyz: direction the light travels, w: intensity
  vec4 lightDirection;
  // equirectangular HDR in the bindless array, or kInvalidTexture
  uint environmentTexture;
  float environmentIntensity;
  float environmentMaxMip;
  // path tracer: samples accumulated so far (0 restarts accumulation)
  uint sampleIndex;
  uint maxBounces;
  // size of the environment alias table (0 when there is none)
  uint environmentTableWidth;
  uint environmentTableHeight;
  // 1: sample the environment directly and combine with BRDF sampling (MIS)
  uint environmentSampling;
};

SHARED_INLINE uint meshletVertexCount(
#ifdef __cplusplus
    const Meshlet &meshlet
#else
    Meshlet meshlet
#endif
) {
  return meshlet.packedCountsAndFlags & 0xffu;
}

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
struct TaskPayload {
  MeshletInstance meshlets[kTaskGroupSize];
};

float2 unpackHalf2x16(uint packed) {
  return float2(f16tof32(packed), f16tof32(packed >> 16u));
}

float2 unpackSnorm2x16(uint packed) {
  int2 values = int2(int(packed << 16u), int(packed)) >> 16;
  return max(float2(values) / 32767.0, -1.0);
}

float3 octDecode(float2 e) {
  float3 n = float3(e, 1.0 - abs(e.x) - abs(e.y));
  float t = saturate(-n.z);
  n.xy += select(n.xy >= 0.0, -t, t);
  return normalize(n);
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

static_assert(sizeof(Meshlet) == 32);
static_assert(offsetof(Meshlet, materialIndex) == 24);

static_assert(sizeof(VertexPosition) == 12);
static_assert(sizeof(VertexAttributes) == 16);

// matches HLSL StructuredBuffer layout, where float3 is 16-byte aligned.
static_assert(sizeof(Material) == 80);
static_assert(offsetof(Material, emissiveFactor) == 16);
static_assert(offsetof(Material, metallicFactor) == 28);
static_assert(offsetof(Material, baseColorTexture) == 48);
static_assert(offsetof(Material, alphaMode) == 68);

static_assert(sizeof(MeshletInstance) == 8);
static_assert(sizeof(EnvironmentAliasEntry) == 16);
static_assert(sizeof(RenderCounters) == 8);

static_assert(sizeof(DrawPushConstants) == 224);
static_assert(offsetof(DrawPushConstants, sampleIndex) == 204);
static_assert(offsetof(DrawPushConstants, maxBounces) == 208);
static_assert(offsetof(DrawPushConstants, environmentTexture) == 192);
static_assert(offsetof(DrawPushConstants, view) == 64);
static_assert(offsetof(DrawPushConstants, frustum) == 128);
static_assert(offsetof(DrawPushConstants, zNearFar) == 144);
static_assert(offsetof(DrawPushConstants, candidateCount) == 152);
static_assert(offsetof(DrawPushConstants, cameraPosition) == 160);
static_assert(offsetof(DrawPushConstants, lightDirection) == 176);
#endif

#undef SHARED_INLINE

#endif
