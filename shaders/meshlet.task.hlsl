#include "scene_bindings.hlsli"

groupshared TaskPayload payload;

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

bool coneBackfacing(float3 viewCenter, float radius, uint boundingCone,
                    float4 rotation) {
  float4 cone = unpackSnorm4x8(boundingCone);
  float3 worldAxis = normalize(rotateByQuat(cone.xyz, rotation));
  float3 viewAxis = normalize(mul((float3x3)pc.view, worldAxis));
  // the camera sits at the view-space origin
  return dot(viewCenter, viewAxis) >= cone.w * length(viewCenter) + radius;
}

bool isVisible(MeshletInstance candidate) {
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

  return frustumCull(viewCenter, worldRadius) &&
         !coneBackfacing(viewCenter, worldRadius, meshlet.boundingCone,
                         instance.rotation);
}

// one invocation per candidate meshlet. The survivors are compacted into the
// payload with wave intrinsics, which assumes a 32-wide wave (NVIDIA).
[numthreads(kTaskGroupSize, 1, 1)]
void main(uint3 groupId : SV_GroupID, uint groupIndex : SV_GroupIndex) {
  const uint candidateId =
      (groupId.y * kTaskGroupsPerRow + groupId.x) * kTaskGroupSize + groupIndex;

  bool visible = false;
  MeshletInstance candidate = (MeshletInstance)0;
  if (candidateId < pc.candidateCount) {
    candidate = candidateMeshlets[candidateId];
    visible = isVisible(candidate);
  }

  if (visible) {
    payload.meshlets[WavePrefixCountBits(visible)] = candidate;
  }
  const uint visibleCount = WaveActiveCountBits(visible);

  const uint visibleTriangles = WaveActiveSum(
      visible ? meshletTriangleCount(meshlets[candidate.meshletId]) : 0u);
  if (WaveIsFirstLane() && visibleCount > 0) {
    uint ignored;
    InterlockedAdd(counters[0].visibleMeshletCount, visibleCount, ignored);
    InterlockedAdd(counters[0].visibleTriangleCount, visibleTriangles, ignored);
  }

  DispatchMesh(visibleCount, 1, 1, payload);
}
