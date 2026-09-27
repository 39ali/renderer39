#include "scene_bindings.hlsli"

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
    const uint vertexIndex =
        meshletVertexRefs[meshlet.vertexOffset + threadIndex];
    const VertexPosition p = vertexPositions[vertexIndex];
    const VertexAttributes attributes = vertexAttributes[vertexIndex];

    const float3 scale = float3(instance.translationScaleX.w,
                                instance.scaleYZ.x, instance.scaleYZ.y);
    const float3 worldPosition =
        instance.translationScaleX.xyz +
        rotateByQuat(float3(p.x, p.y, p.z) * scale, instance.rotation);

    // normals take the inverse scale, tangents the forward scale
    const float3 normal = octDecode(unpackSnorm2x16(attributes.normal));
    const float3 tangent = octDecode(unpackSnorm2x16(attributes.tangent));
    const float tangentSign = (attributes.tangent & 1u) != 0u ? -1.0 : 1.0;

    VertexOutput output;
    output.position = mul(pc.viewProjection, float4(worldPosition, 1.0));
    output.worldPosition = worldPosition;
    output.normal =
        normalize(rotateByQuat(normal / scale, instance.rotation));
    output.tangent = float4(
        normalize(rotateByQuat(tangent * scale, instance.rotation)),
        tangentSign);
    output.uv = attributes.uv;
    output.materialIndex = meshlet.materialIndex;
    outVertices[threadIndex] = output;
  }

  if (threadIndex < triangleCount) {
    // three 10-bit meshlet-local vertex indices
    const uint packedTriangle =
        meshletTriangles[meshlet.triangleOffset + threadIndex];
    outTriangles[threadIndex] =
        uint3(packedTriangle & 0x3ffu, (packedTriangle >> 10u) & 0x3ffu,
              (packedTriangle >> 20u) & 0x3ffu);
  }
}
