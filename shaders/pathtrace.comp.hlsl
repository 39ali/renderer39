#include "environment.hlsli"
#include "sampling.hlsli"

// Reference path tracer: one path per pixel per frame, averaged into
// `accumulation` until pc.sampleIndex is reset. Next-event estimation for the
// directional light; the environment and emissive surfaces are only found by
// BRDF sampling.

static const float3 kLightColor = float3(1.0, 0.96, 0.9);
static const uint kRussianRouletteStart = 3u;

struct HitSurface {
  float3 position;
  // faces the incoming ray
  float3 geometricNormal;
  // normal-mapped, on the same side as geometricNormal
  float3 normal;
  SurfaceProperties material;
  float3 emissive;
};

float4 sampleMaterialTexture(uint textureIndex, float2 uv, float4 fallback) {
  if (textureIndex == kInvalidTexture) {
    return fallback;
  }
  // no screen-space derivatives in compute; the reference uses mip 0
  return textures[NonUniformResourceIndex(textureIndex)].SampleLevel(
      materialSampler, uv, 0.0);
}

float3 transformPosition(InstanceData instance, float3 p) {
  const float3 scale = float3(instance.translationScaleX.w, instance.scaleYZ.x,
                              instance.scaleYZ.y);
  return instance.translationScaleX.xyz +
         rotateByQuat(p * scale, instance.rotation);
}

float3 loadPosition(uint vertexIndex) {
  const VertexPosition p = vertexPositions[vertexIndex];
  return float3(p.x, p.y, p.z);
}

HitSurface loadHit(uint instanceId, uint primitiveIndex, float2 barycentrics,
                   float3 rayDirection) {
  const InstanceData instance = instances[instanceId];
  const uint triangleIndex = instance.firstTriangle + primitiveIndex;
  const Meshlet meshlet = meshlets[triangleMeshlets[triangleIndex]];
  const uint packedTriangle = meshletTriangles[triangleIndex];
  const uint3 local = uint3(packedTriangle & 0x3ffu,
                            (packedTriangle >> 10u) & 0x3ffu,
                            (packedTriangle >> 20u) & 0x3ffu);
  const uint3 v = uint3(meshletVertexRefs[meshlet.vertexOffset + local.x],
                        meshletVertexRefs[meshlet.vertexOffset + local.y],
                        meshletVertexRefs[meshlet.vertexOffset + local.z]);
  const float3 w = float3(1.0 - barycentrics.x - barycentrics.y,
                          barycentrics.x, barycentrics.y);

  const float3 p0 = transformPosition(instance, loadPosition(v.x));
  const float3 p1 = transformPosition(instance, loadPosition(v.y));
  const float3 p2 = transformPosition(instance, loadPosition(v.z));

  const VertexAttributes a0 = vertexAttributes[v.x];
  const VertexAttributes a1 = vertexAttributes[v.y];
  const VertexAttributes a2 = vertexAttributes[v.z];
  const float3 scale = float3(instance.translationScaleX.w, instance.scaleYZ.x,
                              instance.scaleYZ.y);
  float3 normal = octDecode(unpackSnorm2x16(a0.normal)) * w.x +
                  octDecode(unpackSnorm2x16(a1.normal)) * w.y +
                  octDecode(unpackSnorm2x16(a2.normal)) * w.z;
  normal = normalize(rotateByQuat(normal / scale, instance.rotation));
  float3 tangent = octDecode(unpackSnorm2x16(a0.tangent)) * w.x +
                   octDecode(unpackSnorm2x16(a1.tangent)) * w.y +
                   octDecode(unpackSnorm2x16(a2.tangent)) * w.z;
  tangent = rotateByQuat(tangent * scale, instance.rotation);
  const float tangentSign = (a0.tangent & 1u) != 0u ? -1.0 : 1.0;
  const float2 uv = a0.uv * w.x + a1.uv * w.y + a2.uv * w.z;

  const Material material = materials[meshlet.materialIndex];
  const float4 baseColor =
      material.baseColorFactor *
      sampleMaterialTexture(material.baseColorTexture, uv, 1.0);
  const float4 metallicRoughness =
      sampleMaterialTexture(material.metallicRoughnessTexture, uv, 1.0);

  HitSurface hit;
  hit.position = p0 * w.x + p1 * w.y + p2 * w.z;
  hit.material.baseColor = baseColor.rgb;
  hit.material.metallic =
      saturate(material.metallicFactor * metallicRoughness.b);
  hit.material.roughness =
      saturate(material.roughnessFactor * metallicRoughness.g);
  hit.emissive = material.emissiveFactor *
                 sampleMaterialTexture(material.emissiveTexture, uv, 1.0).rgb;

  if (material.normalTexture != kInvalidTexture) {
    float3 tangentNormal =
        sampleMaterialTexture(material.normalTexture, uv, 0.0).xyz * 2.0 - 1.0;
    tangentNormal.xy *= material.normalScale;
    const float3 t = normalize(tangent - normal * dot(normal, tangent));
    const float3 b = cross(normal, t) * tangentSign;
    normal = normalize(t * tangentNormal.x + b * tangentNormal.y +
                       normal * tangentNormal.z);
  }

  // every surface is treated as two-sided
  hit.geometricNormal = normalize(cross(p1 - p0, p2 - p0));
  if (dot(hit.geometricNormal, rayDirection) > 0.0) {
    hit.geometricNormal = -hit.geometricNormal;
  }
  if (dot(normal, hit.geometricNormal) < 0.0) {
    normal = -normal;
  }
  hit.normal = normal;
  return hit;
}

// offsets along the geometric normal to avoid self-intersection
float3 offsetRayOrigin(float3 position, float3 geometricNormal) {
  const float scale = 1e-4 * (1.0 + max(max(abs(position.x), abs(position.y)),
                                        abs(position.z)));
  return position + geometricNormal * scale;
}

bool isVisible(float3 origin, float3 direction) {
  RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH |
           RAY_FLAG_SKIP_CLOSEST_HIT_SHADER>
      query;
  RayDesc ray;
  ray.Origin = origin;
  ray.Direction = direction;
  ray.TMin = 0.0;
  ray.TMax = 1e30;
  query.TraceRayInline(sceneTlas, RAY_FLAG_NONE, 0xffu, ray);
  query.Proceed();
  return query.CommittedStatus() == COMMITTED_NOTHING;
}

float luminance(float3 color) {
  return dot(color, float3(0.2126, 0.7152, 0.0722));
}

float3 tracePath(float3 origin, float3 direction, inout Rng rng) {
  float3 radiance = 0.0;
  float3 throughput = 1.0;
  const float3 toLight = -normalize(pc.lightDirection.xyz);

  for (uint bounce = 0u; bounce <= pc.maxBounces; ++bounce) {
    RayQuery<RAY_FLAG_FORCE_OPAQUE> query;
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = 0.0;
    ray.TMax = 1e30;
    query.TraceRayInline(sceneTlas, RAY_FLAG_NONE, 0xffu, ray);
    query.Proceed();

    if (query.CommittedStatus() != COMMITTED_TRIANGLE_HIT) {
      radiance += throughput * sampleEnvironment(direction, 0.0);
      break;
    }

    const HitSurface hit = loadHit(
        query.CommittedInstanceID(), query.CommittedPrimitiveIndex(),
        query.CommittedTriangleBarycentrics(), direction);
    radiance += throughput * hit.emissive;
    if (bounce == pc.maxBounces) {
      break;
    }

    const float3 v = -direction;
    const float3 n = hit.normal;
    const float3 nextOrigin = offsetRayOrigin(hit.position, hit.geometricNormal);

    // directional light: a delta light, so only reachable by NEE
    if (pc.lightDirection.w > 0.0 && dot(n, toLight) > 0.0 &&
        dot(hit.geometricNormal, toLight) > 0.0 &&
        isVisible(nextOrigin, toLight)) {
      radiance += throughput * evaluateBrdf(hit.material, n, v, toLight) *
                  kLightColor * pc.lightDirection.w;
    }

    // pick the specular or diffuse lobe; the pdf below is the mixture of both
    // so either choice is weighted correctly.
    const float alpha =
        max(hit.material.roughness * hit.material.roughness, 1e-3);
    const float specularProbability =
        lerp(0.25, 1.0, hit.material.metallic);
    const float2 u = float2(nextFloat(rng), nextFloat(rng));
    const float3 l = nextFloat(rng) < specularProbability
                         ? sampleGgxReflection(n, v, alpha, u)
                         : sampleCosineHemisphere(n, u);
    const float nDotL = dot(n, l);
    if (nDotL <= 0.0 || dot(hit.geometricNormal, l) <= 0.0) {
      break;
    }
    const float pdf = specularProbability * ggxReflectionPdf(n, v, l, alpha) +
                      (1.0 - specularProbability) * nDotL / kPi;
    if (pdf <= 0.0) {
      break;
    }
    throughput *= evaluateBrdf(hit.material, n, v, l) / pdf;

    if (bounce >= kRussianRouletteStart) {
      const float survival =
          min(max(max(throughput.x, throughput.y), throughput.z), 0.95);
      if (nextFloat(rng) >= survival) {
        break;
      }
      throughput /= survival;
    }

    origin = nextOrigin;
    direction = l;
  }
  return radiance;
}

[numthreads(8, 8, 1)]
void main(uint3 dispatchId : SV_DispatchThreadID) {
  uint width, height;
  accumulation.GetDimensions(width, height);
  if (dispatchId.x >= width || dispatchId.y >= height) {
    return;
  }

  Rng rng = createRng(dispatchId.xy, pc.sampleIndex);
  const float2 pixel =
      float2(dispatchId.xy) + float2(nextFloat(rng), nextFloat(rng));
  const float2 ndc = pixel / float2(width, height) * 2.0 - 1.0;

  // same camera reconstruction as the sky pass
  const float tanHalfFovX = pc.frustum.y / pc.frustum.x;
  const float tanHalfFovY = pc.frustum.w / pc.frustum.z;
  const float3 direction =
      normalize(ndc.x * tanHalfFovX * pc.view[0].xyz -
                ndc.y * tanHalfFovY * pc.view[1].xyz - pc.view[2].xyz);

  float3 radiance = tracePath(pc.cameraPosition.xyz, direction, rng);
  if (any(isnan(radiance)) || any(isinf(radiance))) {
    radiance = 0.0;
  }

  const float3 previous =
      pc.sampleIndex == 0u ? 0.0 : accumulation[dispatchId.xy].rgb;
  accumulation[dispatchId.xy] = float4(
      lerp(previous, radiance, 1.0 / float(pc.sampleIndex + 1u)), 1.0);
}
