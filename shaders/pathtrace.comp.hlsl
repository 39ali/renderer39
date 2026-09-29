#include "environment.hlsli"
#include "sampling.hlsli"

// Reference path tracer: one path per pixel per frame, averaged into
// `accumulation` until pc.sampleIndex is reset. At every hit it samples the
// directional light, the environment and the emissive triangles directly
// (next-event estimation) and combines the last two with BRDF sampling via
// multiple importance sampling.

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
  // world-space triangle area and emissive light-selection weight (0 when the
  // triangle is not in the emissive light list)
  float triangleArea;
  float lightWeight;
};

// a triangle of an instance, in world space
struct TriangleVertices {
  float3 p0;
  float3 p1;
  float3 p2;
  VertexAttributes a0;
  VertexAttributes a1;
  VertexAttributes a2;
  uint materialIndex;
  // into meshletTriangles, triangleMeshlets and triangleEmission
  uint triangleIndex;
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

TriangleVertices loadTriangle(uint instanceId, uint primitiveIndex) {
  const InstanceData instance = instances[instanceId];
  TriangleVertices triangleVertices;
  triangleVertices.triangleIndex = instance.firstTriangle + primitiveIndex;
  const Meshlet meshlet =
      meshlets[triangleMeshlets[triangleVertices.triangleIndex]];
  const uint packedTriangle = meshletTriangles[triangleVertices.triangleIndex];
  const uint3 local = uint3(packedTriangle & 0x3ffu,
                            (packedTriangle >> 10u) & 0x3ffu,
                            (packedTriangle >> 20u) & 0x3ffu);
  const uint3 v = uint3(meshletVertexRefs[meshlet.vertexOffset + local.x],
                        meshletVertexRefs[meshlet.vertexOffset + local.y],
                        meshletVertexRefs[meshlet.vertexOffset + local.z]);
  triangleVertices.p0 = transformPosition(instance, loadPosition(v.x));
  triangleVertices.p1 = transformPosition(instance, loadPosition(v.y));
  triangleVertices.p2 = transformPosition(instance, loadPosition(v.z));
  triangleVertices.a0 = vertexAttributes[v.x];
  triangleVertices.a1 = vertexAttributes[v.y];
  triangleVertices.a2 = vertexAttributes[v.z];
  triangleVertices.materialIndex = meshlet.materialIndex;
  return triangleVertices;
}

float3 emissionAt(Material material, float2 uv) {
  return material.emissiveFactor * pc.emissiveIntensity *
         sampleMaterialTexture(material.emissiveTexture, uv, 1.0).rgb;
}

HitSurface loadHit(uint instanceId, uint primitiveIndex, float2 barycentrics,
                   float3 rayDirection) {
  const InstanceData instance = instances[instanceId];
  const TriangleVertices tri = loadTriangle(instanceId, primitiveIndex);
  const float3 w = float3(1.0 - barycentrics.x - barycentrics.y,
                          barycentrics.x, barycentrics.y);
  const float3 p0 = tri.p0;
  const float3 p1 = tri.p1;
  const float3 p2 = tri.p2;
  const VertexAttributes a0 = tri.a0;
  const VertexAttributes a1 = tri.a1;
  const VertexAttributes a2 = tri.a2;
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

  const Material material = materials[tri.materialIndex];
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
  hit.emissive = emissionAt(material, uv);

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
  const float3 faceNormal = cross(p1 - p0, p2 - p0);
  hit.triangleArea = 0.5 * length(faceNormal);
  hit.lightWeight = triangleEmission[tri.triangleIndex] * hit.triangleArea;
  hit.geometricNormal = normalize(faceNormal);
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

bool isVisible(float3 origin, float3 direction, float maxDistance) {
  RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH |
           RAY_FLAG_SKIP_CLOSEST_HIT_SHADER>
      query;
  RayDesc ray;
  ray.Origin = origin;
  ray.Direction = direction;
  ray.TMin = 0.0;
  ray.TMax = maxDistance;
  query.TraceRayInline(sceneTlas, RAY_FLAG_NONE, 0xffu, ray);
  query.Proceed();
  return query.CommittedStatus() == COMMITTED_NOTHING;
}

float luminance(float3 color) {
  return dot(color, float3(0.2126, 0.7152, 0.0722));
}

float specularLobeProbability(SurfaceProperties material) {
  return lerp(0.25, 1.0, material.metallic);
}

// density of the BRDF sampling below producing l: a mixture of the GGX and
// cosine lobes, so either choice is weighted correctly.
float brdfSamplePdf(SurfaceProperties material, float3 n, float3 v, float3 l) {
  const float alpha = max(material.roughness * material.roughness, 1e-3);
  const float specularProbability = specularLobeProbability(material);
  return specularProbability * ggxReflectionPdf(n, v, l, alpha) +
         (1.0 - specularProbability) * saturate(dot(n, l)) / kPi;
}

bool hasEmissiveSampling() {
  return pc.emissiveSampling != 0u && pc.emissiveLightCount > 0u;
}

// solid-angle density of sampleEmissiveLight picking a point on a triangle
// with this light weight and area, seen at `distance` under `cosLight`
float emissiveLightPdf(float lightWeight, float triangleArea, float distance,
                       float cosLight) {
  if (lightWeight <= 0.0 || cosLight <= 0.0) {
    return 0.0;
  }
  // P(triangle) * (1 / area) converted from area to solid angle
  return lightWeight / pc.emissiveTotalWeight * distance * distance /
         (triangleArea * cosLight);
}

struct EmissiveSample {
  float3 direction;
  float distance;
  float3 radiance;
  // solid angle
  float pdf;
};

// picks an emissive triangle proportional to its weight, then a uniform
// point on it
EmissiveSample sampleEmissiveLight(float3 origin, float4 u) {
  uint index = min(uint(u.x * float(pc.emissiveLightCount)),
                   pc.emissiveLightCount - 1u);
  EmissiveLight light = emissiveLights[index];
  if (u.y >= light.probability) {
    light = emissiveLights[light.alias];
  }
  const TriangleVertices tri =
      loadTriangle(light.instanceId, light.primitiveIndex);

  // uniform barycentrics (Osada et al. 2002)
  const float rootU = sqrt(u.z);
  const float3 w = float3(1.0 - rootU, rootU * (1.0 - u.w), rootU * u.w);
  const float3 position = tri.p0 * w.x + tri.p1 * w.y + tri.p2 * w.z;
  const float2 uv = tri.a0.uv * w.x + tri.a1.uv * w.y + tri.a2.uv * w.z;

  const float3 faceNormal = cross(tri.p1 - tri.p0, tri.p2 - tri.p0);
  const float triangleArea = 0.5 * length(faceNormal);
  const float3 toLight = position - origin;

  EmissiveSample result;
  result.distance = length(toLight);
  result.direction = toLight / result.distance;
  result.radiance = emissionAt(materials[tri.materialIndex], uv);
  // emitters are two-sided, like every surface here
  const float cosLight =
      abs(dot(normalize(faceNormal), result.direction));
  result.pdf = emissiveLightPdf(triangleEmission[tri.triangleIndex] *
                                    triangleArea,
                                triangleArea, result.distance, cosLight);
  return result;
}

// power heuristic (beta = 2)
float misWeight(float pdf, float otherPdf) {
  const float a = pdf * pdf;
  const float b = otherPdf * otherPdf;
  return a + b > 0.0 ? a / (a + b) : 0.0;
}

float3 tracePath(float3 origin, float3 direction, inout Rng rng) {
  float3 radiance = 0.0;
  float3 throughput = 1.0;
  const float3 toLight = -normalize(pc.lightDirection.xyz);
  const bool environmentSampling = hasEnvironmentSampling();
  const bool emissiveSampling = hasEmissiveSampling();
  // pdf of the BRDF sample that produced `direction`
  float lastBrdfPdf = 0.0;

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
      // camera rays see the environment directly; bounce rays share it with
      // the environment light sample taken at the previous hit
      const float weight =
          environmentSampling && bounce > 0u
              ? misWeight(lastBrdfPdf, environmentPdf(direction))
              : 1.0;
      radiance += throughput * sampleEnvironment(direction, 0.0) * weight;
      break;
    }

    const HitSurface hit = loadHit(
        query.CommittedInstanceID(), query.CommittedPrimitiveIndex(),
        query.CommittedTriangleBarycentrics(), direction);
    // camera rays see emitters directly; bounce rays share them with the
    // emissive light sample taken at the previous hit
    if (any(hit.emissive > 0.0)) {
      const float weight =
          emissiveSampling && bounce > 0u
              ? misWeight(lastBrdfPdf,
                          emissiveLightPdf(
                              hit.lightWeight, hit.triangleArea,
                              query.CommittedRayT(),
                              abs(dot(hit.geometricNormal, direction))))
              : 1.0;
      radiance += throughput * hit.emissive * weight;
    }
    if (bounce == pc.maxBounces) {
      break;
    }

    const float3 v = -direction;
    const float3 n = hit.normal;
    const float3 nextOrigin = offsetRayOrigin(hit.position, hit.geometricNormal);

    // directional light: a delta light, so only reachable by NEE
    if (pc.lightDirection.w > 0.0 && dot(n, toLight) > 0.0 &&
        dot(hit.geometricNormal, toLight) > 0.0 &&
        isVisible(nextOrigin, toLight, 1e30)) {
      radiance += throughput * evaluateBrdf(hit.material, n, v, toLight) *
                  kLightColor * pc.lightDirection.w;
    }

    // environment light sample, weighted against BRDF sampling finding the
    // same direction
    if (environmentSampling) {
      float environmentSamplePdf;
      const float3 l = sampleEnvironmentDirection(
          float4(nextFloat(rng), nextFloat(rng), nextFloat(rng),
                 nextFloat(rng)),
          environmentSamplePdf);
      if (environmentSamplePdf > 0.0 && dot(n, l) > 0.0 &&
          dot(hit.geometricNormal, l) > 0.0 && isVisible(nextOrigin, l, 1e30)) {
        const float weight = misWeight(
            environmentSamplePdf, brdfSamplePdf(hit.material, n, v, l));
        radiance += throughput * evaluateBrdf(hit.material, n, v, l) *
                    sampleEnvironment(l, 0.0) * (weight / environmentSamplePdf);
      }
    }

    // emissive light sample, weighted against BRDF sampling hitting the same
    // point
    if (emissiveSampling) {
      const EmissiveSample light = sampleEmissiveLight(
          nextOrigin, float4(nextFloat(rng), nextFloat(rng), nextFloat(rng),
                             nextFloat(rng)));
      const float3 l = light.direction;
      // stop just short of the light so it does not shadow itself
      if (light.pdf > 0.0 && dot(n, l) > 0.0 &&
          dot(hit.geometricNormal, l) > 0.0 &&
          isVisible(nextOrigin, l, light.distance * (1.0 - 1e-4))) {
        const float weight =
            misWeight(light.pdf, brdfSamplePdf(hit.material, n, v, l));
        radiance += throughput * evaluateBrdf(hit.material, n, v, l) *
                    light.radiance * (weight / light.pdf);
      }
    }

    // pick the specular or diffuse lobe; the pdf below is the mixture of both
    // so either choice is weighted correctly.
    const float alpha =
        max(hit.material.roughness * hit.material.roughness, 1e-3);
    const float specularProbability = specularLobeProbability(hit.material);
    const float2 u = float2(nextFloat(rng), nextFloat(rng));
    const float3 l = nextFloat(rng) < specularProbability
                         ? sampleGgxReflection(n, v, alpha, u)
                         : sampleCosineHemisphere(n, u);
    const float nDotL = dot(n, l);
    if (nDotL <= 0.0 || dot(hit.geometricNormal, l) <= 0.0) {
      break;
    }
    const float pdf = brdfSamplePdf(hit.material, n, v, l);
    if (pdf <= 0.0) {
      break;
    }
    lastBrdfPdf = pdf;
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
