#include "brdf.hlsli"
#include "environment.hlsli"
#include "tonemap.hlsli"

static const float3 kLightColor = float3(1.0, 0.96, 0.9);

float4 sampleTexture(uint textureIndex, float2 uv, float4 fallback) {
  if (textureIndex == kInvalidTexture) {
    return fallback;
  }
  return textures[NonUniformResourceIndex(textureIndex)].Sample(materialSampler,
                                                                uv);
}

// Karis' analytic fit of the split-sum environment BRDF (no lookup table)
float3 environmentBrdfApprox(float3 f0, float roughness, float nDotV) {
  const float4 c0 = float4(-1.0, -0.0275, -0.572, 0.022);
  const float4 c1 = float4(1.0, 0.0425, 1.04, -0.04);
  float4 r = roughness * c0 + c1;
  float a004 = min(r.x * r.x, exp2(-9.28 * nDotV)) * r.x + r.y;
  float2 ab = float2(-1.04, 1.04) * a004 + r.zw;
  return f0 * ab.x + ab.y;
}

float4 main(VertexOutput input) : SV_Target0 {
  const Material material = materials[input.materialIndex];

  const float4 baseColor =
      material.baseColorFactor *
      sampleTexture(material.baseColorTexture, input.uv, 1.0);
  if (material.alphaMode != kAlphaModeOpaque &&
      baseColor.a < material.alphaCutoff) {
    discard;
  }

  // glTF packs roughness in G and metallic in B
  const float4 metallicRoughness =
      sampleTexture(material.metallicRoughnessTexture, input.uv, 1.0);
  SurfaceProperties surface;
  surface.baseColor = baseColor.rgb;
  surface.metallic = saturate(material.metallicFactor * metallicRoughness.b);
  surface.roughness = saturate(material.roughnessFactor * metallicRoughness.g);

  float3 normal = normalize(input.normal);
  if (material.normalTexture != kInvalidTexture) {
    float3 tangentNormal =
        sampleTexture(material.normalTexture, input.uv, 0.0).xyz * 2.0 - 1.0;
    tangentNormal.xy *= material.normalScale;
    const float3 tangent =
        normalize(input.tangent.xyz - normal * dot(normal, input.tangent.xyz));
    const float3 bitangent = cross(normal, tangent) * input.tangent.w;
    normal = normalize(tangent * tangentNormal.x + bitangent * tangentNormal.y +
                       normal * tangentNormal.z);
  }

  const float occlusion =
      1.0 +
      material.occlusionStrength *
          (sampleTexture(material.occlusionTexture, input.uv, 1.0).r - 1.0);
  const float3 emissive =
      material.emissiveFactor *
      sampleTexture(material.emissiveTexture, input.uv, 1.0).rgb;

  switch (pc.debugView) {
  case kDebugViewBaseColor:
    return float4(surface.baseColor, 1.0);
  case kDebugViewNormal:
    // the swapchain applies the sRGB curve, so undo it for raw values
    return float4(pow(normal * 0.5 + 0.5, 2.2), 1.0);
  case kDebugViewMetallic:
    return float4(pow(surface.metallic.xxx, 2.2), 1.0);
  case kDebugViewRoughness:
    return float4(pow(surface.roughness.xxx, 2.2), 1.0);
  case kDebugViewOcclusion:
    return float4(pow(occlusion.xxx, 2.2), 1.0);
  case kDebugViewEmissive:
    return float4(emissive, 1.0);
  case kDebugViewUv:
    return float4(pow(float3(frac(input.uv), 0.0), 2.2), 1.0);
  default:
    break;
  }

  const float3 v = normalize(pc.cameraPosition.xyz - input.worldPosition);
  const float3 l = -normalize(pc.lightDirection.xyz);
  float3 color =
      evaluateBrdf(surface, normal, v, l) * kLightColor * pc.lightDirection.w;

  // Cheap stand-in for image-based lighting: blurry mip lookups instead of
  // prefiltered maps, and no shadowing.
  const float blurMip = max(pc.environmentMaxMip - 2.0, 0.0);
  const float3 diffuseLight = sampleEnvironment(normal, blurMip);
  const float3 specularLight =
      sampleEnvironment(reflect(-v, normal), surface.roughness * blurMip);
  const float3 f0 = lerp(0.04, surface.baseColor, surface.metallic);
  const float3 specularWeight =
      environmentBrdfApprox(f0, surface.roughness, saturate(dot(normal, v)));
  color += (diffuseLight * surface.baseColor * (1.0 - surface.metallic) +
            specularLight * specularWeight) *
           occlusion;
  color += emissive;
  return float4(tonemapAces(color), 1.0);
}
