#include "brdf.hlsli"
#include "scene_bindings.hlsli"

static const float3 kLightColor = float3(1.0, 0.96, 0.9);
// stand-in for indirect light until the path tracer exists
static const float3 kAmbientColor = float3(0.03, 0.035, 0.045);

float4 sampleTexture(uint textureIndex, float2 uv, float4 fallback) {
  if (textureIndex == kInvalidTexture) {
    return fallback;
  }
  return textures[NonUniformResourceIndex(textureIndex)].Sample(
      materialSampler, uv);
}

// Stephen Hill's fitted ACES approximation
float3 tonemapAces(float3 color) {
  const float3x3 inputMatrix = float3x3(0.59719, 0.35458, 0.04823,
                                        0.07600, 0.90834, 0.01566,
                                        0.02840, 0.13383, 0.83777);
  const float3x3 outputMatrix = float3x3(1.60475, -0.53108, -0.07367,
                                         -0.10208, 1.10813, -0.00605,
                                         -0.00327, -0.07276, 1.07602);
  color = mul(inputMatrix, color);
  float3 a = color * (color + 0.0245786) - 0.000090537;
  float3 b = color * (0.983729 * color + 0.4329510) + 0.238081;
  return saturate(mul(outputMatrix, a / b));
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
    normal = normalize(tangent * tangentNormal.x +
                       bitangent * tangentNormal.y + normal * tangentNormal.z);
  }

  const float occlusion =
      1.0 + material.occlusionStrength *
                (sampleTexture(material.occlusionTexture, input.uv, 1.0).r -
                 1.0);
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
  float3 color = evaluateBrdf(surface, normal, v, l) * kLightColor *
                 pc.lightDirection.w;
  color += kAmbientColor * surface.baseColor * occlusion;
  color += emissive;
  return float4(tonemapAces(color), 1.0);
}
