#ifndef BRDF_HLSLI
#define BRDF_HLSLI

// glTF metallic-roughness BRDF: GGX specular with height-correlated Smith
// visibility and a Lambert diffuse lobe. Shared with the future path tracer.

static const float kPi = 3.14159265359;

struct SurfaceProperties {
  float3 baseColor;
  float metallic;
  // perceptual roughness, alpha = roughness^2
  float roughness;
};

float3 fresnelSchlick(float3 f0, float vDotH) {
  return f0 + (1.0 - f0) * pow(1.0 - vDotH, 5.0);
}

float distributionGgx(float nDotH, float alpha) {
  float alpha2 = alpha * alpha;
  float d = nDotH * nDotH * (alpha2 - 1.0) + 1.0;
  return alpha2 / (kPi * d * d);
}

float visibilitySmithGgxCorrelated(float nDotV, float nDotL, float alpha) {
  float alpha2 = alpha * alpha;
  float ggxV = nDotL * sqrt(nDotV * nDotV * (1.0 - alpha2) + alpha2);
  float ggxL = nDotV * sqrt(nDotL * nDotL * (1.0 - alpha2) + alpha2);
  return 0.5 / max(ggxV + ggxL, 1e-6);
}

// returns BRDF * cos(theta_l)
float3 evaluateBrdf(SurfaceProperties surface, float3 n, float3 v, float3 l) {
  float nDotL = saturate(dot(n, l));
  float nDotV = max(dot(n, v), 1e-4);
  if (nDotL <= 0.0) {
    return 0.0;
  }
  float3 h = normalize(v + l);
  float nDotH = saturate(dot(n, h));
  float vDotH = saturate(dot(v, h));

  // clamp so perfectly smooth surfaces still produce a highlight
  float alpha = max(surface.roughness * surface.roughness, 1e-3);
  float3 f0 = lerp(0.04, surface.baseColor, surface.metallic);
  float3 f = fresnelSchlick(f0, vDotH);

  float3 specular = f * distributionGgx(nDotH, alpha) *
                    visibilitySmithGgxCorrelated(nDotV, nDotL, alpha);
  float3 diffuse = (1.0 - f) * (1.0 - surface.metallic) * surface.baseColor / kPi;
  return (diffuse + specular) * nDotL;
}

#endif
