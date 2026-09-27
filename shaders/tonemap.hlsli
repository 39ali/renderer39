#ifndef TONEMAP_HLSLI
#define TONEMAP_HLSLI

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

#endif
