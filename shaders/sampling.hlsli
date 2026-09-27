#ifndef SAMPLING_HLSLI
#define SAMPLING_HLSLI

#include "brdf.hlsli"

// PCG hash (Jarzynski & Olano, "Hash Functions for GPU Rendering")
uint pcgHash(uint value) {
  uint state = value * 747796405u + 2891336453u;
  uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

struct Rng {
  uint state;
};

Rng createRng(uint2 pixel, uint sampleIndex) {
  Rng rng;
  rng.state = pcgHash(pixel.x ^ pcgHash(pixel.y ^ pcgHash(sampleIndex)));
  return rng;
}

// uniform in [0, 1)
float nextFloat(inout Rng rng) {
  rng.state = pcgHash(rng.state);
  return float(rng.state >> 8) * (1.0 / 16777216.0);
}

// orthonormal basis around n (Duff et al. 2017)
void buildBasis(float3 n, out float3 t, out float3 b) {
  float s = n.z >= 0.0 ? 1.0 : -1.0;
  float a = -1.0 / (s + n.z);
  float c = n.x * n.y * a;
  t = float3(1.0 + s * n.x * n.x * a, s * c, -s * n.x);
  b = float3(c, s + n.y * n.y * a, -n.y);
}

float3 toWorld(float3 local, float3 n) {
  float3 t, b;
  buildBasis(n, t, b);
  return local.x * t + local.y * b + local.z * n;
}

float3 toLocal(float3 world, float3 n) {
  float3 t, b;
  buildBasis(n, t, b);
  return float3(dot(world, t), dot(world, b), dot(world, n));
}

float3 sampleCosineHemisphere(float3 n, float2 u) {
  float r = sqrt(u.x);
  float phi = 2.0 * kPi * u.y;
  return toWorld(float3(r * cos(phi), r * sin(phi), sqrt(max(1.0 - u.x, 0.0))),
                 n);
}

float smithG1(float nDotV, float alpha) {
  float alpha2 = alpha * alpha;
  return 2.0 * nDotV / (nDotV + sqrt(alpha2 + (1.0 - alpha2) * nDotV * nDotV));
}

// GGX visible-normal sampling (Heitz 2018); returns the reflected direction
float3 sampleGgxReflection(float3 n, float3 v, float alpha, float2 u) {
  float3 vLocal = toLocal(v, n);
  float3 vh = normalize(float3(alpha * vLocal.x, alpha * vLocal.y, vLocal.z));
  float lengthSquared = vh.x * vh.x + vh.y * vh.y;
  float3 t1 = lengthSquared > 0.0 ? float3(-vh.y, vh.x, 0.0) * rsqrt(lengthSquared)
                                  : float3(1.0, 0.0, 0.0);
  float3 t2 = cross(vh, t1);
  float r = sqrt(u.x);
  float phi = 2.0 * kPi * u.y;
  float p1 = r * cos(phi);
  float p2 = r * sin(phi);
  float s = 0.5 * (1.0 + vh.z);
  p2 = (1.0 - s) * sqrt(1.0 - p1 * p1) + s * p2;
  float3 nh = p1 * t1 + p2 * t2 + sqrt(max(0.0, 1.0 - p1 * p1 - p2 * p2)) * vh;
  float3 h = normalize(float3(alpha * nh.x, alpha * nh.y, max(0.0, nh.z)));
  return reflect(-v, toWorld(h, n));
}

// pdf of sampleGgxReflection, in solid angle of l
float ggxReflectionPdf(float3 n, float3 v, float3 l, float alpha) {
  float3 h = normalize(v + l);
  float nDotV = max(dot(n, v), 1e-4);
  return smithG1(nDotV, alpha) * distributionGgx(saturate(dot(n, h)), alpha) /
         (4.0 * nDotV);
}

#endif
