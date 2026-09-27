#pragma once

#include "shared_structs.h"

#include <cstdint>
#include <filesystem>
#include <vector>

using MeshId = uint32_t;

struct TextureData {
  uint32_t width = 0;
  uint32_t height = 0;
  // base color and emissive are sRGB; data textures are linear.
  bool srgb = false;
  // tightly packed RGBA8
  std::vector<uint8_t> rgba;
};

// equirectangular HDR environment
struct HdrImage {
  uint32_t width = 0;
  uint32_t height = 0;
  // tightly packed RGBA16F
  std::vector<uint16_t> rgbaHalf;
};

// one glTF file flattened into a single mesh; node transforms are baked in.
struct GltfModel {
  std::vector<VertexPosition> positions;
  std::vector<VertexAttributes> attributes;
  std::vector<uint32_t> meshletVertexRefs;
  std::vector<uint32_t> meshletTriangles;
  // materialIndex indexes materials
  std::vector<Meshlet> meshlets;
  // texture indices index textures
  std::vector<Material> materials;
  std::vector<TextureData> textures;
  glm::vec3 boundsCenter{0.0f};
  float boundsRadius = 0.0f;
};

class Resources {
public:
  Resources(std::filesystem::path baseDir);

  std::vector<char> readBinary(const std::filesystem::path &path) const;
  GltfModel loadGltfModel(const std::filesystem::path &path) const;
  HdrImage loadHdrImage(const std::filesystem::path &path) const;

private:
  std::filesystem::path _baseDir;
};
