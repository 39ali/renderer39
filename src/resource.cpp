#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "resource.h"

#include <meshoptimizer.h>
#include <tiny_gltf.h>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/packing.hpp>

#include <cassert>
#include <fstream>
#include <iostream>
#include <map>

namespace {

// one glTF primitive after node transforms are applied
struct PrimitiveGeometry {
  std::vector<glm::vec3> positions;
  std::vector<glm::vec3> normals;
  std::vector<glm::vec4> tangents;
  std::vector<glm::vec2> uvs;
  std::vector<uint32_t> indices;
  uint32_t materialIndex = 0;
};

struct AccessorView {
  const uint8_t *data = nullptr;
  size_t stride = 0;
  size_t count = 0;
  int componentType = 0;
  int componentCount = 0;
  bool normalized = false;
};

uint32_t packCountsAndFlags(uint32_t vertexCount, uint32_t triangleCount,
                            uint32_t flags = 0) {
  return (vertexCount & 0xffu) | ((triangleCount & 0xffu) << 8u) |
         ((flags & 0xffffu) << 16u);
}

uint32_t packMeshletCone(const meshopt_Bounds &bounds) {
  return static_cast<uint8_t>(bounds.cone_axis_s8[0]) |
         (static_cast<uint32_t>(static_cast<uint8_t>(bounds.cone_axis_s8[1]))
          << 8u) |
         (static_cast<uint32_t>(static_cast<uint8_t>(bounds.cone_axis_s8[2]))
          << 16u) |
         (static_cast<uint32_t>(static_cast<uint8_t>(bounds.cone_cutoff_s8))
          << 24u);
}

uint32_t packTriangle(uint8_t a, uint8_t b, uint8_t c) {
  return static_cast<uint32_t>(a) | (static_cast<uint32_t>(b) << 10) |
         (static_cast<uint32_t>(c) << 20);
}

glm::vec2 octEncode(glm::vec3 n) {
  n /= std::abs(n.x) + std::abs(n.y) + std::abs(n.z);
  if (n.z < 0.0f) {
    const glm::vec2 signs{n.x >= 0.0f ? 1.0f : -1.0f,
                          n.y >= 0.0f ? 1.0f : -1.0f};
    return (1.0f - glm::abs(glm::vec2{n.y, n.x})) * signs;
  }
  return {n.x, n.y};
}

VertexAttributes packAttributes(const glm::vec3 &normal,
                                const glm::vec4 &tangent, const glm::vec2 &uv) {
  uint32_t packedTangent = glm::packSnorm2x16(octEncode(glm::vec3(tangent)));
  packedTangent = (packedTangent & ~1u) | (tangent.w < 0.0f ? 1u : 0u);
  return {
      .uv = uv,
      .normal = glm::packSnorm2x16(octEncode(normal)),
      .tangent = packedTangent,
  };
}

AccessorView accessorView(const tinygltf::Model &model, int accessorIndex) {
  const auto &accessor = model.accessors[accessorIndex];
  assert(accessor.bufferView >= 0 && "sparse glTF accessors are unsupported");
  const auto &view = model.bufferViews[accessor.bufferView];
  const auto &buffer = model.buffers[view.buffer];
  const int componentSize =
      tinygltf::GetComponentSizeInBytes(accessor.componentType);
  const int componentCount = tinygltf::GetNumComponentsInType(accessor.type);
  const int byteStride = accessor.ByteStride(view);
  return {
      .data = buffer.data.data() + view.byteOffset + accessor.byteOffset,
      .stride = byteStride > 0
                    ? static_cast<size_t>(byteStride)
                    : static_cast<size_t>(componentSize * componentCount),
      .count = accessor.count,
      .componentType = accessor.componentType,
      .componentCount = componentCount,
      .normalized = accessor.normalized,
  };
}

AccessorView findAttribute(const tinygltf::Model &model,
                           const tinygltf::Primitive &primitive,
                           const char *name) {
  const auto it = primitive.attributes.find(name);
  return it == primitive.attributes.end() ? AccessorView{}
                                          : accessorView(model, it->second);
}

float readComponent(const uint8_t *value, int componentType, bool normalized) {
  switch (componentType) {
  case TINYGLTF_COMPONENT_TYPE_FLOAT:
    return *reinterpret_cast<const float *>(value);
  case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: {
    const float v = *value;
    return normalized ? v / 255.0f : v;
  }
  case TINYGLTF_COMPONENT_TYPE_BYTE: {
    const float v = *reinterpret_cast<const int8_t *>(value);
    return normalized ? std::max(v / 127.0f, -1.0f) : v;
  }
  case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
    const float v = *reinterpret_cast<const uint16_t *>(value);
    return normalized ? v / 65535.0f : v;
  }
  case TINYGLTF_COMPONENT_TYPE_SHORT: {
    const float v = *reinterpret_cast<const int16_t *>(value);
    return normalized ? std::max(v / 32767.0f, -1.0f) : v;
  }
  case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
    return static_cast<float>(*reinterpret_cast<const uint32_t *>(value));
  default:
    assert(false && "unsupported glTF component type");
    return 0.0f;
  }
}

// reads element i as a vec4; missing components are 0 (w defaults to 1).
glm::vec4 readVec(const AccessorView &view, size_t i) {
  glm::vec4 result{0.0f, 0.0f, 0.0f, 1.0f};
  const uint8_t *element = view.data + i * view.stride;
  const int componentSize = tinygltf::GetComponentSizeInBytes(view.componentType);
  for (int c = 0; c < view.componentCount && c < 4; ++c) {
    result[c] = readComponent(element + c * componentSize, view.componentType,
                              view.normalized);
  }
  return result;
}

uint32_t readIndex(const AccessorView &view, size_t i) {
  const uint8_t *value = view.data + i * view.stride;
  switch (view.componentType) {
  case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
    return *value;
  case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
    return *reinterpret_cast<const uint16_t *>(value);
  case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
    return *reinterpret_cast<const uint32_t *>(value);
  default:
    assert(false && "unsupported glTF index component type");
    return 0;
  }
}

glm::mat4 nodeMatrix(const tinygltf::Node &node) {
  if (node.matrix.size() == 16) {
    return glm::mat4(glm::make_mat4x4(node.matrix.data()));
  }
  glm::mat4 matrix(1.0f);
  if (node.translation.size() == 3) {
    matrix = glm::translate(matrix,
                            glm::vec3(node.translation[0], node.translation[1],
                                      node.translation[2]));
  }
  if (node.rotation.size() == 4) {
    const glm::quat q(static_cast<float>(node.rotation[3]),
                      static_cast<float>(node.rotation[0]),
                      static_cast<float>(node.rotation[1]),
                      static_cast<float>(node.rotation[2]));
    matrix *= glm::mat4_cast(q);
  }
  if (node.scale.size() == 3) {
    matrix = glm::scale(matrix,
                        glm::vec3(node.scale[0], node.scale[1], node.scale[2]));
  }
  return matrix;
}

tinygltf::Model loadGltf(const std::filesystem::path &path) {
  tinygltf::TinyGLTF loader;
  tinygltf::Model model;
  std::string error;
  std::string warning;

  const std::string extension = path.extension().string();
  const bool loaded =
      extension == ".glb"
          ? loader.LoadBinaryFromFile(&model, &error, &warning, path.string())
          : loader.LoadASCIIFromFile(&model, &error, &warning, path.string());

  if (!warning.empty()) {
    std::cerr << "tinygltf warning: " << warning << '\n';
  }
  if (!loaded) {
    std::cerr << "tinygltf error: " << error << '\n';
  }
  assert(loaded && "failed to load glTF");

  return model;
}

void generateNormals(PrimitiveGeometry &primitive) {
  primitive.normals.assign(primitive.positions.size(), glm::vec3(0.0f));
  for (size_t i = 0; i + 2 < primitive.indices.size(); i += 3) {
    const uint32_t a = primitive.indices[i];
    const uint32_t b = primitive.indices[i + 1];
    const uint32_t c = primitive.indices[i + 2];
    // area weighted
    const glm::vec3 faceNormal =
        glm::cross(primitive.positions[b] - primitive.positions[a],
                   primitive.positions[c] - primitive.positions[a]);
    primitive.normals[a] += faceNormal;
    primitive.normals[b] += faceNormal;
    primitive.normals[c] += faceNormal;
  }
  for (glm::vec3 &normal : primitive.normals) {
    const float length = glm::length(normal);
    normal = length > 0.0f ? normal / length : glm::vec3(0.0f, 0.0f, 1.0f);
  }
}

glm::vec3 anyPerpendicular(const glm::vec3 &n) {
  const glm::vec3 axis = std::abs(n.x) < 0.9f ? glm::vec3(1.0f, 0.0f, 0.0f)
                                              : glm::vec3(0.0f, 1.0f, 0.0f);
  return glm::normalize(glm::cross(axis, n));
}

// per-vertex tangents accumulated from triangle UV gradients. Not MikkTSpace,
// so baked normal maps may show small seams; assets that ship tangents use
// those instead.
void generateTangents(PrimitiveGeometry &primitive) {
  const size_t vertexCount = primitive.positions.size();
  std::vector<glm::vec3> tangents(vertexCount, glm::vec3(0.0f));
  std::vector<glm::vec3> bitangents(vertexCount, glm::vec3(0.0f));

  if (!primitive.uvs.empty()) {
    for (size_t i = 0; i + 2 < primitive.indices.size(); i += 3) {
      const uint32_t v[3] = {primitive.indices[i], primitive.indices[i + 1],
                             primitive.indices[i + 2]};
      const glm::vec3 e1 =
          primitive.positions[v[1]] - primitive.positions[v[0]];
      const glm::vec3 e2 =
          primitive.positions[v[2]] - primitive.positions[v[0]];
      const glm::vec2 d1 = primitive.uvs[v[1]] - primitive.uvs[v[0]];
      const glm::vec2 d2 = primitive.uvs[v[2]] - primitive.uvs[v[0]];
      const float determinant = d1.x * d2.y - d2.x * d1.y;
      if (std::abs(determinant) < 1e-12f) {
        continue;
      }
      const float r = 1.0f / determinant;
      const glm::vec3 tangent = (e1 * d2.y - e2 * d1.y) * r;
      const glm::vec3 bitangent = (e2 * d1.x - e1 * d2.x) * r;
      for (const uint32_t vertex : v) {
        tangents[vertex] += tangent;
        bitangents[vertex] += bitangent;
      }
    }
  }

  primitive.tangents.resize(vertexCount);
  for (size_t i = 0; i < vertexCount; ++i) {
    const glm::vec3 &n = primitive.normals[i];
    glm::vec3 t = tangents[i] - n * glm::dot(n, tangents[i]);
    const float length = glm::length(t);
    t = length > 1e-8f ? t / length : anyPerpendicular(n);
    const float sign =
        glm::dot(glm::cross(n, t), bitangents[i]) < 0.0f ? -1.0f : 1.0f;
    primitive.tangents[i] = glm::vec4(t, sign);
  }
}

PrimitiveGeometry loadPrimitive(const tinygltf::Model &model,
                                const tinygltf::Primitive &primitive,
                                const glm::mat4 &transform,
                                uint32_t materialIndex) {
  PrimitiveGeometry geometry{.materialIndex = materialIndex};

  const AccessorView positions = findAttribute(model, primitive, "POSITION");
  const AccessorView normals = findAttribute(model, primitive, "NORMAL");
  const AccessorView tangents = findAttribute(model, primitive, "TANGENT");
  const AccessorView uvs = findAttribute(model, primitive, "TEXCOORD_0");

  const glm::mat3 linear(transform);
  const glm::mat3 normalMatrix = glm::transpose(glm::inverse(linear));

  geometry.positions.reserve(positions.count);
  for (size_t i = 0; i < positions.count; ++i) {
    geometry.positions.push_back(
        glm::vec3(transform * glm::vec4(glm::vec3(readVec(positions, i)), 1.0f)));
  }
  if (normals.data) {
    for (size_t i = 0; i < normals.count; ++i) {
      geometry.normals.push_back(
          glm::normalize(normalMatrix * glm::vec3(readVec(normals, i))));
    }
  }
  if (tangents.data) {
    for (size_t i = 0; i < tangents.count; ++i) {
      const glm::vec4 tangent = readVec(tangents, i);
      geometry.tangents.push_back(glm::vec4(
          glm::normalize(linear * glm::vec3(tangent)), tangent.w));
    }
  }
  if (uvs.data) {
    for (size_t i = 0; i < uvs.count; ++i) {
      geometry.uvs.push_back(glm::vec2(readVec(uvs, i)));
    }
  }

  if (primitive.indices >= 0) {
    const AccessorView indices = accessorView(model, primitive.indices);
    geometry.indices.reserve(indices.count);
    for (size_t i = 0; i < indices.count; ++i) {
      geometry.indices.push_back(readIndex(indices, i));
    }
  } else {
    for (size_t i = 0; i < positions.count; ++i) {
      geometry.indices.push_back(static_cast<uint32_t>(i));
    }
  }

  // mirrored transforms flip the winding
  if (glm::determinant(linear) < 0.0f) {
    for (size_t i = 0; i + 2 < geometry.indices.size(); i += 3) {
      std::swap(geometry.indices[i + 1], geometry.indices[i + 2]);
    }
    for (glm::vec4 &tangent : geometry.tangents) {
      tangent.w = -tangent.w;
    }
  }

  if (geometry.normals.empty()) {
    generateNormals(geometry);
  }
  if (geometry.tangents.empty()) {
    generateTangents(geometry);
  }
  return geometry;
}

class GltfConverter {
public:
  explicit GltfConverter(const tinygltf::Model &model) : _model(model) {}

  GltfModel convert() {
    for (const auto &material : _model.materials) {
      _result.materials.push_back(convertMaterial(material));
    }

    const int sceneIndex = _model.defaultScene >= 0 ? _model.defaultScene : 0;
    for (const int nodeIndex : _model.scenes[sceneIndex].nodes) {
      loadNode(_model.nodes[nodeIndex], glm::mat4(1.0f));
    }
    assert(!_result.meshlets.empty() && "glTF contains no triangle geometry");

    computeBounds();
    return std::move(_result);
  }

private:
  void loadNode(const tinygltf::Node &node, const glm::mat4 &parentTransform) {
    const glm::mat4 transform = parentTransform * nodeMatrix(node);

    if (node.mesh >= 0) {
      for (const auto &primitive : _model.meshes[node.mesh].primitives) {
        if (primitive.mode != TINYGLTF_MODE_TRIANGLES ||
            primitive.attributes.count("POSITION") == 0) {
          continue;
        }
        appendPrimitive(loadPrimitive(_model, primitive, transform,
                                      materialIndex(primitive.material)));
      }
    }

    for (const int childIndex : node.children) {
      loadNode(_model.nodes[childIndex], transform);
    }
  }

  uint32_t materialIndex(int gltfMaterial) {
    if (gltfMaterial >= 0) {
      return static_cast<uint32_t>(gltfMaterial);
    }
    if (_defaultMaterial == kInvalidTexture) {
      _defaultMaterial = static_cast<uint32_t>(_result.materials.size());
      _result.materials.push_back(convertMaterial(tinygltf::Material{}));
    }
    return _defaultMaterial;
  }

  void appendPrimitive(const PrimitiveGeometry &primitive) {
    const uint32_t vertexBase =
        static_cast<uint32_t>(_result.positions.size());
    for (size_t i = 0; i < primitive.positions.size(); ++i) {
      const glm::vec3 &p = primitive.positions[i];
      _result.positions.push_back({p.x, p.y, p.z});
      _result.attributes.push_back(packAttributes(
          primitive.normals[i], primitive.tangents[i],
          primitive.uvs.empty() ? glm::vec2(0.0f) : primitive.uvs[i]));
    }
    buildMeshlets(primitive, vertexBase);
  }

  void buildMeshlets(const PrimitiveGeometry &primitive, uint32_t vertexBase) {
    constexpr float kConeWeight = 0.0f;
    const size_t maxMeshlets = meshopt_buildMeshletsBound(
        primitive.indices.size(), kMaxMeshletVertices, kMaxMeshletTriangles);

    std::vector<meshopt_Meshlet> meshlets(maxMeshlets);
    std::vector<unsigned int> meshletVertices(maxMeshlets *
                                              kMaxMeshletVertices);
    std::vector<unsigned char> meshletTriangles(maxMeshlets *
                                                kMaxMeshletTriangles * 3);

    const auto *positions =
        reinterpret_cast<const float *>(primitive.positions.data());
    const size_t meshletCount = meshopt_buildMeshlets(
        meshlets.data(), meshletVertices.data(), meshletTriangles.data(),
        primitive.indices.data(), primitive.indices.size(), positions,
        primitive.positions.size(), sizeof(glm::vec3), kMaxMeshletVertices,
        kMaxMeshletTriangles, kConeWeight);

    for (size_t i = 0; i < meshletCount; ++i) {
      const meshopt_Meshlet &meshlet = meshlets[i];
      const meshopt_Bounds bounds = meshopt_computeMeshletBounds(
          &meshletVertices[meshlet.vertex_offset],
          &meshletTriangles[meshlet.triangle_offset], meshlet.triangle_count,
          positions, primitive.positions.size(), sizeof(glm::vec3));

      _result.meshlets.push_back({
          .packedSphere0 =
              glm::packHalf2x16({bounds.center[0], bounds.center[1]}),
          .packedSphere1 =
              glm::packHalf2x16({bounds.center[2], bounds.radius}),
          .vertexOffset =
              static_cast<uint32_t>(_result.meshletVertexRefs.size()),
          .triangleOffset =
              static_cast<uint32_t>(_result.meshletTriangles.size()),
          .packedCountsAndFlags = packCountsAndFlags(meshlet.vertex_count,
                                                     meshlet.triangle_count),
          .boundingCone = packMeshletCone(bounds),
          .materialIndex = primitive.materialIndex,
      });

      for (uint32_t v = 0; v < meshlet.vertex_count; ++v) {
        _result.meshletVertexRefs.push_back(
            vertexBase + meshletVertices[meshlet.vertex_offset + v]);
      }
      for (uint32_t t = 0; t < meshlet.triangle_count; ++t) {
        const size_t offset = meshlet.triangle_offset + t * 3;
        _result.meshletTriangles.push_back(packTriangle(
            meshletTriangles[offset], meshletTriangles[offset + 1],
            meshletTriangles[offset + 2]));
      }
    }
  }

  Material convertMaterial(const tinygltf::Material &material) {
    const auto &pbr = material.pbrMetallicRoughness;
    Material result{
        .baseColorFactor = glm::vec4(
            static_cast<float>(pbr.baseColorFactor[0]),
            static_cast<float>(pbr.baseColorFactor[1]),
            static_cast<float>(pbr.baseColorFactor[2]),
            static_cast<float>(pbr.baseColorFactor[3])),
        .emissiveFactor =
            glm::vec3(static_cast<float>(material.emissiveFactor[0]),
                      static_cast<float>(material.emissiveFactor[1]),
                      static_cast<float>(material.emissiveFactor[2])),
        .metallicFactor = static_cast<float>(pbr.metallicFactor),
        .roughnessFactor = static_cast<float>(pbr.roughnessFactor),
        .normalScale = static_cast<float>(material.normalTexture.scale),
        .occlusionStrength =
            static_cast<float>(material.occlusionTexture.strength),
        .alphaCutoff = static_cast<float>(material.alphaCutoff),
        .baseColorTexture = texture(pbr.baseColorTexture.index, true),
        .metallicRoughnessTexture =
            texture(pbr.metallicRoughnessTexture.index, false),
        .normalTexture = texture(material.normalTexture.index, false),
        .occlusionTexture = texture(material.occlusionTexture.index, false),
        .emissiveTexture = texture(material.emissiveTexture.index, true),
        .alphaMode = material.alphaMode == "MASK"    ? kAlphaModeMask
                     : material.alphaMode == "BLEND" ? kAlphaModeBlend
                                                     : kAlphaModeOpaque,
    };

    if (const auto it =
            material.extensions.find("KHR_materials_emissive_strength");
        it != material.extensions.end() && it->second.Has("emissiveStrength")) {
      result.emissiveFactor *= static_cast<float>(
          it->second.Get("emissiveStrength").GetNumberAsDouble());
    }
    return result;
  }

  // returns the index into _result.textures, decoding each (image, color
  // space) pair once.
  uint32_t texture(int gltfTexture, bool srgb) {
    if (gltfTexture < 0) {
      return kInvalidTexture;
    }
    const int imageIndex = _model.textures[gltfTexture].source;
    if (imageIndex < 0) {
      return kInvalidTexture;
    }

    const auto key = std::make_pair(imageIndex, srgb);
    if (const auto it = _textureIndices.find(key); it != _textureIndices.end()) {
      return it->second;
    }

    const tinygltf::Image &image = _model.images[imageIndex];
    if (image.image.empty() || image.width <= 0 || image.height <= 0) {
      std::cerr << "glTF image " << imageIndex << " could not be decoded\n";
      return kInvalidTexture;
    }

    TextureData data{
        .width = static_cast<uint32_t>(image.width),
        .height = static_cast<uint32_t>(image.height),
        .srgb = srgb,
    };
    const size_t pixelCount = static_cast<size_t>(data.width) * data.height;
    const size_t bytesPerComponent = image.bits == 16 ? 2 : 1;
    data.rgba.resize(pixelCount * 4);
    for (size_t pixel = 0; pixel < pixelCount; ++pixel) {
      uint8_t rgba[4] = {0, 0, 0, 255};
      for (int c = 0; c < image.component && c < 4; ++c) {
        // 16-bit images keep their high byte (little endian)
        const size_t byte =
            (pixel * image.component + c) * bytesPerComponent +
            (bytesPerComponent - 1);
        rgba[c] = image.image[byte];
      }
      if (image.component == 1) {
        rgba[1] = rgba[2] = rgba[0];
      }
      std::memcpy(&data.rgba[pixel * 4], rgba, 4);
    }

    const uint32_t index = static_cast<uint32_t>(_result.textures.size());
    _result.textures.push_back(std::move(data));
    _textureIndices.emplace(key, index);
    return index;
  }

  void computeBounds() {
    glm::vec3 minimum(std::numeric_limits<float>::max());
    glm::vec3 maximum(-std::numeric_limits<float>::max());
    for (const VertexPosition &p : _result.positions) {
      minimum = glm::min(minimum, glm::vec3(p.x, p.y, p.z));
      maximum = glm::max(maximum, glm::vec3(p.x, p.y, p.z));
    }
    _result.boundsCenter = (minimum + maximum) * 0.5f;
    _result.boundsRadius = glm::length(maximum - minimum) * 0.5f;
  }

  const tinygltf::Model &_model;
  GltfModel _result;
  std::map<std::pair<int, bool>, uint32_t> _textureIndices;
  uint32_t _defaultMaterial = kInvalidTexture;
};

} // namespace

Resources::Resources(std::filesystem::path baseDir)
    : _baseDir(std::move(baseDir)) {}

std::vector<char>
Resources::readBinary(const std::filesystem::path &path) const {
  const auto resolvedPath = _baseDir / path;
  std::ifstream file(resolvedPath, std::ios::ate | std::ios::binary);
  if (!file) {
    std::cerr << "failed to open file: " << resolvedPath << '\n';
    assert(false && "failed to open file");
    return {};
  }

  const auto size = static_cast<size_t>(file.tellg());
  std::vector<char> buffer(size);
  file.seekg(0);
  file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
  return buffer;
}

HdrImage Resources::loadHdrImage(const std::filesystem::path &path) const {
  const std::string resolvedPath = (_baseDir / path).string();
  int width = 0;
  int height = 0;
  int channels = 0;
  float *pixels = stbi_loadf(resolvedPath.c_str(), &width, &height, &channels,
                             STBI_rgb_alpha);
  if (!pixels) {
    std::cerr << "failed to load HDR " << resolvedPath << ": "
              << stbi_failure_reason() << '\n';
    assert(false && "failed to load HDR image");
    return {};
  }

  HdrImage image{
      .width = static_cast<uint32_t>(width),
      .height = static_cast<uint32_t>(height),
  };
  const size_t valueCount = static_cast<size_t>(width) * height * 4;
  image.rgbaHalf.resize(valueCount);
  for (size_t i = 0; i < valueCount; i += 2) {
    const uint32_t packed = glm::packHalf2x16({pixels[i], pixels[i + 1]});
    image.rgbaHalf[i] = static_cast<uint16_t>(packed);
    image.rgbaHalf[i + 1] = static_cast<uint16_t>(packed >> 16);
  }
  stbi_image_free(pixels);
  return image;
}

GltfModel Resources::loadGltfModel(const std::filesystem::path &path) const {
  const tinygltf::Model model = loadGltf(_baseDir / path);
  return GltfConverter(model).convert();
}
