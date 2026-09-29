#pragma once

#include "camera.h"
#include "object3d.h"
#include "raytracing.h"
#include "resource.h"
#include "vk_context.h"

#include <filesystem>
#include <vector>

struct GLFWwindow;

struct MeshUploadInfo {
  uint32_t firstMeshlet = 0;
  uint32_t meshletCount = 0;
  // into _meshletTriangles; also the base of ray tracing primitive indices
  uint32_t firstTriangle = 0;
};

enum class RenderMode { Raster, PathTracing };

struct RenderBucket {
  Buffer<MeshletInstance> candidateMeshletBuffer;
  Buffer<RenderCounters> counterBuffer;
  Buffer<RenderCounters> counterReadbackBuffers[VkContext::kMaxFramesInFlight];
};

struct RenderStats {
  uint32_t visibleMeshlets = 0;
  uint32_t totalMeshlets = 0;
  uint32_t visibleTriangles = 0;
  uint64_t totalTriangles = 0;
};

class Renderer {
public:
  Renderer(GLFWwindow *window, std::filesystem::path appDir);
  ~Renderer();

  Renderer(const Renderer &) = delete;
  Renderer &operator=(const Renderer &) = delete;

  void init();
  MeshId loadModel(const GltfModel &model);
  void setObjects(const std::vector<Object3D> &objects);
  void setEnvironment(const HdrImage &image);
  void render(const glm::mat4 &viewProjection, const CameraCullData &cullData,
              float dt);
  void waitIdle();
  void setRenderMode(RenderMode mode) { _renderMode = mode; }
  void setEnvironmentSampling(bool enabled) { _environmentSampling = enabled; }
  // path tracing stops accumulating after this many samples (0: never)
  void setMaxSamples(uint32_t samples) { _maxSamples = samples; }
  // true once path tracing has reached its sample limit
  bool accumulationComplete() const {
    return _renderMode == RenderMode::PathTracing && _maxSamples > 0 &&
           _sampleIndex >= _maxSamples;
  }
  // writes the path tracer's linear radiance as a PFM image
  void saveAccumulation(const std::filesystem::path &path);

private:
  void cleanup();
  void createDescriptorSetLayout();
  void createGraphicsPipeline();
  void createPathTracingPipeline();
  void buildBlas(const MeshUploadInfo &mesh);
  DrawPushConstants makePushConstants(const glm::mat4 &viewProjection,
                                      const CameraCullData &cullData) const;
  void recordPathTracing(VkCommandBuffer commandBuffer);
  void createImgui();
  void createDescriptorPool();
  void createDescriptorSet();
  void updateInstanceDescriptorSet();
  void updateMeshDescriptorSet();
  void createSampler();
  void writeEnvironmentAliasDescriptor();
  void uploadTextures(const std::vector<TextureData> &textures);
  void updateRenderBucketDescriptorSet();
  void rebuildMeshBuffers();
  void uploadRenderBucket();
  void readCompletedCounters(size_t frameIndex);
  template <typename T>
  bool uploadHostBuffer(const std::vector<T> &source, VkBufferUsageFlags usage,
                        Buffer<T> &target, size_t &capacity);
  template <typename T>
  void uploadDeviceLocalBuffer(const std::vector<T> &source,
                               VkBufferUsageFlags usage, Buffer<T> &target);
  void recordCommandBuffer(const FrameContext &frame, float dt);
  void recordCounterReset(VkCommandBuffer commandBuffer);
  void recordCounterReadback(VkCommandBuffer commandBuffer,
                             VkBuffer counterReadbackBuffer);
  void recordRenderingCommands(VkCommandBuffer commandBuffer,
                               const FrameContext &frame, float dt);
  void renderImgui(VkCommandBuffer commandBuffer, float dt);
  void cleanupImgui();

  VkContext _vkContext;
  Resources _resources;
  std::vector<Meshlet> _meshlets;
  std::vector<VertexPosition> _vertexPositions;
  std::vector<VertexAttributes> _vertexAttributes;
  std::vector<uint32_t> _meshletVertexRefs;
  std::vector<uint32_t> _meshletTriangles;
  // meshlet of every triangle in _meshletTriangles
  std::vector<uint32_t> _triangleMeshlets;
  std::vector<Material> _materials;
  std::vector<Texture> _textures;
  std::vector<MeshUploadInfo> _meshes;
  std::vector<InstanceData> _frameInstances;
  std::vector<MeshletInstance> _candidateMeshlets;

  VkDescriptorSetLayout _descriptorSetLayout = VK_NULL_HANDLE;
  VkPipelineLayout _pipelineLayout = VK_NULL_HANDLE;
  VkPipeline _graphicsPipeline = VK_NULL_HANDLE;
  VkPipeline _skyPipeline = VK_NULL_HANDLE;
  VkPipeline _presentPipeline = VK_NULL_HANDLE;
  VkPipeline _pathTracingPipeline = VK_NULL_HANDLE;
  VkDescriptorPool _descriptorPool = VK_NULL_HANDLE;
  VkDescriptorPool _imguiDescriptorPool = VK_NULL_HANDLE;
  VkDescriptorSet _descriptorSet = VK_NULL_HANDLE;
  Buffer<Meshlet> _meshletBuffer;
  Buffer<VertexPosition> _vertexPositionBuffer;
  Buffer<VertexAttributes> _vertexAttributeBuffer;
  Buffer<uint32_t> _meshletVertexRefBuffer;
  Buffer<uint32_t> _meshletTriangleBuffer;
  Buffer<uint32_t> _triangleMeshletBuffer;
  Buffer<Material> _materialBuffer;
  Buffer<EnvironmentAliasEntry> _environmentAliasBuffer;
  uint32_t _environmentTableWidth = 0;
  uint32_t _environmentTableHeight = 0;
  bool _environmentSampling = true;
  uint32_t _maxSamples = 0;
  VkSampler _materialSampler = VK_NULL_HANDLE;
  Buffer<InstanceData> _instanceBuffer;
  RenderBucket _renderBucket;
  RayTracingScene _rayTracingScene{_vkContext};
  // path tracer radiance, averaged over _sampleIndex frames
  Texture _accumulationImage;
  RenderMode _renderMode = RenderMode::Raster;
  uint32_t _sampleIndex = 0;
  int _maxBounces = 4;
  // the state the accumulated samples were rendered with
  DrawPushConstants _accumulatedState{};
  DrawPushConstants _pushConstants{};
  size_t _instanceBufferCapacity = 0;
  size_t _candidateMeshletCapacity = 0;
  RenderStats _stats;
  uint32_t _debugView = kDebugViewLit;
  float _lightAzimuth = 45.0f;
  float _lightElevation = 50.0f;
  float _lightIntensity = 3.0f;
  uint32_t _environmentTexture = kInvalidTexture;
  float _environmentMaxMip = 0.0f;
  float _environmentIntensity = 1.0f;
};
