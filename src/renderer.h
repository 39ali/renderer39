#pragma once

#include "camera.h"
#include "object3d.h"
#include "resource.h"
#include "vk_context.h"

#include <filesystem>
#include <vector>

struct GLFWwindow;

struct MeshUploadInfo {
  uint32_t firstMeshlet = 0;
  uint32_t meshletCount = 0;
};

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
  void render(const glm::mat4 &viewProjection, const CameraCullData &cullData,
              float dt);
  void waitIdle();

private:
  void cleanup();
  void createDescriptorSetLayout();
  void createGraphicsPipeline();
  void createImgui();
  void createDescriptorPool();
  void createDescriptorSet();
  void updateInstanceDescriptorSet();
  void updateMeshDescriptorSet();
  void createSampler();
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
  void recordCommandBuffer(const FrameContext &frame,
                           const glm::mat4 &viewProjection,
                           const CameraCullData &cullData, float dt);
  void recordCounterReset(VkCommandBuffer commandBuffer);
  void recordCounterReadback(VkCommandBuffer commandBuffer,
                             VkBuffer counterReadbackBuffer);
  void recordRenderingCommands(VkCommandBuffer commandBuffer,
                               const FrameContext &frame,
                               const glm::mat4 &viewProjection,
                               const CameraCullData &cullData, float dt);
  void renderImgui(VkCommandBuffer commandBuffer, float dt);
  void cleanupImgui();

  VkContext _vkContext;
  Resources _resources;
  std::vector<Meshlet> _meshlets;
  std::vector<VertexPosition> _vertexPositions;
  std::vector<VertexAttributes> _vertexAttributes;
  // CPU only; kept for acceleration structure builds
  std::vector<uint32_t> _indices;
  std::vector<uint32_t> _meshletVertexRefs;
  std::vector<uint32_t> _meshletTriangles;
  std::vector<Material> _materials;
  std::vector<Texture> _textures;
  std::vector<MeshUploadInfo> _meshes;
  std::vector<InstanceData> _frameInstances;
  std::vector<MeshletInstance> _candidateMeshlets;

  VkDescriptorSetLayout _descriptorSetLayout = VK_NULL_HANDLE;
  VkPipelineLayout _pipelineLayout = VK_NULL_HANDLE;
  VkPipeline _graphicsPipeline = VK_NULL_HANDLE;
  VkDescriptorPool _descriptorPool = VK_NULL_HANDLE;
  VkDescriptorPool _imguiDescriptorPool = VK_NULL_HANDLE;
  VkDescriptorSet _descriptorSet = VK_NULL_HANDLE;
  Buffer<Meshlet> _meshletBuffer;
  Buffer<VertexPosition> _vertexPositionBuffer;
  Buffer<VertexAttributes> _vertexAttributeBuffer;
  Buffer<uint32_t> _meshletVertexRefBuffer;
  Buffer<uint32_t> _meshletTriangleBuffer;
  Buffer<Material> _materialBuffer;
  VkSampler _materialSampler = VK_NULL_HANDLE;
  Buffer<InstanceData> _instanceBuffer;
  RenderBucket _renderBucket;
  size_t _instanceBufferCapacity = 0;
  size_t _candidateMeshletCapacity = 0;
  RenderStats _stats;
  uint32_t _debugView = kDebugViewLit;
  float _lightAzimuth = 45.0f;
  float _lightElevation = 50.0f;
  float _lightIntensity = 3.0f;
};
