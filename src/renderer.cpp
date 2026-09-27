#define NOMINMAX
#include "renderer.h"

#include "math_helper.h"

#include <array>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include <string>

namespace {
constexpr uint32_t kBindingCount = 7;
constexpr VkShaderStageFlags kDrawStages =
    VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT;

uint32_t packColor(const glm::vec4 &color) {
  const auto packChannel = [](float value) {
    return static_cast<uint32_t>(clamp(value, 0.0f, 1.0f) * 255.0f + 0.5f);
  };
  return packChannel(color.r) | (packChannel(color.g) << 8) |
         (packChannel(color.b) << 16) | (packChannel(color.a) << 24);
}

std::string formatNumber(uint64_t value) {
  std::string text = std::to_string(value);
  for (int insertPosition = static_cast<int>(text.size()) - 3;
       insertPosition > 0; insertPosition -= 3) {
    text.insert(static_cast<size_t>(insertPosition), ",");
  }
  return text;
}

InstanceData makeInstanceData(const Object3D &object) {
  return {
      .translationScaleX = {object.position.x, object.position.y,
                            object.position.z, object.scale.x},
      .rotation = {object.rotation.x, object.rotation.y, object.rotation.z,
                   object.rotation.w},
      .scaleYZ = {object.scale.y, object.scale.z},
      .packedColor = packColor(object.color),
  };
}

} // namespace

Renderer::Renderer(GLFWwindow *window, std::filesystem::path appDir)
    : _vkContext(window), _resources(std::move(appDir)) {}

Renderer::~Renderer() { cleanup(); }

void Renderer::init() {
  _vkContext.init();
  createDescriptorSetLayout();
  createGraphicsPipeline();
  createDescriptorPool();
  createDescriptorSet();
  for (Buffer<RenderCounters> &buffer : _renderBucket.counterReadbackBuffers) {
    _vkContext.createBuffer(sizeof(RenderCounters),
                            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                            buffer);
  }
  createImgui();
}

void Renderer::waitIdle() { _vkContext.waitIdle(); }

void Renderer::cleanup() {
  if (_vkContext.device() != VK_NULL_HANDLE) {
    cleanupImgui();
    for (Buffer<RenderCounters> &buffer :
         _renderBucket.counterReadbackBuffers) {
      _vkContext.destroyBuffer(buffer);
    }
    _vkContext.destroyBuffer(_renderBucket.counterBuffer);
    _vkContext.destroyBuffer(_renderBucket.candidateMeshletBuffer);
    _vkContext.destroyBuffer(_instanceBuffer);
    _vkContext.destroyBuffer(_clusterTriangleBuffer);
    _vkContext.destroyBuffer(_meshletVertexRefBuffer);
    _vkContext.destroyBuffer(_clusterVertexBuffer);
    _vkContext.destroyBuffer(_meshletBuffer);
    vkDestroyPipeline(_vkContext.device(), _graphicsPipeline, nullptr);
    vkDestroyPipelineLayout(_vkContext.device(), _pipelineLayout, nullptr);
    vkDestroyDescriptorPool(_vkContext.device(), _descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(_vkContext.device(), _descriptorSetLayout,
                                 nullptr);
    _vkContext.cleanup();
  }
}

void Renderer::createDescriptorSetLayout() {
  std::array<VkDescriptorSetLayoutBinding, kBindingCount> bindings{};
  for (uint32_t binding = 0; binding < bindings.size(); ++binding) {
    bindings[binding] = {
        .binding = binding,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags = kDrawStages,
    };
  }

  VkDescriptorSetLayoutCreateInfo layoutInfo{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = static_cast<uint32_t>(bindings.size()),
      .pBindings = bindings.data(),
  };

  const VkResult createDescriptorSetLayoutResult = vkCreateDescriptorSetLayout(
      _vkContext.device(), &layoutInfo, nullptr, &_descriptorSetLayout);
  assert(createDescriptorSetLayoutResult == VK_SUCCESS &&
         "failed to create descriptor set layout");
}

void Renderer::createGraphicsPipeline() {
  const VkShaderModule taskShaderModule = _vkContext.createShaderModule(
      _resources.readBinary("shaders/meshlet.task.spv"));
  const VkShaderModule meshShaderModule = _vkContext.createShaderModule(
      _resources.readBinary("shaders/meshlet.mesh.spv"));
  const VkShaderModule fragShaderModule = _vkContext.createShaderModule(
      _resources.readBinary("shaders/meshlet.frag.spv"));

  const std::array<VkPipelineShaderStageCreateInfo, 3> shaderStages{{
      {
          .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_TASK_BIT_EXT,
          .module = taskShaderModule,
          .pName = "main",
      },
      {
          .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_MESH_BIT_EXT,
          .module = meshShaderModule,
          .pName = "main",
      },
      {
          .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
          .module = fragShaderModule,
          .pName = "main",
      },
  }};

  const RenderTargetInfo renderTarget = _vkContext.renderTargetInfo();
  VkViewport viewport{
      .width = static_cast<float>(renderTarget.extent.width),
      .height = static_cast<float>(renderTarget.extent.height),
      .maxDepth = 1.0f,
  };

  VkRect2D scissor{
      .offset = {0, 0},
      .extent = renderTarget.extent,
  };
  VkPipelineViewportStateCreateInfo viewportState{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1,
      .pViewports = &viewport,
      .scissorCount = 1,
      .pScissors = &scissor,
  };

  VkPipelineRasterizationStateCreateInfo rasterizer{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL,
      .cullMode = VK_CULL_MODE_BACK_BIT,
      .frontFace = VK_FRONT_FACE_CLOCKWISE,
      .lineWidth = 1.0f,
  };

  VkPipelineMultisampleStateCreateInfo multisampling{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
  };

  VkPipelineDepthStencilStateCreateInfo depthStencil{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .depthTestEnable = VK_TRUE,
      .depthWriteEnable = VK_TRUE,
      .depthCompareOp = VK_COMPARE_OP_LESS,
      .depthBoundsTestEnable = VK_FALSE,
      .stencilTestEnable = VK_FALSE,
  };

  VkPipelineColorBlendAttachmentState colorBlendAttachment{
      .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
  };

  VkPipelineColorBlendStateCreateInfo colorBlending{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1,
      .pAttachments = &colorBlendAttachment,
  };

  VkPushConstantRange pushConstantRange{
      .stageFlags = kDrawStages,
      .offset = 0,
      .size = sizeof(DrawPushConstants),
  };

  VkPipelineLayoutCreateInfo pipelineLayoutInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &_descriptorSetLayout,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &pushConstantRange,
  };
  const VkResult createPipelineLayoutResult = vkCreatePipelineLayout(
      _vkContext.device(), &pipelineLayoutInfo, nullptr, &_pipelineLayout);
  assert(createPipelineLayoutResult == VK_SUCCESS &&
         "failed to create pipeline layout");

  VkPipelineRenderingCreateInfo renderingInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .colorAttachmentCount = 1,
      .pColorAttachmentFormats = &renderTarget.colorFormat,
      .depthAttachmentFormat = renderTarget.depthFormat,
  };

  VkGraphicsPipelineCreateInfo pipelineInfo{
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .pNext = &renderingInfo,
      .stageCount = static_cast<uint32_t>(shaderStages.size()),
      .pStages = shaderStages.data(),
      .pViewportState = &viewportState,
      .pRasterizationState = &rasterizer,
      .pMultisampleState = &multisampling,
      .pDepthStencilState = &depthStencil,
      .pColorBlendState = &colorBlending,
      .layout = _pipelineLayout,
      .renderPass = VK_NULL_HANDLE,
      .subpass = 0,
  };

  const VkResult createGraphicsPipelineResult =
      vkCreateGraphicsPipelines(_vkContext.device(), VK_NULL_HANDLE, 1,
                                &pipelineInfo, nullptr, &_graphicsPipeline);
  assert(createGraphicsPipelineResult == VK_SUCCESS &&
         "failed to create graphics pipeline");

  vkDestroyShaderModule(_vkContext.device(), fragShaderModule, nullptr);
  vkDestroyShaderModule(_vkContext.device(), meshShaderModule, nullptr);
  vkDestroyShaderModule(_vkContext.device(), taskShaderModule, nullptr);
}

void Renderer::createImgui() {
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGui::GetIO().IniFilename = nullptr;
  ImGui::StyleColorsDark();

  std::array<VkDescriptorPoolSize, 1> poolSizes{{
      {
          .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          .descriptorCount = 16,
      },
  }};

  VkDescriptorPoolCreateInfo poolInfo{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
      .maxSets = 16,
      .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
      .pPoolSizes = poolSizes.data(),
  };
  const VkResult createPoolResult = vkCreateDescriptorPool(
      _vkContext.device(), &poolInfo, nullptr, &_imguiDescriptorPool);
  assert(createPoolResult == VK_SUCCESS &&
         "failed to create ImGui descriptor pool");

  const RenderTargetInfo renderTarget = _vkContext.renderTargetInfo();
  VkPipelineRenderingCreateInfo renderingInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .colorAttachmentCount = 1,
      .pColorAttachmentFormats = &renderTarget.colorFormat,
      .depthAttachmentFormat = renderTarget.depthFormat,
  };

  ImGui_ImplGlfw_InitForVulkan(_vkContext._window, true);
  ImGui_ImplVulkan_InitInfo initInfo{
      .ApiVersion = VK_API_VERSION_1_3,
      .Instance = _vkContext.instance(),
      .PhysicalDevice = _vkContext.physicalDevice(),
      .Device = _vkContext.device(),
      .QueueFamily = _vkContext.graphicsQueueFamily(),
      .Queue = _vkContext.graphicsQueue(),
      .DescriptorPool = _imguiDescriptorPool,
      .MinImageCount = _vkContext.swapChainImageCount(),
      .ImageCount = _vkContext.swapChainImageCount(),
      .MSAASamples = VK_SAMPLE_COUNT_1_BIT,
      .UseDynamicRendering = true,
      .PipelineRenderingCreateInfo = renderingInfo,
  };
  ImGui_ImplVulkan_Init(&initInfo);
}

void Renderer::createDescriptorPool() {
  VkDescriptorPoolSize poolSize{
      .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .descriptorCount = kBindingCount,
  };

  VkDescriptorPoolCreateInfo poolInfo{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1,
      .poolSizeCount = 1,
      .pPoolSizes = &poolSize,
  };

  const VkResult createDescriptorPoolResult = vkCreateDescriptorPool(
      _vkContext.device(), &poolInfo, nullptr, &_descriptorPool);
  assert(createDescriptorPoolResult == VK_SUCCESS &&
         "failed to create descriptor pool");
}

void Renderer::createDescriptorSet() {
  VkDescriptorSetLayout layout = _descriptorSetLayout;
  VkDescriptorSetAllocateInfo allocInfo{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = _descriptorPool,
      .descriptorSetCount = 1,
      .pSetLayouts = &layout,
  };

  const VkResult allocateDescriptorSetResult = vkAllocateDescriptorSets(
      _vkContext.device(), &allocInfo, &_descriptorSet);
  assert(allocateDescriptorSetResult == VK_SUCCESS &&
         "failed to allocate descriptor set");
}

void Renderer::updateInstanceDescriptorSet() {
  const VkDeviceSize bufferSize =
      sizeof(InstanceData) * _instanceBufferCapacity;

  VkDescriptorBufferInfo bufferInfo{
      .buffer = _instanceBuffer.buffer,
      .offset = 0,
      .range = bufferSize,
  };

  VkWriteDescriptorSet descriptorWrite{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = _descriptorSet,
      .dstBinding = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &bufferInfo,
  };

  vkUpdateDescriptorSets(_vkContext.device(), 1, &descriptorWrite, 0, nullptr);
}

void Renderer::updateMeshDescriptorSet() {
  assert(!_meshlets.empty());
  assert(!_clusterVertices.empty());
  assert(!_meshletVertexRefs.empty());
  assert(!_packedClusterTriangles.empty());

  std::array<VkDescriptorBufferInfo, 4> bufferInfos{{
      {
          .buffer = _meshletBuffer.buffer,
          .range = sizeof(Meshlet) * _meshlets.size(),
      },
      {
          .buffer = _clusterVertexBuffer.buffer,
          .range = sizeof(MeshletVertex) * _clusterVertices.size(),
      },
      {
          .buffer = _clusterTriangleBuffer.buffer,
          .range = sizeof(uint32_t) * _packedClusterTriangles.size(),
      },
      {
          .buffer = _meshletVertexRefBuffer.buffer,
          .range = sizeof(uint32_t) * _meshletVertexRefs.size(),
      },
  }};

  constexpr std::array<uint32_t, 4> kBindings = {1, 2, 3, 4};
  std::array<VkWriteDescriptorSet, 4> descriptorWrites{};
  for (uint32_t writeIndex = 0; writeIndex < descriptorWrites.size();
       ++writeIndex) {
    descriptorWrites[writeIndex] = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = _descriptorSet,
        .dstBinding = kBindings[writeIndex],
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &bufferInfos[writeIndex],
    };
  }

  vkUpdateDescriptorSets(_vkContext.device(),
                         static_cast<uint32_t>(descriptorWrites.size()),
                         descriptorWrites.data(), 0, nullptr);
}

void Renderer::updateRenderBucketDescriptorSet() {
  assert(_candidateMeshletCapacity > 0);
  assert(_renderBucket.counterBuffer.buffer != VK_NULL_HANDLE);

  std::array<VkDescriptorBufferInfo, 2> bufferInfos{{
      {
          .buffer = _renderBucket.candidateMeshletBuffer.buffer,
          .range = sizeof(MeshletInstance) * _candidateMeshletCapacity,
      },
      {
          .buffer = _renderBucket.counterBuffer.buffer,
          .range = sizeof(RenderCounters),
      },
  }};

  std::array<VkWriteDescriptorSet, 2> descriptorWrites{};
  for (uint32_t writeIndex = 0; writeIndex < descriptorWrites.size();
       ++writeIndex) {
    descriptorWrites[writeIndex] = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = _descriptorSet,
        .dstBinding = 5 + writeIndex,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &bufferInfos[writeIndex],
    };
  }

  vkUpdateDescriptorSets(_vkContext.device(),
                         static_cast<uint32_t>(descriptorWrites.size()),
                         descriptorWrites.data(), 0, nullptr);
}

MeshId Renderer::loadModel(const GltfModel &model) {
  MeshUploadInfo info{
      .firstMeshlet = static_cast<uint32_t>(_meshlets.size()),
      .meshletCount = static_cast<uint32_t>(model.packedMeshlets.size()),
  };
  const uint32_t clusterVertexBase =
      static_cast<uint32_t>(_clusterVertices.size());
  const uint32_t meshletVertexRefBase =
      static_cast<uint32_t>(_meshletVertexRefs.size());
  const uint32_t clusterTriangleBase =
      static_cast<uint32_t>(_packedClusterTriangles.size());

  _packedClusterTriangles.insert(_packedClusterTriangles.end(),
                                 model.packedClusterTriangles.begin(),
                                 model.packedClusterTriangles.end());

  _clusterVertices.insert(_clusterVertices.end(),
                          model.packedClusterVertices.begin(),
                          model.packedClusterVertices.end());

  _meshletVertexRefs.reserve(_meshletVertexRefs.size() +
                             model.packedMeshletVertexRefs.size());
  for (uint32_t vertexRef : model.packedMeshletVertexRefs) {
    _meshletVertexRefs.push_back(clusterVertexBase + vertexRef);
  }

  _meshlets.insert(_meshlets.end(), model.packedMeshlets.begin(),
                   model.packedMeshlets.end());
  for (auto it = _meshlets.begin() + info.firstMeshlet; it != _meshlets.end();
       ++it) {
    Meshlet &meshlet = *it;
    meshlet.vertexOffset += meshletVertexRefBase;
    meshlet.triangleOffset += clusterTriangleBase;
  }
  _meshes.push_back(info);

  rebuildMeshBuffers();
  return static_cast<MeshId>(_meshes.size() - 1);
}

void Renderer::rebuildMeshBuffers() {

  uploadDeviceLocalBuffer(_meshlets, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          _meshletBuffer);
  uploadDeviceLocalBuffer(_clusterVertices, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          _clusterVertexBuffer);
  uploadDeviceLocalBuffer(_meshletVertexRefs,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          _meshletVertexRefBuffer);
  uploadDeviceLocalBuffer(_packedClusterTriangles,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          _clusterTriangleBuffer);
  updateMeshDescriptorSet();
}

void Renderer::setObjects(const std::vector<Object3D> &objects) {
  _frameInstances.clear();
  _candidateMeshlets.clear();
  uint64_t totalCandidateTriangles = 0;
  _frameInstances.reserve(objects.size());

  for (const Object3D &object : objects) {
    assert(object.meshId < _meshes.size() && "invalid object mesh id");

    const uint32_t instanceId = static_cast<uint32_t>(_frameInstances.size());
    _frameInstances.push_back(makeInstanceData(object));

    const MeshUploadInfo &mesh = _meshes[object.meshId];
    for (uint32_t meshletOffset = 0; meshletOffset < mesh.meshletCount;
         ++meshletOffset) {
      const uint32_t meshletId = mesh.firstMeshlet + meshletOffset;
      const uint32_t triangleCount = meshletTriangleCount(_meshlets[meshletId]);

      _candidateMeshlets.push_back({
          .instanceId = instanceId,
          .meshletId = meshletId,
      });
      totalCandidateTriangles += triangleCount;
    }
  }

  _stats.totalMeshlets = static_cast<uint32_t>(_candidateMeshlets.size());
  _stats.totalTriangles = totalCandidateTriangles;

  if (_frameInstances.empty()) {
    return;
  }

  if (_frameInstances.size() > _instanceBufferCapacity) {
    _vkContext.waitIdle();
    _vkContext.destroyBuffer(_instanceBuffer);
    _instanceBufferCapacity = _frameInstances.size();
    const VkDeviceSize capacitySize =
        sizeof(InstanceData) * _instanceBufferCapacity;
    _vkContext.createBuffer(capacitySize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                            _instanceBuffer);
    updateInstanceDescriptorSet();
  }

  const VkDeviceSize uploadSize =
      sizeof(_frameInstances[0]) * _frameInstances.size();

  void *data = nullptr;
  vmaMapMemory(_vkContext.allocator(), _instanceBuffer.allocation, &data);
  std::memcpy(data, _frameInstances.data(), static_cast<size_t>(uploadSize));
  vmaUnmapMemory(_vkContext.allocator(), _instanceBuffer.allocation);

  uploadRenderBucket();
}

template <typename T>
bool Renderer::uploadHostBuffer(const std::vector<T> &source,
                                VkBufferUsageFlags usage, Buffer<T> &target,
                                size_t &capacity) {
  if (source.empty()) {
    return false;
  }

  bool recreated = false;
  const VkDeviceSize bufferSize = sizeof(source[0]) * source.size();
  if (source.size() > capacity) {
    _vkContext.waitIdle();
    _vkContext.destroyBuffer(target);
    capacity = source.size();
    _vkContext.createBuffer(sizeof(source[0]) * capacity, usage,
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                            target);
    recreated = true;
  }

  void *data = nullptr;
  vmaMapMemory(_vkContext.allocator(), target.allocation, &data);
  std::memcpy(data, source.data(), static_cast<size_t>(bufferSize));
  vmaUnmapMemory(_vkContext.allocator(), target.allocation);
  return recreated;
}

void Renderer::uploadRenderBucket() {
  bool recreated = uploadHostBuffer(
      _candidateMeshlets, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      _renderBucket.candidateMeshletBuffer, _candidateMeshletCapacity);

  if (_renderBucket.counterBuffer.buffer == VK_NULL_HANDLE) {
    _vkContext.createBuffer(
        sizeof(RenderCounters),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, _renderBucket.counterBuffer);
    recreated = true;
  }

  if (recreated) {
    updateRenderBucketDescriptorSet();
  }
}

void Renderer::readCompletedCounters(size_t frameIndex) {
  const Buffer<RenderCounters> &buffer =
      _renderBucket.counterReadbackBuffers[frameIndex];

  RenderCounters counters{};
  void *data = nullptr;
  vmaMapMemory(_vkContext.allocator(), buffer.allocation, &data);
  std::memcpy(&counters, data, sizeof(counters));
  vmaUnmapMemory(_vkContext.allocator(), buffer.allocation);

  _stats.visibleMeshlets = counters.visibleMeshletCount;
  _stats.visibleTriangles = counters.visibleTriangleCount;
}

template <typename T>
void Renderer::uploadDeviceLocalBuffer(const std::vector<T> &source,
                                       VkBufferUsageFlags usage,
                                       Buffer<T> &target) {
  _vkContext.destroyBuffer(target);
  if (source.empty()) {
    return;
  }

  const VkDeviceSize bufferSize = sizeof(source[0]) * source.size();
  Buffer<T> staging;
  _vkContext.createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                          staging);

  void *data = nullptr;
  vmaMapMemory(_vkContext.allocator(), staging.allocation, &data);
  std::memcpy(data, source.data(), static_cast<size_t>(bufferSize));
  vmaUnmapMemory(_vkContext.allocator(), staging.allocation);

  _vkContext.createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | usage,
                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, target);
  _vkContext.copyBuffer(staging.buffer, target.buffer, bufferSize);
  _vkContext.destroyBuffer(staging);
}

void Renderer::cleanupImgui() {
  if (_imguiDescriptorPool == VK_NULL_HANDLE) {
    return;
  }

  ImGui_ImplVulkan_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  vkDestroyDescriptorPool(_vkContext.device(), _imguiDescriptorPool, nullptr);
  _imguiDescriptorPool = VK_NULL_HANDLE;
}

void Renderer::recordCounterReset(VkCommandBuffer commandBuffer) {
  vkCmdFillBuffer(commandBuffer, _renderBucket.counterBuffer.buffer, 0,
                  sizeof(RenderCounters), 0);

  VkBufferMemoryBarrier barrier{
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .buffer = _renderBucket.counterBuffer.buffer,
      .size = VK_WHOLE_SIZE,
  };
  vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_TASK_SHADER_BIT_EXT, 0, 0, nullptr, 1,
                       &barrier, 0, nullptr);
}

void Renderer::recordCounterReadback(VkCommandBuffer commandBuffer,
                                     VkBuffer counterReadbackBuffer) {
  VkBufferMemoryBarrier shaderWriteBarrier{
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .buffer = _renderBucket.counterBuffer.buffer,
      .size = VK_WHOLE_SIZE,
  };
  vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TASK_SHADER_BIT_EXT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1,
                       &shaderWriteBarrier, 0, nullptr);

  VkBufferCopy counterCopy{
      .size = sizeof(RenderCounters),
  };
  vkCmdCopyBuffer(commandBuffer, _renderBucket.counterBuffer.buffer,
                  counterReadbackBuffer, 1, &counterCopy);

  VkBufferMemoryBarrier hostReadBarrier{
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .buffer = counterReadbackBuffer,
      .size = VK_WHOLE_SIZE,
  };
  vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1,
                       &hostReadBarrier, 0, nullptr);
}

void Renderer::renderImgui(VkCommandBuffer commandBuffer, float dt) {
  ImGui_ImplVulkan_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();

  ImGuiWindowFlags flags =
      ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
      ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |
      ImGuiWindowFlags_NoNav;
  ImGui::SetNextWindowPos(ImVec2{12.0f, 12.0f}, ImGuiCond_Always);
  ImGui::SetNextWindowBgAlpha(0.55f);
  ImGui::Begin("stats", nullptr, flags);
  const float fps = dt > 0.0f ? 1000.0f / dt : 0.0f;
  ImGui::Text("FPS: %.2f", fps);
  ImGui::Text("frame time: %.3f ms", dt);
  const std::string visibleMeshlets = formatNumber(_stats.visibleMeshlets);
  const std::string totalMeshlets = formatNumber(_stats.totalMeshlets);
  const std::string visibleTriangles = formatNumber(_stats.visibleTriangles);
  const std::string totalTriangles = formatNumber(_stats.totalTriangles);
  ImGui::Text("rendered meshlets: %s / %s", visibleMeshlets.c_str(),
              totalMeshlets.c_str());
  ImGui::Text("rendered triangles: %s / %s", visibleTriangles.c_str(),
              totalTriangles.c_str());

  ImGui::End();

  ImGui::Render();
  ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commandBuffer);
}

void Renderer::recordRenderingCommands(VkCommandBuffer commandBuffer,
                                       const FrameContext &frame,
                                       const glm::mat4 &viewProjection,
                                       const CameraCullData &cullData,
                                       float dt) {
  std::array<VkImageMemoryBarrier, 2> beginRenderingBarriers{{
      {
          .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
          .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
          .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
          .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
          .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .image = frame.colorImage,
          .subresourceRange =
              {
                  .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                  .levelCount = 1,
                  .layerCount = 1,
              },
      },
      {
          .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
          .dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
          .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
          .newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
          .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .image = frame.depthImage,
          .subresourceRange =
              {
                  .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
                  .levelCount = 1,
                  .layerCount = 1,
              },
      },
  }};

  vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                       VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                           VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
                       0, 0, nullptr, 0, nullptr,
                       static_cast<uint32_t>(beginRenderingBarriers.size()),
                       beginRenderingBarriers.data());

  VkRenderingAttachmentInfo colorAttachment{
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = frame.colorImageView,
      .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .clearValue = {.color = {{0.02f, 0.03f, 0.04f, 1.0f}}},
  };

  VkRenderingAttachmentInfo depthAttachment{
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = frame.depthImageView,
      .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
      .clearValue = {.depthStencil = {.depth = 1.0f, .stencil = 0}},
  };

  const RenderTargetInfo renderTarget = _vkContext.renderTargetInfo();
  VkRenderingInfo renderingInfo{
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = {.offset = {0, 0}, .extent = renderTarget.extent},
      .layerCount = 1,
      .colorAttachmentCount = 1,
      .pColorAttachments = &colorAttachment,
      .pDepthAttachment = &depthAttachment,
  };

  // rendering
  vkCmdBeginRendering(commandBuffer, &renderingInfo);
  vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    _graphicsPipeline);

  if (!_candidateMeshlets.empty()) {
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            _pipelineLayout, 0, 1, &_descriptorSet, 0, nullptr);
    DrawPushConstants pushConstants{
        .viewProjection = viewProjection,
        .view = cullData.view,
        .frustum = cullData.frustum,
        .zNearFar = cullData.zNearFar,
        .candidateCount = static_cast<uint32_t>(_candidateMeshlets.size()),
    };
    vkCmdPushConstants(commandBuffer, _pipelineLayout, kDrawStages, 0,
                       sizeof(DrawPushConstants), &pushConstants);

    const uint32_t groupCount =
        (pushConstants.candidateCount + kTaskGroupSize - 1) / kTaskGroupSize;
    _vkContext.cmdDrawMeshTasks(
        commandBuffer, minValue(groupCount, kTaskGroupsPerRow),
        (groupCount + kTaskGroupsPerRow - 1) / kTaskGroupsPerRow, 1);
  }

  renderImgui(commandBuffer, dt);
  vkCmdEndRendering(commandBuffer);

  VkImageMemoryBarrier presentBarrier{
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = frame.colorImage,
      .subresourceRange =
          {
              .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
              .levelCount = 1,
              .layerCount = 1,
          },
  };

  vkCmdPipelineBarrier(commandBuffer,
                       VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                       VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &presentBarrier);
}

void Renderer::recordCommandBuffer(const FrameContext &frame,
                                   const glm::mat4 &viewProjection,
                                   const CameraCullData &cullData, float dt) {
  VkCommandBuffer commandBuffer = frame.commandBuffer;
  VkCommandBufferBeginInfo beginInfo{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
  };
  const VkResult beginCommandBufferResult =
      vkBeginCommandBuffer(commandBuffer, &beginInfo);
  assert(beginCommandBufferResult == VK_SUCCESS &&
         "failed to begin recording command buffer");

  recordCounterReset(commandBuffer);
  recordRenderingCommands(commandBuffer, frame, viewProjection, cullData, dt);
  recordCounterReadback(
      commandBuffer,
      _renderBucket.counterReadbackBuffers[frame.frameIndex].buffer);

  const VkResult endCommandBufferResult = vkEndCommandBuffer(commandBuffer);
  assert(endCommandBufferResult == VK_SUCCESS &&
         "failed to record command buffer");
}

void Renderer::render(const glm::mat4 &viewProjection,
                      const CameraCullData &cullData, float dt) {
  const FrameContext frame = _vkContext.beginFrame();
  readCompletedCounters(frame.frameIndex);
  recordCommandBuffer(frame, viewProjection, cullData, dt);
  _vkContext.endFrame(frame);
}
