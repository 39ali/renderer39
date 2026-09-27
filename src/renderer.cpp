#define NOMINMAX
#include "renderer.h"

#include "math_helper.h"

#include <array>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include <string>

namespace {
// must match shaders/scene_bindings.hlsli
enum Binding : uint32_t {
  kBindingInstances = 0,
  kBindingMeshlets = 1,
  kBindingVertexPositions = 2,
  kBindingVertexAttributes = 3,
  kBindingMeshletVertexRefs = 4,
  kBindingMeshletTriangles = 5,
  kBindingCandidateMeshlets = 6,
  kBindingCounters = 7,
  kBindingMaterials = 8,
  kBindingSampler = 9,
  kBindingTextures = 10,
  kBindingCount = 11,
};
constexpr uint32_t kStorageBufferBindingCount = 9;
constexpr VkShaderStageFlags kDrawStages = VK_SHADER_STAGE_TASK_BIT_EXT |
                                           VK_SHADER_STAGE_MESH_BIT_EXT |
                                           VK_SHADER_STAGE_FRAGMENT_BIT;

const char *const kDebugViewNames[] = {
    "lit",       "base color", "normal",   "metallic",
    "roughness", "occlusion",  "emissive", "uv",
};

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
  createSampler();
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
    _vkContext.destroyBuffer(_materialBuffer);
    _vkContext.destroyBuffer(_meshletTriangleBuffer);
    _vkContext.destroyBuffer(_meshletVertexRefBuffer);
    _vkContext.destroyBuffer(_vertexAttributeBuffer);
    _vkContext.destroyBuffer(_vertexPositionBuffer);
    _vkContext.destroyBuffer(_meshletBuffer);
    for (Texture &texture : _textures) {
      _vkContext.destroyTexture(texture);
    }
    vkDestroySampler(_vkContext.device(), _materialSampler, nullptr);
    vkDestroyPipeline(_vkContext.device(), _skyPipeline, nullptr);
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
  std::array<VkDescriptorBindingFlags, kBindingCount> bindingFlags{};
  for (uint32_t binding = 0; binding < kStorageBufferBindingCount; ++binding) {
    bindings[binding] = {
        .binding = binding,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags = kDrawStages,
    };
  }
  bindings[kBindingSampler] = {
      .binding = kBindingSampler,
      .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
  };
  // bindless: only the first _textures.size() entries are ever written
  bindings[kBindingTextures] = {
      .binding = kBindingTextures,
      .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
      .descriptorCount = kMaxTextures,
      .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
  };
  bindingFlags[kBindingTextures] = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;

  VkDescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsInfo{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
      .bindingCount = static_cast<uint32_t>(bindingFlags.size()),
      .pBindingFlags = bindingFlags.data(),
  };
  VkDescriptorSetLayoutCreateInfo layoutInfo{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .pNext = &bindingFlagsInfo,
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
      .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
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

  // the sky: a fullscreen triangle at the far plane drawn after the meshes
  const VkShaderModule skyVertShaderModule = _vkContext.createShaderModule(
      _resources.readBinary("shaders/sky.vert.spv"));
  const VkShaderModule skyFragShaderModule = _vkContext.createShaderModule(
      _resources.readBinary("shaders/sky.frag.spv"));
  const std::array<VkPipelineShaderStageCreateInfo, 2> skyStages{{
      {
          .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_VERTEX_BIT,
          .module = skyVertShaderModule,
          .pName = "main",
      },
      {
          .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
          .module = skyFragShaderModule,
          .pName = "main",
      },
  }};
  VkPipelineVertexInputStateCreateInfo skyVertexInput{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
  };
  VkPipelineInputAssemblyStateCreateInfo skyInputAssembly{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
  };
  VkPipelineRasterizationStateCreateInfo skyRasterizer = rasterizer;
  skyRasterizer.cullMode = VK_CULL_MODE_NONE;
  VkPipelineDepthStencilStateCreateInfo skyDepthStencil = depthStencil;
  skyDepthStencil.depthWriteEnable = VK_FALSE;
  skyDepthStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

  VkGraphicsPipelineCreateInfo skyPipelineInfo = pipelineInfo;
  skyPipelineInfo.stageCount = static_cast<uint32_t>(skyStages.size());
  skyPipelineInfo.pStages = skyStages.data();
  skyPipelineInfo.pVertexInputState = &skyVertexInput;
  skyPipelineInfo.pInputAssemblyState = &skyInputAssembly;
  skyPipelineInfo.pRasterizationState = &skyRasterizer;
  skyPipelineInfo.pDepthStencilState = &skyDepthStencil;
  const VkResult createSkyPipelineResult =
      vkCreateGraphicsPipelines(_vkContext.device(), VK_NULL_HANDLE, 1,
                                &skyPipelineInfo, nullptr, &_skyPipeline);
  assert(createSkyPipelineResult == VK_SUCCESS &&
         "failed to create sky pipeline");
  vkDestroyShaderModule(_vkContext.device(), skyFragShaderModule, nullptr);
  vkDestroyShaderModule(_vkContext.device(), skyVertShaderModule, nullptr);

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
  const std::array<VkDescriptorPoolSize, 3> poolSizes{{
      {
          .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .descriptorCount = kStorageBufferBindingCount,
      },
      {
          .type = VK_DESCRIPTOR_TYPE_SAMPLER,
          .descriptorCount = 1,
      },
      {
          .type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
          .descriptorCount = kMaxTextures,
      },
  }};

  VkDescriptorPoolCreateInfo poolInfo{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1,
      .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
      .pPoolSizes = poolSizes.data(),
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

void Renderer::createSampler() {
  // one trilinear, anisotropic, repeating sampler for every material texture;
  // per-texture glTF sampler settings are ignored for now.
  VkSamplerCreateInfo samplerInfo{
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = VK_FILTER_LINEAR,
      .minFilter = VK_FILTER_LINEAR,
      .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
      .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
      .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
      .anisotropyEnable = VK_TRUE,
      .maxAnisotropy = 16.0f,
      .maxLod = VK_LOD_CLAMP_NONE,
  };
  const VkResult createSamplerResult = vkCreateSampler(
      _vkContext.device(), &samplerInfo, nullptr, &_materialSampler);
  assert(createSamplerResult == VK_SUCCESS && "failed to create sampler");

  VkDescriptorImageInfo imageInfo{.sampler = _materialSampler};
  VkWriteDescriptorSet descriptorWrite{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = _descriptorSet,
      .dstBinding = kBindingSampler,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
      .pImageInfo = &imageInfo,
  };
  vkUpdateDescriptorSets(_vkContext.device(), 1, &descriptorWrite, 0, nullptr);
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
      .dstBinding = kBindingInstances,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &bufferInfo,
  };

  vkUpdateDescriptorSets(_vkContext.device(), 1, &descriptorWrite, 0, nullptr);
}

void Renderer::updateMeshDescriptorSet() {
  const std::array<std::pair<Binding, VkDescriptorBufferInfo>, 6> buffers{{
      {kBindingMeshlets,
       {.buffer = _meshletBuffer.buffer,
        .range = sizeof(Meshlet) * _meshlets.size()}},
      {kBindingVertexPositions,
       {.buffer = _vertexPositionBuffer.buffer,
        .range = sizeof(VertexPosition) * _vertexPositions.size()}},
      {kBindingVertexAttributes,
       {.buffer = _vertexAttributeBuffer.buffer,
        .range = sizeof(VertexAttributes) * _vertexAttributes.size()}},
      {kBindingMeshletVertexRefs,
       {.buffer = _meshletVertexRefBuffer.buffer,
        .range = sizeof(uint32_t) * _meshletVertexRefs.size()}},
      {kBindingMeshletTriangles,
       {.buffer = _meshletTriangleBuffer.buffer,
        .range = sizeof(uint32_t) * _meshletTriangles.size()}},
      {kBindingMaterials,
       {.buffer = _materialBuffer.buffer,
        .range = sizeof(Material) * _materials.size()}},
  }};

  std::vector<VkWriteDescriptorSet> descriptorWrites;
  for (const auto &[binding, bufferInfo] : buffers) {
    assert(bufferInfo.buffer != VK_NULL_HANDLE && bufferInfo.range > 0);
    descriptorWrites.push_back({
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = _descriptorSet,
        .dstBinding = binding,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &bufferInfo,
    });
  }

  std::vector<VkDescriptorImageInfo> imageInfos;
  imageInfos.reserve(_textures.size());
  for (const Texture &texture : _textures) {
    imageInfos.push_back({
        .imageView = texture.view,
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    });
  }
  if (!imageInfos.empty()) {
    descriptorWrites.push_back({
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = _descriptorSet,
        .dstBinding = kBindingTextures,
        .descriptorCount = static_cast<uint32_t>(imageInfos.size()),
        .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
        .pImageInfo = imageInfos.data(),
    });
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
        .dstBinding = kBindingCandidateMeshlets + writeIndex,
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
      .meshletCount = static_cast<uint32_t>(model.meshlets.size()),
  };
  const uint32_t vertexBase = static_cast<uint32_t>(_vertexPositions.size());
  const uint32_t meshletVertexRefBase =
      static_cast<uint32_t>(_meshletVertexRefs.size());
  const uint32_t meshletTriangleBase =
      static_cast<uint32_t>(_meshletTriangles.size());
  const uint32_t materialBase = static_cast<uint32_t>(_materials.size());
  const uint32_t textureBase = static_cast<uint32_t>(_textures.size());
  assert(textureBase + model.textures.size() <= kMaxTextures);

  _vertexPositions.insert(_vertexPositions.end(), model.positions.begin(),
                          model.positions.end());
  _vertexAttributes.insert(_vertexAttributes.end(), model.attributes.begin(),
                           model.attributes.end());
  for (const uint32_t index : model.indices) {
    _indices.push_back(vertexBase + index);
  }
  for (const uint32_t vertexRef : model.meshletVertexRefs) {
    _meshletVertexRefs.push_back(vertexBase + vertexRef);
  }
  _meshletTriangles.insert(_meshletTriangles.end(),
                           model.meshletTriangles.begin(),
                           model.meshletTriangles.end());

  for (Meshlet meshlet : model.meshlets) {
    meshlet.vertexOffset += meshletVertexRefBase;
    meshlet.triangleOffset += meshletTriangleBase;
    meshlet.materialIndex += materialBase;
    _meshlets.push_back(meshlet);
  }

  const auto rebase = [textureBase](uint32_t &texture) {
    if (texture != kInvalidTexture) {
      texture += textureBase;
    }
  };
  for (Material material : model.materials) {
    rebase(material.baseColorTexture);
    rebase(material.metallicRoughnessTexture);
    rebase(material.normalTexture);
    rebase(material.occlusionTexture);
    rebase(material.emissiveTexture);
    _materials.push_back(material);
  }

  uploadTextures(model.textures);
  _meshes.push_back(info);

  rebuildMeshBuffers();
  return static_cast<MeshId>(_meshes.size() - 1);
}

void Renderer::uploadTextures(const std::vector<TextureData> &textures) {
  for (const TextureData &texture : textures) {
    _textures.push_back(_vkContext.createTexture(
        texture.width, texture.height,
        texture.srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM,
        texture.rgba.data()));
  }
}

void Renderer::setEnvironment(const HdrImage &image) {
  assert(_environmentTexture == kInvalidTexture &&
         "replacing the environment is not supported yet");
  assert(_textures.size() < kMaxTextures);
  _vkContext.waitIdle();
  _environmentTexture = static_cast<uint32_t>(_textures.size());
  _textures.push_back(_vkContext.createTexture(
      image.width, image.height, VK_FORMAT_R16G16B16A16_SFLOAT,
      image.rgbaHalf.data(), 4 * sizeof(uint16_t)));
  _environmentMaxMip =
      std::floor(std::log2(static_cast<float>(std::max(image.width,
                                                       image.height))));
  // materials may not be loaded yet, so the texture array is written with
  // the rest of the mesh descriptors.
  if (!_materials.empty()) {
    updateMeshDescriptorSet();
  }
}

void Renderer::rebuildMeshBuffers() {
  _vkContext.waitIdle();
  uploadDeviceLocalBuffer(_meshlets, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          _meshletBuffer);
  uploadDeviceLocalBuffer(_vertexPositions, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          _vertexPositionBuffer);
  uploadDeviceLocalBuffer(_vertexAttributes,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          _vertexAttributeBuffer);
  uploadDeviceLocalBuffer(_meshletVertexRefs,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          _meshletVertexRefBuffer);
  uploadDeviceLocalBuffer(_meshletTriangles, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          _meshletTriangleBuffer);
  uploadDeviceLocalBuffer(_materials, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          _materialBuffer);
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

  ImGui::Separator();
  int debugView = static_cast<int>(_debugView);
  ImGui::SetNextItemWidth(160.0f);
  if (ImGui::Combo("view", &debugView, kDebugViewNames,
                   IM_ARRAYSIZE(kDebugViewNames))) {
    _debugView = static_cast<uint32_t>(debugView);
  }
  ImGui::SetNextItemWidth(160.0f);
  ImGui::SliderFloat("light azimuth", &_lightAzimuth, -180.0f, 180.0f);
  ImGui::SetNextItemWidth(160.0f);
  ImGui::SliderFloat("light elevation", &_lightElevation, -10.0f, 90.0f);
  ImGui::SetNextItemWidth(160.0f);
  ImGui::SliderFloat("light intensity", &_lightIntensity, 0.0f, 10.0f);
  ImGui::SetNextItemWidth(160.0f);
  ImGui::SliderFloat("environment", &_environmentIntensity, 0.0f, 4.0f);

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
    const float azimuth = glm::radians(_lightAzimuth);
    const float elevation = glm::radians(_lightElevation);
    const glm::vec3 towardsLight{std::cos(elevation) * std::sin(azimuth),
                                 std::sin(elevation),
                                 std::cos(elevation) * std::cos(azimuth)};
    DrawPushConstants pushConstants{
        .viewProjection = viewProjection,
        .view = cullData.view,
        .frustum = cullData.frustum,
        .zNearFar = cullData.zNearFar,
        .candidateCount = static_cast<uint32_t>(_candidateMeshlets.size()),
        .debugView = _debugView,
        .cameraPosition = glm::vec4(glm::vec3(glm::inverse(cullData.view)[3]),
                                    1.0f),
        .lightDirection = glm::vec4(-towardsLight, _lightIntensity),
        .environmentTexture = _environmentTexture,
        .environmentIntensity = _environmentIntensity,
        .environmentMaxMip = _environmentMaxMip,
    };
    vkCmdPushConstants(commandBuffer, _pipelineLayout, kDrawStages, 0,
                       sizeof(DrawPushConstants), &pushConstants);

    const uint32_t groupCount =
        (pushConstants.candidateCount + kTaskGroupSize - 1) / kTaskGroupSize;
    _vkContext.cmdDrawMeshTasks(
        commandBuffer, minValue(groupCount, kTaskGroupsPerRow),
        (groupCount + kTaskGroupsPerRow - 1) / kTaskGroupsPerRow, 1);

    // same layout, descriptors and push constants as the mesh draw
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      _skyPipeline);
    vkCmdDraw(commandBuffer, 3, 1, 0, 0);
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
