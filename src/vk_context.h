#pragma once

#include <cassert>
#include <functional>
#include <optional>
#include <vector>
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

struct GLFWwindow;

struct FrameContext {
  size_t frameIndex = 0;
  uint32_t imageIndex = 0;
  VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
  VkImage colorImage = VK_NULL_HANDLE;
  VkImageView colorImageView = VK_NULL_HANDLE;
  VkImage depthImage = VK_NULL_HANDLE;
  VkImageView depthImageView = VK_NULL_HANDLE;
  // graphics queue waits until the swapchain image is available to render into
  VkSemaphore imageAvailableSemaphore = VK_NULL_HANDLE;
  // present queue waits until the graphics queue finished rendering the image
  VkSemaphore renderFinishedSemaphore = VK_NULL_HANDLE;
  // CPU waits until the submitted GPU work for that frame slot is done
  VkFence inFlightFence = VK_NULL_HANDLE;
};

struct RenderTargetInfo {
  VkFormat colorFormat = VK_FORMAT_UNDEFINED;
  VkFormat depthFormat = VK_FORMAT_D32_SFLOAT;
  VkExtent2D extent{};
};

struct Texture {
  VkImage image = VK_NULL_HANDLE;
  VmaAllocation allocation = VK_NULL_HANDLE;
  VkImageView view = VK_NULL_HANDLE;
};

// extension entry points are not exported by the loader
struct RayTracingFunctions {
  PFN_vkCreateAccelerationStructureKHR createAccelerationStructure = nullptr;
  PFN_vkDestroyAccelerationStructureKHR destroyAccelerationStructure = nullptr;
  PFN_vkGetAccelerationStructureBuildSizesKHR
      getAccelerationStructureBuildSizes = nullptr;
  PFN_vkGetAccelerationStructureDeviceAddressKHR
      getAccelerationStructureDeviceAddress = nullptr;
  PFN_vkCmdBuildAccelerationStructuresKHR cmdBuildAccelerationStructures =
      nullptr;
  PFN_vkCmdWriteAccelerationStructuresPropertiesKHR
      cmdWriteAccelerationStructuresProperties = nullptr;
  PFN_vkCmdCopyAccelerationStructureKHR cmdCopyAccelerationStructure = nullptr;
};

template <typename T> struct Buffer {
  VkBuffer buffer = VK_NULL_HANDLE;
  VmaAllocation allocation = VK_NULL_HANDLE;
};

class VkContext {
public:
  explicit VkContext(GLFWwindow *window);
  ~VkContext();

  VkContext(const VkContext &) = delete;
  VkContext &operator=(const VkContext &) = delete;

  static constexpr size_t kMaxFramesInFlight = 2;

  void init();
  void cleanup();
  FrameContext beginFrame();
  void endFrame(const FrameContext &frame);
  VkShaderModule createShaderModule(const std::vector<char> &code) const;
  RenderTargetInfo renderTargetInfo() const;
  void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);
  // records into a one-off command buffer and waits for it to finish.
  void immediateSubmit(
      const std::function<void(VkCommandBuffer)> &record) const;
  // uploads tightly packed pixels and generates the full mip chain.
  Texture createTexture(uint32_t width, uint32_t height, VkFormat format,
                        const void *pixels, uint32_t bytesPerPixel = 4);
  void destroyTexture(Texture &texture) const;
  // a single-mip image left in VK_IMAGE_LAYOUT_GENERAL
  Texture createStorageImage(uint32_t width, uint32_t height, VkFormat format);
  VkDeviceAddress bufferAddress(VkBuffer buffer) const;
  void waitIdle() const;
  template <typename T>
  void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                    VkMemoryPropertyFlags properties, Buffer<T> &buffer,
                    VkDeviceSize minAlignment = 0) const;
  template <typename T> void destroyBuffer(Buffer<T> &buffer) const;

  VkDevice device() const { return _device; }
  VmaAllocator allocator() const { return _allocator; }
  VkInstance instance() const { return _instance; }
  VkPhysicalDevice physicalDevice() const { return _physicalDevice; }
  VkQueue graphicsQueue() const { return _graphicsQueue; }
  uint32_t graphicsQueueFamily() const {
    return _queueFamilies.graphicsFamily.value();
  }
  uint32_t swapChainImageCount() const {
    return static_cast<uint32_t>(_swapChainImages.size());
  }
  const RayTracingFunctions &rayTracing() const { return _rayTracing; }
  VkDeviceSize accelerationStructureScratchAlignment() const {
    return _scratchAlignment;
  }
  void cmdDrawMeshTasks(VkCommandBuffer commandBuffer, uint32_t groupCountX,
                       uint32_t groupCountY, uint32_t groupCountZ) const {
    _vkCmdDrawMeshTasksEXT(commandBuffer, groupCountX, groupCountY,
                           groupCountZ);
  }

private:
  struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete() const {
      return graphicsFamily.has_value() && presentFamily.has_value();
    }
  };

  struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities{};
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
  };

  struct Image {
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
  };

  void createInstance();
  bool checkValidationLayerSupport() const;
  std::vector<const char *> requiredExtensions() const;
  void populateDebugMessengerCreateInfo(
      VkDebugUtilsMessengerCreateInfoEXT &createInfo) const;
  void setupDebugMessenger();
  void createSurface();
  void pickPhysicalDevice();
  bool isDeviceSuitable(VkPhysicalDevice device,
                        const QueueFamilyIndices &indices) const;
  bool checkDeviceExtensionSupport(VkPhysicalDevice device) const;
  void createLogicalDevice();
  void createAllocator();
  void createCommandPool();
  void createSwapChain();
  void cleanupSwapChain();
  void createDepthResources();
  void createCommandBuffers();
  void createSyncObjects();
  void cleanupRenderTargets();
  QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) const;
  SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) const;
  VkSurfaceFormatKHR
  chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR> &formats) const;
  VkPresentModeKHR chooseSwapPresentMode(
      const std::vector<VkPresentModeKHR> &presentModes) const;
  VkExtent2D
  chooseSwapExtent(const VkSurfaceCapabilitiesKHR &capabilities) const;
  void createImageViews();
  VkImageView createImageView(VkImage image, VkFormat format, uint32_t mipLevels,
                              VkImageAspectFlags aspectFlags) const;
  void createImage(uint32_t width, uint32_t height, VkFormat format,
                   VkImageUsageFlags usage, VkMemoryPropertyFlags properties,
                   Image &image) const;
  void cleanupSyncObjects();

  VkInstance _instance = VK_NULL_HANDLE;
  VkDebugUtilsMessengerEXT _debugMessenger = VK_NULL_HANDLE;
  VkSurfaceKHR _surface = VK_NULL_HANDLE;
  VkPhysicalDevice _physicalDevice = VK_NULL_HANDLE;
  VkDevice _device = VK_NULL_HANDLE;
  VkQueue _graphicsQueue = VK_NULL_HANDLE;
  VkQueue _presentQueue = VK_NULL_HANDLE;
  VmaAllocator _allocator = VK_NULL_HANDLE;
  QueueFamilyIndices _queueFamilies;
  VkCommandPool _commandPool = VK_NULL_HANDLE;
  VkSwapchainKHR _swapChain = VK_NULL_HANDLE;
  std::vector<VkImage> _swapChainImages;
  std::vector<VkImageView> _swapChainImageViews;
  VkFormat _swapChainImageFormat = VK_FORMAT_UNDEFINED;
  VkExtent2D _swapChainExtent{};
  Image _depthImage;
  std::vector<VkCommandBuffer> _commandBuffers;
  std::vector<VkSemaphore> _imageAvailableSemaphores;
  std::vector<VkSemaphore> _renderFinishedSemaphores;
  std::vector<VkFence> _inFlightFences;
  size_t _currentFrame = 0;
  // extension entry points are not exported by the loader
  PFN_vkCmdDrawMeshTasksEXT _vkCmdDrawMeshTasksEXT = nullptr;
  RayTracingFunctions _rayTracing;
  VkDeviceSize _scratchAlignment = 0;

public:
  GLFWwindow *_window = nullptr;
};

template <typename T>
void VkContext::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                             VkMemoryPropertyFlags properties,
                             Buffer<T> &buffer,
                             VkDeviceSize minAlignment) const {
  VkBufferCreateInfo bufferInfo{
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = size,
      .usage = usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
  };

  VmaAllocationCreateInfo allocationInfo{
      .usage = VMA_MEMORY_USAGE_AUTO,
      .requiredFlags = properties,
  };
  if (properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
    const bool readbackBuffer = (usage & VK_BUFFER_USAGE_TRANSFER_DST_BIT) &&
                                !(usage & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    allocationInfo.flags =
        readbackBuffer ? VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
                       : VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
  }

  const VkResult createBufferResult = vmaCreateBufferWithAlignment(
      _allocator, &bufferInfo, &allocationInfo, minAlignment, &buffer.buffer,
      &buffer.allocation, nullptr);
  assert(createBufferResult == VK_SUCCESS && "failed to create buffer");
}

template <typename T> void VkContext::destroyBuffer(Buffer<T> &buffer) const {
  if (buffer.buffer != VK_NULL_HANDLE) {
    vmaDestroyBuffer(_allocator, buffer.buffer, buffer.allocation);
  }
  buffer = {};
}
