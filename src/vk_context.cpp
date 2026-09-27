#define NOMINMAX
#define VMA_IMPLEMENTATION
#include "vk_context.h"
#include "math_helper.h"

#include <GLFW/glfw3.h>
#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <type_traits>

namespace {

const std::vector<const char *> kValidationLayers = {
    "VK_LAYER_KHRONOS_validation",
};

const std::vector<const char *> kDeviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    VK_EXT_MESH_SHADER_EXTENSION_NAME,
    VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
    VK_KHR_RAY_QUERY_EXTENSION_NAME,
    VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
};

#ifdef NDEBUG
constexpr bool kEnableValidationLayers = false;
#else
constexpr bool kEnableValidationLayers = true;
#endif

VkResult createDebugUtilsMessengerEXT(
    VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT *createInfo,
    const VkAllocationCallbacks *allocator,
    VkDebugUtilsMessengerEXT *messenger) {
  auto fn = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
      vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
  return fn ? fn(instance, createInfo, allocator, messenger)
            : VK_ERROR_EXTENSION_NOT_PRESENT;
}

void destroyDebugUtilsMessengerEXT(VkInstance instance,
                                   VkDebugUtilsMessengerEXT messenger,
                                   const VkAllocationCallbacks *allocator) {
  auto fn = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
      vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
  if (fn) {
    fn(instance, messenger, allocator);
  }
}

VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT, VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT *callbackData, void *) {
  std::cerr << "validation: " << callbackData->pMessage << '\n';
  return VK_FALSE;
}

} // namespace

VkContext::VkContext(GLFWwindow *window) : _window(window) {}

VkContext::~VkContext() { cleanup(); }

void VkContext::init() {
  createInstance();
  setupDebugMessenger();
  createSurface();
  pickPhysicalDevice();
  createLogicalDevice();
  createAllocator();
  createCommandPool();
  createCommandBuffers();
  createSwapChain();
  createSyncObjects();
  createDepthResources();
}

void VkContext::cleanup() {
  if (_device != VK_NULL_HANDLE) {
    cleanupSyncObjects();
    cleanupSwapChain();
    if (_commandPool != VK_NULL_HANDLE) {
      vkDestroyCommandPool(_device, _commandPool, nullptr);
      _commandPool = VK_NULL_HANDLE;
    }
    if (_allocator != VK_NULL_HANDLE) {
      vmaDestroyAllocator(_allocator);
      _allocator = VK_NULL_HANDLE;
    }
    vkDestroyDevice(_device, nullptr);
    _device = VK_NULL_HANDLE;
  }

  if (_instance != VK_NULL_HANDLE) {
    if (kEnableValidationLayers && _debugMessenger != VK_NULL_HANDLE) {
      destroyDebugUtilsMessengerEXT(_instance, _debugMessenger, nullptr);
      _debugMessenger = VK_NULL_HANDLE;
    }
    if (_surface != VK_NULL_HANDLE) {
      vkDestroySurfaceKHR(_instance, _surface, nullptr);
      _surface = VK_NULL_HANDLE;
    }
    vkDestroyInstance(_instance, nullptr);
    _instance = VK_NULL_HANDLE;
  }
}

void VkContext::cleanupSwapChain() {
  cleanupRenderTargets();

  for (auto imageView : _swapChainImageViews) {
    vkDestroyImageView(_device, imageView, nullptr);
  }
  _swapChainImageViews.clear();
  _swapChainImages.clear();

  if (_swapChain != VK_NULL_HANDLE) {
    vkDestroySwapchainKHR(_device, _swapChain, nullptr);
    _swapChain = VK_NULL_HANDLE;
  }
}

void VkContext::cleanupRenderTargets() {
  if (_depthImage.view != VK_NULL_HANDLE) {
    vkDestroyImageView(_device, _depthImage.view, nullptr);
  }
  if (_depthImage.image != VK_NULL_HANDLE) {
    vmaDestroyImage(_allocator, _depthImage.image, _depthImage.allocation);
  }
  _depthImage = {};
}

void VkContext::cleanupSyncObjects() {
  for (auto semaphore : _renderFinishedSemaphores) {
    vkDestroySemaphore(_device, semaphore, nullptr);
  }
  _renderFinishedSemaphores.clear();

  for (auto semaphore : _imageAvailableSemaphores) {
    vkDestroySemaphore(_device, semaphore, nullptr);
  }
  _imageAvailableSemaphores.clear();

  for (auto fence : _inFlightFences) {
    vkDestroyFence(_device, fence, nullptr);
  }
  _inFlightFences.clear();

  _commandBuffers.clear();
}

void VkContext::waitIdle() const {
  if (_device != VK_NULL_HANDLE) {
    vkDeviceWaitIdle(_device);
  }
}

void VkContext::createInstance() {
  assert((!kEnableValidationLayers || checkValidationLayerSupport()) &&
         "validation layers requested but not available");

  VkApplicationInfo appInfo{
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vktest",
      .applicationVersion = VK_MAKE_VERSION(0, 1, 0),
      .pEngineName = "No Engine",
      .engineVersion = VK_MAKE_VERSION(0, 1, 0),
      .apiVersion = VK_API_VERSION_1_3,
  };

  auto extensions = requiredExtensions();
  VkInstanceCreateInfo createInfo{
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &appInfo,
      .enabledExtensionCount = static_cast<uint32_t>(extensions.size()),
      .ppEnabledExtensionNames = extensions.data(),
  };

  VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
  if (kEnableValidationLayers) {
    createInfo.enabledLayerCount =
        static_cast<uint32_t>(kValidationLayers.size());
    createInfo.ppEnabledLayerNames = kValidationLayers.data();
    populateDebugMessengerCreateInfo(debugCreateInfo);
    createInfo.pNext = &debugCreateInfo;
  }

  const VkResult createInstanceResult =
      vkCreateInstance(&createInfo, nullptr, &_instance);
  assert(createInstanceResult == VK_SUCCESS &&
         "failed to create Vulkan instance");
}

bool VkContext::checkValidationLayerSupport() const {
  uint32_t layerCount = 0;
  vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
  std::vector<VkLayerProperties> availableLayers(layerCount);
  vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());
  for (const char *layerName : kValidationLayers) {
    bool found = false;
    for (const VkLayerProperties &layer : availableLayers) {
      if (std::strcmp(layerName, layer.layerName) == 0) {
        found = true;
        break;
      }
    }
    if (!found) {
      return false;
    }
  }
  return true;
}

std::vector<const char *> VkContext::requiredExtensions() const {
  uint32_t glfwExtensionCount = 0;
  const char **glfwExtensions =
      glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
  std::vector<const char *> extensions(glfwExtensions,
                                       glfwExtensions + glfwExtensionCount);
  if (kEnableValidationLayers) {
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  }
  return extensions;
}

void VkContext::populateDebugMessengerCreateInfo(
    VkDebugUtilsMessengerCreateInfoEXT &createInfo) const {
  createInfo = {
      .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
      .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                         VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
      .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                     VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                     VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
      .pfnUserCallback = debugCallback,
  };
}

void VkContext::setupDebugMessenger() {
  if (!kEnableValidationLayers) {
    return;
  }
  VkDebugUtilsMessengerCreateInfoEXT createInfo{};
  populateDebugMessengerCreateInfo(createInfo);
  const VkResult createDebugMessengerResult = createDebugUtilsMessengerEXT(
      _instance, &createInfo, nullptr, &_debugMessenger);
  assert(createDebugMessengerResult == VK_SUCCESS &&
         "failed to set up debug messenger");
}

void VkContext::createSurface() {
  const VkResult createSurfaceResult =
      glfwCreateWindowSurface(_instance, _window, nullptr, &_surface);
  assert(createSurfaceResult == VK_SUCCESS &&
         "failed to create window surface");
}

void VkContext::pickPhysicalDevice() {
  uint32_t deviceCount = 0;
  vkEnumeratePhysicalDevices(_instance, &deviceCount, nullptr);
  assert(deviceCount > 0 && "failed to find GPUs with Vulkan support");
  std::vector<VkPhysicalDevice> devices(deviceCount);
  vkEnumeratePhysicalDevices(_instance, &deviceCount, devices.data());
  for (const auto &device : devices) {
    const QueueFamilyIndices indices = findQueueFamilies(device);
    if (isDeviceSuitable(device, indices)) {
      _physicalDevice = device;
      _queueFamilies = indices;
      return;
    }
  }
  assert(false && "failed to find a suitable GPU");
}

bool VkContext::isDeviceSuitable(VkPhysicalDevice device,
                                 const QueueFamilyIndices &indices) const {
  VkPhysicalDeviceProperties properties{};
  vkGetPhysicalDeviceProperties(device, &properties);
  if (properties.apiVersion < VK_API_VERSION_1_3) {
    return false;
  }

  const bool extensionsSupported = checkDeviceExtensionSupport(device);
  bool swapChainAdequate = false;
  if (extensionsSupported) {
    const auto swapChainSupport = querySwapChainSupport(device);
    swapChainAdequate = !swapChainSupport.formats.empty() &&
                        !swapChainSupport.presentModes.empty();
  }
  return indices.isComplete() && extensionsSupported && swapChainAdequate;
}

VkContext::QueueFamilyIndices
VkContext::findQueueFamilies(VkPhysicalDevice device) const {
  QueueFamilyIndices indices;
  uint32_t queueFamilyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
  std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount,
                                           queueFamilies.data());

  for (uint32_t i = 0; i < queueFamilies.size(); ++i) {
    if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
      indices.graphicsFamily = i;
    }
    VkBool32 presentSupport = false;
    vkGetPhysicalDeviceSurfaceSupportKHR(device, i, _surface, &presentSupport);
    if (presentSupport) {
      indices.presentFamily = i;
    }
    if (indices.isComplete()) {
      break;
    }
  }
  return indices;
}

bool VkContext::checkDeviceExtensionSupport(VkPhysicalDevice device) const {
  uint32_t extensionCount = 0;
  vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount,
                                       nullptr);
  std::vector<VkExtensionProperties> availableExtensions(extensionCount);
  vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount,
                                       availableExtensions.data());

  for (const char *requiredExtension : kDeviceExtensions) {
    bool found = false;
    for (const VkExtensionProperties &extension : availableExtensions) {
      if (std::strcmp(requiredExtension, extension.extensionName) == 0) {
        found = true;
        break;
      }
    }
    if (!found) {
      return false;
    }
  }
  return true;
}

VkContext::SwapChainSupportDetails
VkContext::querySwapChainSupport(VkPhysicalDevice device) const {
  SwapChainSupportDetails details;
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, _surface,
                                            &details.capabilities);
  uint32_t formatCount = 0;
  vkGetPhysicalDeviceSurfaceFormatsKHR(device, _surface, &formatCount, nullptr);
  details.formats.resize(formatCount);
  vkGetPhysicalDeviceSurfaceFormatsKHR(device, _surface, &formatCount,
                                       details.formats.data());
  uint32_t presentModeCount = 0;
  vkGetPhysicalDeviceSurfacePresentModesKHR(device, _surface, &presentModeCount,
                                            nullptr);
  details.presentModes.resize(presentModeCount);
  vkGetPhysicalDeviceSurfacePresentModesKHR(device, _surface, &presentModeCount,
                                            details.presentModes.data());
  return details;
}

void VkContext::createLogicalDevice() {
  std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
  const uint32_t queueFamilies[] = {_queueFamilies.graphicsFamily.value(),
                                    _queueFamilies.presentFamily.value()};
  const float queuePriority = 1.0f;
  for (size_t i = 0; i < 2; ++i) {
    const uint32_t queueFamily = queueFamilies[i];
    if (i > 0 && queueFamily == queueFamilies[0]) {
      continue;
    }

    VkDeviceQueueCreateInfo queueCreateInfo{
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = queueFamily,
        .queueCount = 1,
        .pQueuePriorities = &queuePriority,
    };
    queueCreateInfos.push_back(queueCreateInfo);
  }

  VkPhysicalDeviceMeshShaderFeaturesEXT meshShaderFeatures{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT,
      .taskShader = VK_TRUE,
      .meshShader = VK_TRUE,
  };
  VkPhysicalDeviceRayQueryFeaturesKHR rayQueryFeatures{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR,
      .pNext = &meshShaderFeatures,
      .rayQuery = VK_TRUE,
  };
  VkPhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructureFeatures{
      .sType =
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
      .pNext = &rayQueryFeatures,
      .accelerationStructure = VK_TRUE,
  };
  VkPhysicalDeviceVulkan12Features vulkan12Features{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
      .pNext = &accelerationStructureFeatures,
      // bindless texture array
      .shaderSampledImageArrayNonUniformIndexing = VK_TRUE,
      .descriptorBindingPartiallyBound = VK_TRUE,
      .runtimeDescriptorArray = VK_TRUE,
      .bufferDeviceAddress = VK_TRUE,
  };
  VkPhysicalDeviceVulkan13Features vulkan13Features{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
      .pNext = &vulkan12Features,
      // DXC lowers discard to OpDemoteToHelperInvocation
      .shaderDemoteToHelperInvocation = VK_TRUE,
      .dynamicRendering = VK_TRUE,
  };
  VkPhysicalDeviceFeatures2 deviceFeatures{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = &vulkan13Features,
      // fragmentStoresAndAtomics: the present pass reads the path tracer's
      // storage image, which DXC does not mark read-only
      .features = {.samplerAnisotropy = VK_TRUE,
                   .fragmentStoresAndAtomics = VK_TRUE},
  };
  VkDeviceCreateInfo createInfo{
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &deviceFeatures,
      .queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size()),
      .pQueueCreateInfos = queueCreateInfos.data(),
      .enabledExtensionCount = static_cast<uint32_t>(kDeviceExtensions.size()),
      .ppEnabledExtensionNames = kDeviceExtensions.data(),
  };
  if (kEnableValidationLayers) {
    createInfo.enabledLayerCount =
        static_cast<uint32_t>(kValidationLayers.size());
    createInfo.ppEnabledLayerNames = kValidationLayers.data();
  }

  const VkResult createDeviceResult =
      vkCreateDevice(_physicalDevice, &createInfo, nullptr, &_device);
  assert(createDeviceResult == VK_SUCCESS && "failed to create logical device");
  vkGetDeviceQueue(_device, _queueFamilies.graphicsFamily.value(), 0,
                   &_graphicsQueue);
  vkGetDeviceQueue(_device, _queueFamilies.presentFamily.value(), 0,
                   &_presentQueue);

  _vkCmdDrawMeshTasksEXT = reinterpret_cast<PFN_vkCmdDrawMeshTasksEXT>(
      vkGetDeviceProcAddr(_device, "vkCmdDrawMeshTasksEXT"));
  assert(_vkCmdDrawMeshTasksEXT && "vkCmdDrawMeshTasksEXT not available");

  const auto load = [this](auto &function, const char *name) {
    function = reinterpret_cast<std::remove_reference_t<decltype(function)>>(
        vkGetDeviceProcAddr(_device, name));
    assert(function && "ray tracing entry point not available");
  };
  load(_rayTracing.createAccelerationStructure,
       "vkCreateAccelerationStructureKHR");
  load(_rayTracing.destroyAccelerationStructure,
       "vkDestroyAccelerationStructureKHR");
  load(_rayTracing.getAccelerationStructureBuildSizes,
       "vkGetAccelerationStructureBuildSizesKHR");
  load(_rayTracing.getAccelerationStructureDeviceAddress,
       "vkGetAccelerationStructureDeviceAddressKHR");
  load(_rayTracing.cmdBuildAccelerationStructures,
       "vkCmdBuildAccelerationStructuresKHR");
  load(_rayTracing.cmdWriteAccelerationStructuresProperties,
       "vkCmdWriteAccelerationStructuresPropertiesKHR");
  load(_rayTracing.cmdCopyAccelerationStructure,
       "vkCmdCopyAccelerationStructureKHR");

  VkPhysicalDeviceAccelerationStructurePropertiesKHR
      accelerationStructureProperties{
          .sType =
              VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR,
      };
  VkPhysicalDeviceProperties2 properties{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
      .pNext = &accelerationStructureProperties,
  };
  vkGetPhysicalDeviceProperties2(_physicalDevice, &properties);
  _scratchAlignment =
      accelerationStructureProperties.minAccelerationStructureScratchOffsetAlignment;
}

void VkContext::createAllocator() {
  VmaAllocatorCreateInfo allocatorInfo{
      .flags = VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT |
               VMA_ALLOCATOR_CREATE_KHR_BIND_MEMORY2_BIT |
               VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
      .physicalDevice = _physicalDevice,
      .device = _device,
      .instance = _instance,
      .vulkanApiVersion = VK_API_VERSION_1_3,
  };

  const VkResult createAllocatorResult =
      vmaCreateAllocator(&allocatorInfo, &_allocator);
  assert(createAllocatorResult == VK_SUCCESS && "failed to create VMA allocator");
}

void VkContext::createCommandPool() {
  VkCommandPoolCreateInfo poolInfo{
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
      .queueFamilyIndex = _queueFamilies.graphicsFamily.value(),
  };
  const VkResult createCommandPoolResult =
      vkCreateCommandPool(_device, &poolInfo, nullptr, &_commandPool);
  assert(createCommandPoolResult == VK_SUCCESS &&
         "failed to create command pool");
}

VkSurfaceFormatKHR VkContext::chooseSwapSurfaceFormat(
    const std::vector<VkSurfaceFormatKHR> &formats) const {
  for (const auto &format : formats) {
    if (format.format == VK_FORMAT_B8G8R8A8_SRGB &&
        format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
      return format;
    }
  }
  return formats[0];
}

VkPresentModeKHR VkContext::chooseSwapPresentMode(
    const std::vector<VkPresentModeKHR> &presentModes) const {
  for (const auto &presentMode : presentModes) {
    if (presentMode == VK_PRESENT_MODE_IMMEDIATE_KHR) {
      return presentMode;
    }
  }

  for (const auto &presentMode : presentModes) {
    if (presentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
      return presentMode;
    }
  }
  return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D VkContext::chooseSwapExtent(
    const VkSurfaceCapabilitiesKHR &capabilities) const {
  if (capabilities.currentExtent.width !=
      std::numeric_limits<uint32_t>::max()) {
    return capabilities.currentExtent;
  }
  int width = 0;
  int height = 0;
  glfwGetFramebufferSize(_window, &width, &height);
  VkExtent2D extent{
      .width = static_cast<uint32_t>(width),
      .height = static_cast<uint32_t>(height),
  };
  extent.width = clamp(extent.width, capabilities.minImageExtent.width,
                       capabilities.maxImageExtent.width);
  extent.height = clamp(extent.height, capabilities.minImageExtent.height,
                        capabilities.maxImageExtent.height);
  return extent;
}

void VkContext::createSwapChain() {
  const auto support = querySwapChainSupport(_physicalDevice);
  const auto surfaceFormat = chooseSwapSurfaceFormat(support.formats);
  const auto presentMode = chooseSwapPresentMode(support.presentModes);
  const auto extent = chooseSwapExtent(support.capabilities);

  uint32_t imageCount = support.capabilities.minImageCount + 1;
  if (support.capabilities.maxImageCount > 0 &&
      imageCount > support.capabilities.maxImageCount) {
    imageCount = support.capabilities.maxImageCount;
  }

  VkSwapchainCreateInfoKHR createInfo{
      .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
      .surface = _surface,
      .minImageCount = imageCount,
      .imageFormat = surfaceFormat.format,
      .imageColorSpace = surfaceFormat.colorSpace,
      .imageExtent = extent,
      .imageArrayLayers = 1,
      .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
      .preTransform = support.capabilities.currentTransform,
      .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
      .presentMode = presentMode,
      .clipped = VK_TRUE,
  };

  uint32_t queueFamilyIndices[] = {_queueFamilies.graphicsFamily.value(),
                                   _queueFamilies.presentFamily.value()};
  if (_queueFamilies.graphicsFamily != _queueFamilies.presentFamily) {
    createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
    createInfo.queueFamilyIndexCount = 2;
    createInfo.pQueueFamilyIndices = queueFamilyIndices;
  } else {
    createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  }

  const VkResult createSwapChainResult =
      vkCreateSwapchainKHR(_device, &createInfo, nullptr, &_swapChain);
  assert(createSwapChainResult == VK_SUCCESS && "failed to create swap chain");

  vkGetSwapchainImagesKHR(_device, _swapChain, &imageCount, nullptr);
  _swapChainImages.resize(imageCount);
  vkGetSwapchainImagesKHR(_device, _swapChain, &imageCount,
                          _swapChainImages.data());
  _swapChainImageFormat = surfaceFormat.format;
  _swapChainExtent = extent;
  createImageViews();
}

void VkContext::createImageViews() {
  _swapChainImageViews.resize(_swapChainImages.size());
  for (size_t i = 0; i < _swapChainImages.size(); ++i) {
    _swapChainImageViews[i] = createImageView(
        _swapChainImages[i], _swapChainImageFormat, 1, VK_IMAGE_ASPECT_COLOR_BIT);
  }
}

VkImageView VkContext::createImageView(VkImage image, VkFormat format,
                                       uint32_t mipLevels,
                                       VkImageAspectFlags aspectFlags) const {
  VkImageViewCreateInfo createInfo{
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = image,
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = format,
      .subresourceRange =
          {
              .aspectMask = aspectFlags,
              .levelCount = mipLevels,
              .layerCount = 1,
          },
  };

  VkImageView imageView = VK_NULL_HANDLE;
  const VkResult createImageViewResult =
      vkCreateImageView(_device, &createInfo, nullptr, &imageView);
  assert(createImageViewResult == VK_SUCCESS && "failed to create image view");
  return imageView;
}

void VkContext::createDepthResources() {
  createImage(_swapChainExtent.width, _swapChainExtent.height,
              VK_FORMAT_D32_SFLOAT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, _depthImage);
  _depthImage.view = createImageView(_depthImage.image, VK_FORMAT_D32_SFLOAT,
                                     1, VK_IMAGE_ASPECT_DEPTH_BIT);
}

void VkContext::createCommandBuffers() {
  _commandBuffers.resize(kMaxFramesInFlight);
  VkCommandBufferAllocateInfo allocInfo{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = _commandPool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = static_cast<uint32_t>(_commandBuffers.size()),
  };
  const VkResult allocateCommandBuffersResult =
      vkAllocateCommandBuffers(_device, &allocInfo, _commandBuffers.data());
  assert(allocateCommandBuffersResult == VK_SUCCESS &&
         "failed to allocate command buffers");
}

void VkContext::createSyncObjects() {
  _imageAvailableSemaphores.resize(kMaxFramesInFlight);
  _renderFinishedSemaphores.resize(_swapChainImageViews.size());
  _inFlightFences.resize(kMaxFramesInFlight);

  VkSemaphoreCreateInfo semaphoreInfo{
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
  };
  VkFenceCreateInfo fenceInfo{
      .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
      .flags = VK_FENCE_CREATE_SIGNALED_BIT,
  };

  for (size_t i = 0; i < kMaxFramesInFlight; ++i) {
    const VkResult createImageAvailableSemaphoreResult = vkCreateSemaphore(
        _device, &semaphoreInfo, nullptr, &_imageAvailableSemaphores[i]);
    assert(createImageAvailableSemaphoreResult == VK_SUCCESS &&
           "failed to create image available semaphore");
    const VkResult createFenceResult =
        vkCreateFence(_device, &fenceInfo, nullptr, &_inFlightFences[i]);
    assert(createFenceResult == VK_SUCCESS &&
           "failed to create in-flight fence");
  }

  for (auto &semaphore : _renderFinishedSemaphores) {
    const VkResult createRenderFinishedSemaphoreResult =
        vkCreateSemaphore(_device, &semaphoreInfo, nullptr, &semaphore);
    assert(createRenderFinishedSemaphoreResult == VK_SUCCESS &&
           "failed to create render finished semaphore");
  }
}

FrameContext VkContext::beginFrame() {
  FrameContext frame{
      .frameIndex = _currentFrame,
      .commandBuffer = _commandBuffers[_currentFrame],
      .depthImage = _depthImage.image,
      .depthImageView = _depthImage.view,
      .imageAvailableSemaphore = _imageAvailableSemaphores[_currentFrame],
      .inFlightFence = _inFlightFences[_currentFrame],
  };

  vkWaitForFences(_device, 1, &frame.inFlightFence, VK_TRUE, UINT64_MAX);

  const VkResult result = vkAcquireNextImageKHR(
      _device, _swapChain, UINT64_MAX, frame.imageAvailableSemaphore,
      VK_NULL_HANDLE, &frame.imageIndex);
  assert(result == VK_SUCCESS && "failed to acquire swap chain image");

  frame.colorImage = _swapChainImages[frame.imageIndex];
  frame.colorImageView = _swapChainImageViews[frame.imageIndex];
  frame.renderFinishedSemaphore = _renderFinishedSemaphores[frame.imageIndex];

  vkResetFences(_device, 1, &frame.inFlightFence);
  vkResetCommandBuffer(frame.commandBuffer, 0);

  return frame;
}

void VkContext::endFrame(const FrameContext &frame) {
  VkSemaphore waitSemaphores[] = {frame.imageAvailableSemaphore};
  VkPipelineStageFlags waitStages[] = {
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
  VkSemaphore signalSemaphores[] = {frame.renderFinishedSemaphore};

  VkSubmitInfo submitInfo{
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .waitSemaphoreCount = 1,
      .pWaitSemaphores = waitSemaphores,
      .pWaitDstStageMask = waitStages,
      .commandBufferCount = 1,
      .pCommandBuffers = &frame.commandBuffer,
      .signalSemaphoreCount = 1,
      .pSignalSemaphores = signalSemaphores,
  };

  const VkResult queueSubmitResult =
      vkQueueSubmit(_graphicsQueue, 1, &submitInfo, frame.inFlightFence);
  assert(queueSubmitResult == VK_SUCCESS &&
         "failed to submit draw command buffer");

  VkSwapchainKHR swapChains[] = {_swapChain};
  VkPresentInfoKHR presentInfo{
      .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
      .waitSemaphoreCount = 1,
      .pWaitSemaphores = signalSemaphores,
      .swapchainCount = 1,
      .pSwapchains = swapChains,
      .pImageIndices = &frame.imageIndex,
  };

  const VkResult result = vkQueuePresentKHR(_presentQueue, &presentInfo);
  assert(result == VK_SUCCESS && "failed to present swap chain image");

  _currentFrame = (_currentFrame + 1) % kMaxFramesInFlight;
}

RenderTargetInfo VkContext::renderTargetInfo() const {
  return {
      .colorFormat = _swapChainImageFormat,
      .depthFormat = VK_FORMAT_D32_SFLOAT,
      .extent = _swapChainExtent,
  };
}

void VkContext::immediateSubmit(
    const std::function<void(VkCommandBuffer)> &record) const {
  VkCommandBufferAllocateInfo allocInfo{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = _commandPool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
  };
  VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
  vkAllocateCommandBuffers(_device, &allocInfo, &commandBuffer);

  VkCommandBufferBeginInfo beginInfo{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
  };
  vkBeginCommandBuffer(commandBuffer, &beginInfo);
  record(commandBuffer);
  vkEndCommandBuffer(commandBuffer);

  VkSubmitInfo submitInfo{
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1,
      .pCommandBuffers = &commandBuffer,
  };
  vkQueueSubmit(_graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
  vkQueueWaitIdle(_graphicsQueue);
  vkFreeCommandBuffers(_device, _commandPool, 1, &commandBuffer);
}

void VkContext::copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer,
                           VkDeviceSize size) {
  immediateSubmit([&](VkCommandBuffer commandBuffer) {
    VkBufferCopy copyRegion{
        .size = size,
    };
    vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);
  });
}

Texture VkContext::createTexture(uint32_t width, uint32_t height,
                                 VkFormat format, const void *pixels,
                                 uint32_t bytesPerPixel) {
  const VkDeviceSize size = VkDeviceSize{width} * height * bytesPerPixel;
  Buffer<uint8_t> staging;
  createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
               staging);
  void *data = nullptr;
  vmaMapMemory(_allocator, staging.allocation, &data);
  std::memcpy(data, pixels, static_cast<size_t>(size));
  vmaUnmapMemory(_allocator, staging.allocation);

  uint32_t mipLevels = 1;
  for (uint32_t extent = std::max(width, height); extent > 1; extent /= 2) {
    ++mipLevels;
  }

  VkImageCreateInfo imageInfo{
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = format,
      .extent = {.width = width, .height = height, .depth = 1},
      .mipLevels = mipLevels,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
               VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
  };
  VmaAllocationCreateInfo allocationInfo{
      .usage = VMA_MEMORY_USAGE_AUTO,
      .requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
  };
  Texture texture;
  const VkResult createImageResult =
      vmaCreateImage(_allocator, &imageInfo, &allocationInfo, &texture.image,
                     &texture.allocation, nullptr);
  assert(createImageResult == VK_SUCCESS && "failed to create texture");

  const auto barrier = [&](VkCommandBuffer commandBuffer, uint32_t mip,
                           uint32_t mipCount, VkImageLayout oldLayout,
                           VkImageLayout newLayout, VkAccessFlags srcAccess,
                           VkAccessFlags dstAccess,
                           VkPipelineStageFlags srcStage,
                           VkPipelineStageFlags dstStage) {
    VkImageMemoryBarrier imageBarrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = srcAccess,
        .dstAccessMask = dstAccess,
        .oldLayout = oldLayout,
        .newLayout = newLayout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = texture.image,
        .subresourceRange =
            {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = mip,
                .levelCount = mipCount,
                .layerCount = 1,
            },
    };
    vkCmdPipelineBarrier(commandBuffer, srcStage, dstStage, 0, 0, nullptr, 0,
                         nullptr, 1, &imageBarrier);
  };

  immediateSubmit([&](VkCommandBuffer commandBuffer) {
    barrier(commandBuffer, 0, mipLevels, VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT);

    VkBufferImageCopy copyRegion{
        .imageSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                             .layerCount = 1},
        .imageExtent = {.width = width, .height = height, .depth = 1},
    };
    vkCmdCopyBufferToImage(commandBuffer, staging.buffer, texture.image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                           &copyRegion);

    // each mip is blitted from the previous one, which is then made readable
    int32_t mipWidth = static_cast<int32_t>(width);
    int32_t mipHeight = static_cast<int32_t>(height);
    for (uint32_t mip = 1; mip < mipLevels; ++mip) {
      barrier(commandBuffer, mip - 1, 1, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
              VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
              VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
              VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

      const int32_t nextWidth = std::max(mipWidth / 2, 1);
      const int32_t nextHeight = std::max(mipHeight / 2, 1);
      VkImageBlit blit{
          .srcSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                             .mipLevel = mip - 1,
                             .layerCount = 1},
          .srcOffsets = {{0, 0, 0}, {mipWidth, mipHeight, 1}},
          .dstSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                             .mipLevel = mip,
                             .layerCount = 1},
          .dstOffsets = {{0, 0, 0}, {nextWidth, nextHeight, 1}},
      };
      vkCmdBlitImage(commandBuffer, texture.image,
                     VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, texture.image,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit,
                     VK_FILTER_LINEAR);

      barrier(commandBuffer, mip - 1, 1, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
              VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT,
              VK_PIPELINE_STAGE_TRANSFER_BIT,
              VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
      mipWidth = nextWidth;
      mipHeight = nextHeight;
    }

    barrier(commandBuffer, mipLevels - 1, 1,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
  });

  destroyBuffer(staging);
  texture.view = createImageView(texture.image, format, mipLevels,
                                 VK_IMAGE_ASPECT_COLOR_BIT);
  return texture;
}

Texture VkContext::createStorageImage(uint32_t width, uint32_t height,
                                      VkFormat format) {
  VkImageCreateInfo imageInfo{
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = format,
      .extent = {.width = width, .height = height, .depth = 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_STORAGE_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
  };
  VmaAllocationCreateInfo allocationInfo{
      .usage = VMA_MEMORY_USAGE_AUTO,
      .requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
  };
  Texture image;
  const VkResult createImageResult =
      vmaCreateImage(_allocator, &imageInfo, &allocationInfo, &image.image,
                     &image.allocation, nullptr);
  assert(createImageResult == VK_SUCCESS && "failed to create storage image");

  immediateSubmit([&](VkCommandBuffer commandBuffer) {
    VkImageMemoryBarrier barrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image.image,
        .subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                             .levelCount = 1,
                             .layerCount = 1},
    };
    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &barrier);
  });

  image.view =
      createImageView(image.image, format, 1, VK_IMAGE_ASPECT_COLOR_BIT);
  return image;
}

VkDeviceAddress VkContext::bufferAddress(VkBuffer buffer) const {
  VkBufferDeviceAddressInfo info{
      .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
      .buffer = buffer,
  };
  return vkGetBufferDeviceAddress(_device, &info);
}

void VkContext::destroyTexture(Texture &texture) const {
  if (texture.view != VK_NULL_HANDLE) {
    vkDestroyImageView(_device, texture.view, nullptr);
  }
  if (texture.image != VK_NULL_HANDLE) {
    vmaDestroyImage(_allocator, texture.image, texture.allocation);
  }
  texture = {};
}

VkShaderModule
VkContext::createShaderModule(const std::vector<char> &code) const {
  VkShaderModuleCreateInfo createInfo{
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = code.size(),
      .pCode = reinterpret_cast<const uint32_t *>(code.data()),
  };

  VkShaderModule shaderModule = VK_NULL_HANDLE;
  const VkResult createShaderModuleResult =
      vkCreateShaderModule(_device, &createInfo, nullptr, &shaderModule);
  assert(createShaderModuleResult == VK_SUCCESS &&
         "failed to create shader module");
  return shaderModule;
}

void VkContext::createImage(uint32_t width, uint32_t height, VkFormat format,
                            VkImageUsageFlags usage,
                            VkMemoryPropertyFlags properties,
                            VkContext::Image &image) const {
  VkImageCreateInfo imageInfo{
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = format,
      .extent = {.width = width, .height = height, .depth = 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
  };

  VmaAllocationCreateInfo allocationInfo{
      .usage = VMA_MEMORY_USAGE_AUTO,
      .requiredFlags = properties,
  };

  const VkResult createImageResult =
      vmaCreateImage(_allocator, &imageInfo, &allocationInfo, &image.image,
                     &image.allocation, nullptr);
  assert(createImageResult == VK_SUCCESS && "failed to create image");
}
