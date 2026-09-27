#pragma once

#include "vk_context.h"

#include <cstdint>
#include <vector>

struct AccelerationStructure {
  VkAccelerationStructureKHR handle = VK_NULL_HANDLE;
  Buffer<uint8_t> buffer;
  VkDeviceAddress address = 0;
};

// Owns one BLAS per mesh and the scene TLAS.
class RayTracingScene {
public:
  explicit RayTracingScene(VkContext &context) : _context(context) {}
  ~RayTracingScene() { destroy(); }

  RayTracingScene(const RayTracingScene &) = delete;
  RayTracingScene &operator=(const RayTracingScene &) = delete;

  // Builds and compacts a BLAS. positions must stay alive only for the build;
  // indices is a flat triangle list into it.
  void addMesh(VkBuffer positions, uint32_t vertexCount,
               const std::vector<uint32_t> &indices);
  // Rebuilds the TLAS; accelerationStructureReference holds the mesh index
  // and is replaced by that BLAS's address.
  void buildTlas(std::vector<VkAccelerationStructureInstanceKHR> instances);
  VkAccelerationStructureKHR tlas() const { return _tlas.handle; }
  void destroy();

private:
  AccelerationStructure
  createAccelerationStructure(VkAccelerationStructureTypeKHR type,
                              VkDeviceSize size) const;
  void destroyAccelerationStructure(AccelerationStructure &structure) const;

  VkContext &_context;
  std::vector<AccelerationStructure> _blases;
  AccelerationStructure _tlas;
};
