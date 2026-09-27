#include "raytracing.h"

#include <cassert>
#include <cstring>

namespace {

template <typename T>
void createHostBuffer(VkContext &context, const std::vector<T> &data,
                      VkBufferUsageFlags usage, Buffer<T> &buffer) {
  const VkDeviceSize size = sizeof(T) * data.size();
  context.createBuffer(size, usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                       buffer);
  void *mapped = nullptr;
  vmaMapMemory(context.allocator(), buffer.allocation, &mapped);
  std::memcpy(mapped, data.data(), static_cast<size_t>(size));
  vmaUnmapMemory(context.allocator(), buffer.allocation);
}

void accelerationStructureBarrier(VkCommandBuffer commandBuffer) {
  VkMemoryBarrier barrier{
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
      .dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR,
  };
  vkCmdPipelineBarrier(
      commandBuffer, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
      VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &barrier,
      0, nullptr, 0, nullptr);
}

} // namespace

AccelerationStructure RayTracingScene::createAccelerationStructure(
    VkAccelerationStructureTypeKHR type, VkDeviceSize size) const {
  AccelerationStructure structure;
  _context.createBuffer(size,
                        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, structure.buffer);

  VkAccelerationStructureCreateInfoKHR createInfo{
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
      .buffer = structure.buffer.buffer,
      .size = size,
      .type = type,
  };
  const VkResult result = _context.rayTracing().createAccelerationStructure(
      _context.device(), &createInfo, nullptr, &structure.handle);
  assert(result == VK_SUCCESS && "failed to create acceleration structure");

  VkAccelerationStructureDeviceAddressInfoKHR addressInfo{
      .sType =
          VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
      .accelerationStructure = structure.handle,
  };
  structure.address =
      _context.rayTracing().getAccelerationStructureDeviceAddress(
          _context.device(), &addressInfo);
  return structure;
}

void RayTracingScene::destroyAccelerationStructure(
    AccelerationStructure &structure) const {
  if (structure.handle != VK_NULL_HANDLE) {
    _context.rayTracing().destroyAccelerationStructure(
        _context.device(), structure.handle, nullptr);
  }
  _context.destroyBuffer(structure.buffer);
  structure = {};
}

void RayTracingScene::addMesh(VkBuffer positions, uint32_t vertexCount,
                              const std::vector<uint32_t> &indices) {
  assert(!indices.empty() && indices.size() % 3 == 0);
  const RayTracingFunctions &rt = _context.rayTracing();

  // build input only; freed once the BLAS holds its own copy
  Buffer<uint32_t> indexBuffer;
  createHostBuffer(
      _context, indices,
      VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
      indexBuffer);

  VkAccelerationStructureGeometryKHR geometry{
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
      .geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
      .geometry = {.triangles =
                       {
                           .sType =
                               VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
                           .vertexFormat = VK_FORMAT_R32G32B32_SFLOAT,
                           .vertexData = {.deviceAddress =
                                              _context.bufferAddress(positions)},
                           .vertexStride = sizeof(float) * 3,
                           .maxVertex = vertexCount - 1,
                           .indexType = VK_INDEX_TYPE_UINT32,
                           .indexData = {.deviceAddress = _context.bufferAddress(
                                             indexBuffer.buffer)},
                       }},
      // alpha-tested materials are treated as opaque for now
      .flags = VK_GEOMETRY_OPAQUE_BIT_KHR,
  };
  const uint32_t triangleCount = static_cast<uint32_t>(indices.size() / 3);

  VkAccelerationStructureBuildGeometryInfoKHR buildInfo{
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
      .type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
      .flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
               VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR,
      .mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
      .geometryCount = 1,
      .pGeometries = &geometry,
  };
  VkAccelerationStructureBuildSizesInfoKHR sizes{
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR,
  };
  rt.getAccelerationStructureBuildSizes(
      _context.device(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
      &buildInfo, &triangleCount, &sizes);

  AccelerationStructure uncompacted = createAccelerationStructure(
      VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
      sizes.accelerationStructureSize);
  Buffer<uint8_t> scratch;
  _context.createBuffer(sizes.buildScratchSize,
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, scratch,
                        _context.accelerationStructureScratchAlignment());
  buildInfo.dstAccelerationStructure = uncompacted.handle;
  buildInfo.scratchData.deviceAddress = _context.bufferAddress(scratch.buffer);

  VkQueryPoolCreateInfo queryPoolInfo{
      .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
      .queryType = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR,
      .queryCount = 1,
  };
  VkQueryPool queryPool = VK_NULL_HANDLE;
  vkCreateQueryPool(_context.device(), &queryPoolInfo, nullptr, &queryPool);

  const VkAccelerationStructureBuildRangeInfoKHR range{
      .primitiveCount = triangleCount,
  };
  const VkAccelerationStructureBuildRangeInfoKHR *ranges[] = {&range};
  _context.immediateSubmit([&](VkCommandBuffer commandBuffer) {
    vkCmdResetQueryPool(commandBuffer, queryPool, 0, 1);
    rt.cmdBuildAccelerationStructures(commandBuffer, 1, &buildInfo, ranges);
    accelerationStructureBarrier(commandBuffer);
    rt.cmdWriteAccelerationStructuresProperties(
        commandBuffer, 1, &uncompacted.handle,
        VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR, queryPool, 0);
  });

  VkDeviceSize compactedSize = 0;
  vkGetQueryPoolResults(_context.device(), queryPool, 0, 1,
                        sizeof(compactedSize), &compactedSize,
                        sizeof(compactedSize),
                        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
  vkDestroyQueryPool(_context.device(), queryPool, nullptr);

  AccelerationStructure compacted = createAccelerationStructure(
      VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, compactedSize);
  _context.immediateSubmit([&](VkCommandBuffer commandBuffer) {
    VkCopyAccelerationStructureInfoKHR copyInfo{
        .sType = VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_INFO_KHR,
        .src = uncompacted.handle,
        .dst = compacted.handle,
        .mode = VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR,
    };
    rt.cmdCopyAccelerationStructure(commandBuffer, &copyInfo);
  });

  destroyAccelerationStructure(uncompacted);
  _context.destroyBuffer(scratch);
  _context.destroyBuffer(indexBuffer);
  _blases.push_back(compacted);
}

void RayTracingScene::buildTlas(
    std::vector<VkAccelerationStructureInstanceKHR> instances) {
  assert(!instances.empty());
  const RayTracingFunctions &rt = _context.rayTracing();
  for (VkAccelerationStructureInstanceKHR &instance : instances) {
    const uint64_t meshIndex = instance.accelerationStructureReference;
    assert(meshIndex < _blases.size() && "instance references unknown mesh");
    instance.accelerationStructureReference = _blases[meshIndex].address;
  }

  _context.waitIdle();
  destroyAccelerationStructure(_tlas);

  Buffer<VkAccelerationStructureInstanceKHR> instanceBuffer;
  createHostBuffer(
      _context, instances,
      VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
      instanceBuffer);

  VkAccelerationStructureGeometryKHR geometry{
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
      .geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR,
      .geometry = {.instances =
                       {
                           .sType =
                               VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
                           .data = {.deviceAddress = _context.bufferAddress(
                                        instanceBuffer.buffer)},
                       }},
  };
  VkAccelerationStructureBuildGeometryInfoKHR buildInfo{
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
      .type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
      .flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
      .mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
      .geometryCount = 1,
      .pGeometries = &geometry,
  };
  const uint32_t instanceCount = static_cast<uint32_t>(instances.size());
  VkAccelerationStructureBuildSizesInfoKHR sizes{
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR,
  };
  rt.getAccelerationStructureBuildSizes(
      _context.device(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
      &buildInfo, &instanceCount, &sizes);

  _tlas = createAccelerationStructure(
      VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
      sizes.accelerationStructureSize);
  Buffer<uint8_t> scratch;
  _context.createBuffer(sizes.buildScratchSize,
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, scratch,
                        _context.accelerationStructureScratchAlignment());
  buildInfo.dstAccelerationStructure = _tlas.handle;
  buildInfo.scratchData.deviceAddress = _context.bufferAddress(scratch.buffer);

  const VkAccelerationStructureBuildRangeInfoKHR range{
      .primitiveCount = instanceCount,
  };
  const VkAccelerationStructureBuildRangeInfoKHR *ranges[] = {&range};
  _context.immediateSubmit([&](VkCommandBuffer commandBuffer) {
    rt.cmdBuildAccelerationStructures(commandBuffer, 1, &buildInfo, ranges);
  });

  _context.destroyBuffer(scratch);
  _context.destroyBuffer(instanceBuffer);
}

void RayTracingScene::destroy() {
  if (_context.device() == VK_NULL_HANDLE) {
    return;
  }
  destroyAccelerationStructure(_tlas);
  for (AccelerationStructure &blas : _blases) {
    destroyAccelerationStructure(blas);
  }
  _blases.clear();
}
