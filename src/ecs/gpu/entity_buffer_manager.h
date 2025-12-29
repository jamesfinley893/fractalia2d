#pragma once

#include "specialized_buffers.h"
#include "position_buffer_coordinator.h"
#include "buffer_upload_service.h"
#include <vulkan/vulkan.h>
#include <memory>

// Forward declarations
class VulkanContext;
class ResourceCoordinator;

/**
 * REFACTORED: Entity buffer manager using SRP-compliant specialized buffer classes
 * Single responsibility: coordinate specialized buffer components for entity rendering
 */
class EntityBufferManager {
public:
    EntityBufferManager();
    ~EntityBufferManager();

    bool initialize(const VulkanContext& context, ResourceCoordinator* resourceCoordinator, uint32_t maxEntities);
    void cleanup();
    
    // SoA buffer access - delegated to specialized buffers
    VkBuffer getVelocityBuffer() const { return velocityBuffer.getBuffer(); }
    VkBuffer getMovementParamsBuffer() const { return movementParamsBuffer.getBuffer(); }
    VkBuffer getRuntimeStateBuffer() const { return runtimeStateBuffer.getBuffer(); }
    VkBuffer getColorBuffer() const { return colorBuffer.getBuffer(); }
    VkBuffer getModelMatrixBuffer() const { return modelMatrixBuffer.getBuffer(); }
    VkBuffer getSpatialMapBuffer() const { return spatialMapBuffer.getBuffer(); }
    VkBuffer getControlParamsBuffer() const { return controlParamsBuffer.getBuffer(); }
    
    // Position buffers - delegated to coordinator
    VkBuffer getPositionBuffer() const { return positionCoordinator.getPrimaryBuffer(); }
    VkBuffer getPositionBufferAlternate() const { return positionCoordinator.getAlternateBuffer(); }
    VkBuffer getCurrentPositionBuffer() const { return positionCoordinator.getCurrentBuffer(); }
    VkBuffer getTargetPositionBuffer() const { return positionCoordinator.getTargetBuffer(); }
    
    
    // Ping-pong buffer access - delegated to coordinator
    VkBuffer getComputeWriteBuffer(uint32_t frameIndex) const { return positionCoordinator.getComputeWriteBuffer(frameIndex); }
    VkBuffer getGraphicsReadBuffer(uint32_t frameIndex) const { return positionCoordinator.getGraphicsReadBuffer(frameIndex); }
    
    // Buffer properties - delegated to specialized buffers
    VkDeviceSize getVelocityBufferSize() const { return velocityBuffer.getSize(); }
    VkDeviceSize getMovementParamsBufferSize() const { return movementParamsBuffer.getSize(); }
    VkDeviceSize getRuntimeStateBufferSize() const { return runtimeStateBuffer.getSize(); }
    VkDeviceSize getColorBufferSize() const { return colorBuffer.getSize(); }
    VkDeviceSize getModelMatrixBufferSize() const { return modelMatrixBuffer.getSize(); }
    VkDeviceSize getSpatialMapBufferSize() const { return spatialMapBuffer.getSize(); }
    VkDeviceSize getControlParamsBufferSize() const { return controlParamsBuffer.getSize(); }
    VkDeviceSize getPositionBufferSize() const { return positionCoordinator.getBufferSize(); }
    uint32_t getMaxEntities() const { return maxEntities; }
    
    
    // Data upload - using shared upload service
    
    // Typed upload methods for better API
    bool uploadVelocityData(const void* data, VkDeviceSize size, VkDeviceSize offset = 0);
    bool uploadMovementParamsData(const void* data, VkDeviceSize size, VkDeviceSize offset = 0);
    bool uploadRuntimeStateData(const void* data, VkDeviceSize size, VkDeviceSize offset = 0);
    bool uploadColorData(const void* data, VkDeviceSize size, VkDeviceSize offset = 0);
    bool uploadModelMatrixData(const void* data, VkDeviceSize size, VkDeviceSize offset = 0);
    bool uploadSpatialMapData(const void* data, VkDeviceSize size, VkDeviceSize offset = 0);
    bool uploadControlParamsData(const void* data, VkDeviceSize size, VkDeviceSize offset = 0);
    bool uploadPositionDataToAllBuffers(const void* data, VkDeviceSize size, VkDeviceSize offset = 0);
    
    // Debug readback methods (expensive - use sparingly)
    struct EntityDebugInfo {
        glm::vec4 position;
        glm::vec4 velocity;
        uint32_t spatialCell;
        uint32_t entityId;
    };
    
    bool readbackEntityAtPosition(glm::vec2 worldPos, EntityDebugInfo& info) const;
    bool readbackEntityById(uint32_t entityId, EntityDebugInfo& info) const;
    bool readbackSpatialCell(uint32_t cellIndex, std::vector<uint32_t>& entityIds) const;
    
    // GPU-synchronized readback (waits for compute shader completion)
    bool readbackEntityAtPositionSafe(glm::vec2 worldPos, EntityDebugInfo& info) const;

private:
    // Configuration
    uint32_t maxEntities = 0;
    const VulkanContext* context = nullptr;
    
    // Helper method for GPU readback
    bool readGPUBuffer(VkBuffer srcBuffer, void* dstData, VkDeviceSize size, VkDeviceSize offset) const;
    
    // Initialize spatial map with NULL values
    bool initializeSpatialMapBuffer();
    
    // Specialized buffer components (SRP-compliant)
    VelocityBuffer velocityBuffer;
    MovementParamsBuffer movementParamsBuffer;
    RuntimeStateBuffer runtimeStateBuffer;
    ColorBuffer colorBuffer;
    ModelMatrixBuffer modelMatrixBuffer;
    SpatialMapBuffer spatialMapBuffer;
    ControlParamsBuffer controlParamsBuffer;
    
    // Position buffer coordination
    PositionBufferCoordinator positionCoordinator;
    
    // Shared upload service
    BufferUploadService uploadService;
    
};

