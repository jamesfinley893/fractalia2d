#pragma once

#include "../components/component.h"
#include "../components/entity.h"
#include "entity_buffer_manager.h"
#include "entity_descriptor_manager.h"
#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <vector>
#include <memory>

// Forward declarations
class VulkanContext;
class VulkanSync;
class ResourceContext;

// Structure of Arrays (SoA) for GPU entities - better cache locality and vectorization
struct GPUEntitySoA {
    std::vector<glm::vec4> velocities;        // velocity.xy, damping, reserved
    std::vector<glm::vec4> movementParams;    // amplitude, frequency, phase, timeOffset
    std::vector<glm::vec4> runtimeStates;     // totalTime, initialized, stateTimer, entityState
    std::vector<glm::vec4> colors;            // RGBA color
    std::vector<glm::mat4> modelMatrices;     // transform matrices (cold data)
    std::vector<glm::vec4> controlParams;     // desiredVel.xy, controlFlag, renderScale
    
    void reserve(size_t capacity) {
        velocities.reserve(capacity);
        movementParams.reserve(capacity);
        runtimeStates.reserve(capacity);
        colors.reserve(capacity);
        modelMatrices.reserve(capacity);
        controlParams.reserve(capacity);
    }
    
    void clear() {
        velocities.clear();
        movementParams.clear();
        runtimeStates.clear();
        colors.clear();
        modelMatrices.clear();
        controlParams.clear();
    }
    
    size_t size() const { return velocities.size(); }
    bool empty() const { return velocities.empty(); }
    
    // Add entity from ECS components
    void addFromECS(const Transform& transform, const Renderable& renderable, const MovementPattern& pattern);
};


// Modular GPU Entity Manager for AAA Frame Graph Architecture
class GPUEntityManager {
public:
    GPUEntityManager();
    ~GPUEntityManager();

    bool initialize(const VulkanContext& context, VulkanSync* sync, ResourceCoordinator* resourceCoordinator);
    void cleanup();
    
    // Entity management - SoA approach
    void addEntitiesFromECS(const std::vector<flecs::entity>& entities);
    void uploadPendingEntities(); // Upload staged entities to GPU
    void clearAllEntities();
    
    
    // Direct buffer access for frame graph - SoA buffers
    VkBuffer getVelocityBuffer() const { return bufferManager.getVelocityBuffer(); }
    VkBuffer getMovementParamsBuffer() const { return bufferManager.getMovementParamsBuffer(); }
    VkBuffer getRuntimeStateBuffer() const { return bufferManager.getRuntimeStateBuffer(); }
    VkBuffer getColorBuffer() const { return bufferManager.getColorBuffer(); }
    VkBuffer getModelMatrixBuffer() const { return bufferManager.getModelMatrixBuffer(); }
    VkBuffer getSpatialMapBuffer() const { return bufferManager.getSpatialMapBuffer(); }
    VkBuffer getControlParamsBuffer() const { return bufferManager.getControlParamsBuffer(); }
    VkBuffer getSpatialNextBuffer() const { return bufferManager.getSpatialNextBuffer(); }
    
    // Position buffers remain the same
    VkBuffer getPositionBuffer() const { return bufferManager.getPositionBuffer(); }
    VkBuffer getPositionBufferAlternate() const { return bufferManager.getPositionBufferAlternate(); }
    VkBuffer getCurrentPositionBuffer() const { return bufferManager.getCurrentPositionBuffer(); }
    VkBuffer getTargetPositionBuffer() const { return bufferManager.getTargetPositionBuffer(); }
    
    
    // Async compute support - ping-pong between position buffers
    VkBuffer getComputeWriteBuffer(uint32_t frameIndex) const { return bufferManager.getComputeWriteBuffer(frameIndex); }
    VkBuffer getGraphicsReadBuffer(uint32_t frameIndex) const { return bufferManager.getGraphicsReadBuffer(frameIndex); }
    
    // Buffer properties - SoA approach
    VkDeviceSize getVelocityBufferSize() const { return bufferManager.getVelocityBufferSize(); }
    VkDeviceSize getMovementParamsBufferSize() const { return bufferManager.getMovementParamsBufferSize(); }
    VkDeviceSize getRuntimeStateBufferSize() const { return bufferManager.getRuntimeStateBufferSize(); }
    VkDeviceSize getColorBufferSize() const { return bufferManager.getColorBufferSize(); }
    VkDeviceSize getModelMatrixBufferSize() const { return bufferManager.getModelMatrixBufferSize(); }
    VkDeviceSize getSpatialMapBufferSize() const { return bufferManager.getSpatialMapBufferSize(); }
    VkDeviceSize getControlParamsBufferSize() const { return bufferManager.getControlParamsBufferSize(); }
    VkDeviceSize getSpatialNextBufferSize() const { return bufferManager.getSpatialNextBufferSize(); }
    VkDeviceSize getPositionBufferSize() const { return bufferManager.getPositionBufferSize(); }
    
    
    // Entity state
    uint32_t getEntityCount() const { return activeEntityCount; }
    uint32_t getMaxEntities() const { return bufferManager.getMaxEntities(); }
    bool hasPendingUploads() const { return !stagingEntities.empty(); }
    
    // Descriptor management delegation
    EntityDescriptorManager& getDescriptorManager() { return descriptorManager; }
    const EntityDescriptorManager& getDescriptorManager() const { return descriptorManager; }
    
    // Debug access to buffer manager for spatial map readback
    const EntityBufferManager& getBufferManager() const { return bufferManager; }
    
    // Debug: Get ECS entity ID from GPU buffer index
    flecs::entity getECSEntityFromGPUIndex(uint32_t gpuIndex) const;

    // Update velocity for a specific GPU entity
    bool updateVelocityForEntity(uint32_t gpuIndex, const glm::vec2& velocity, float damping, bool manualControl);

    // Update movement params for a specific GPU entity
    bool updateMovementParamsForEntity(uint32_t gpuIndex, const glm::vec4& params);

    // Update runtime state for a specific GPU entity
    bool updateRuntimeStateForEntity(uint32_t gpuIndex, const glm::vec4& state);
    bool updateControlParamsForEntity(uint32_t gpuIndex, const glm::vec4& params);

private:
    static constexpr uint32_t MAX_ENTITIES = 131072; // 128k entities max
    
    // Dependencies
    const VulkanContext* context = nullptr;
    VulkanSync* sync = nullptr;
    ResourceCoordinator* resourceCoordinator = nullptr;
    
    // Core components
    EntityBufferManager bufferManager;
    EntityDescriptorManager descriptorManager;
    
    // Staging data - SoA approach
    GPUEntitySoA stagingEntities;
    uint32_t activeEntityCount = 0;
    
    // Debug: Mapping from GPU buffer index to ECS entity ID
    std::vector<flecs::entity> gpuIndexToECSEntity;
};
