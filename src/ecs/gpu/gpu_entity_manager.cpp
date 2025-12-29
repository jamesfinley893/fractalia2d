#include "gpu_entity_manager.h"
#include "../../vulkan/core/vulkan_context.h"
#include "../../vulkan/core/vulkan_sync.h"
#include "../../vulkan/resources/core/resource_coordinator.h"
#include "../../vulkan/core/vulkan_function_loader.h"
#include "../../vulkan/core/vulkan_utils.h"
#include <iostream>
#include <cstring>
#include <random>
#include <array>

// Static RNG for performance - initialized once per thread
thread_local std::mt19937 rng{std::random_device{}()};
thread_local std::uniform_real_distribution<float> stateTimerDist{0.0f, 600.0f};

void GPUEntitySoA::addFromECS(const Transform& transform, const Renderable& renderable, const MovementPattern& pattern) {
    // Velocity (initialized to zero, set by compute shader)
    velocities.emplace_back(
        0.0f,                      // velocity.x
        0.0f,                      // velocity.y  
        0.001f,                    // damping factor
        0.0f                       // reserved
    );
    
    // Movement parameters
    movementParams.emplace_back(
        pattern.amplitude,
        pattern.frequency, 
        pattern.phase,
        pattern.timeOffset
    );
    
    // Runtime state
    runtimeStates.emplace_back(
        0.0f,                          // totalTime (updated by compute shader)
        0.0f,                          // reserved 
        stateTimerDist(rng),           // stateTimer (random staggering)
        0.0f                           // initialized flag (starts as 0.0)
    );
    
    // Color
    colors.emplace_back(renderable.color);
    
    // Model matrix  
    modelMatrices.emplace_back(transform.getMatrix());

    // Default control params: no control, unit scale
    controlParams.emplace_back(0.0f, 0.0f, 0.0f, 1.0f);
}


GPUEntityManager::GPUEntityManager() {
}

GPUEntityManager::~GPUEntityManager() {
    cleanup();
}

bool GPUEntityManager::initialize(const VulkanContext& context, VulkanSync* sync, ResourceCoordinator* resourceCoordinator) {
    this->context = &context;
    this->sync = sync;
    this->resourceCoordinator = resourceCoordinator;
    
    // Initialize buffer manager
    if (!bufferManager.initialize(context, resourceCoordinator, MAX_ENTITIES)) {
        std::cerr << "GPUEntityManager: Failed to initialize buffer manager" << std::endl;
        return false;
    }
    
    // Initialize base descriptor manager functionality
    if (!descriptorManager.initialize(context)) {
        std::cerr << "GPUEntityManager: Failed to initialize base descriptor manager" << std::endl;
        return false;
    }
    
    // Initialize entity-specific descriptor manager functionality
    if (!descriptorManager.initializeEntity(bufferManager, resourceCoordinator)) {
        std::cerr << "GPUEntityManager: Failed to initialize entity descriptor manager" << std::endl;
        return false;
    }
    
    std::cout << "GPUEntityManager: Initialized successfully with descriptor manager" << std::endl;
    return true;
}

void GPUEntityManager::cleanup() {
    if (!context) return;
    
    // Cleanup descriptor manager first
    descriptorManager.cleanup();
    
    // Cleanup buffer manager
    bufferManager.cleanup();
}


void GPUEntityManager::addEntitiesFromECS(const std::vector<flecs::entity>& entities) {
    for (const auto& entity : entities) {
        if (activeEntityCount + stagingEntities.size() >= MAX_ENTITIES) {
            std::cerr << "GPUEntityManager: Reached max capacity, stopping entity addition" << std::endl;
            break;
        }
        
        // Get components from entity using .get<>()
        const Transform* transform = entity.get<Transform>();
        const Renderable* renderable = entity.get<Renderable>();
        const MovementPattern* movement = entity.get<MovementPattern>();
        
        if (transform && renderable && movement) {
            stagingEntities.addFromECS(*transform, *renderable, *movement);
            
            // Store mapping from GPU buffer index to ECS entity ID for debugging
            uint32_t gpuIndex = activeEntityCount + stagingEntities.size() - 1;
            if (gpuIndex >= gpuIndexToECSEntity.size()) {
                gpuIndexToECSEntity.resize(gpuIndex + 1);
            }
            gpuIndexToECSEntity[gpuIndex] = entity;

            if (entity.has<Player>()) {
                stagingEntities.velocities.back().w = 1.0f; // Mark as manually controlled
                float renderScale = 1.8f;
                if (const auto* control = entity.get<PlayerControl>()) {
                    renderScale = control->renderScale;
                }
                stagingEntities.controlParams.back() = glm::vec4(0.0f, 0.0f, 1.0f, renderScale);
            }
            entity.set<GPUIndex>({gpuIndex});
        }
    }
}

void GPUEntityManager::uploadPendingEntities() {
    if (stagingEntities.empty()) return;
    
    std::cout << "GPUEntityManager: WARNING - Uploading entities during runtime! This will overwrite computed positions!" << std::endl;
    
    size_t entityCount = stagingEntities.size();
    
    // Upload each SoA buffer separately
    VkDeviceSize velocityOffset = activeEntityCount * sizeof(glm::vec4);
    VkDeviceSize movementParamsOffset = activeEntityCount * sizeof(glm::vec4);
    VkDeviceSize runtimeStateOffset = activeEntityCount * sizeof(glm::vec4);
    VkDeviceSize colorOffset = activeEntityCount * sizeof(glm::vec4);
    VkDeviceSize modelMatrixOffset = activeEntityCount * sizeof(glm::mat4);
    VkDeviceSize controlParamsOffset = activeEntityCount * sizeof(glm::vec4);
    
    VkDeviceSize velocitySize = entityCount * sizeof(glm::vec4);
    VkDeviceSize movementParamsSize = entityCount * sizeof(glm::vec4);
    VkDeviceSize runtimeStateSize = entityCount * sizeof(glm::vec4);
    VkDeviceSize colorSize = entityCount * sizeof(glm::vec4);
    VkDeviceSize modelMatrixSize = entityCount * sizeof(glm::mat4);
    VkDeviceSize controlParamsSize = entityCount * sizeof(glm::vec4);
    
    // Copy SoA data to GPU buffers using new typed upload methods
    bufferManager.uploadVelocityData(stagingEntities.velocities.data(), velocitySize, velocityOffset);
    bufferManager.uploadMovementParamsData(stagingEntities.movementParams.data(), movementParamsSize, movementParamsOffset);
    bufferManager.uploadRuntimeStateData(stagingEntities.runtimeStates.data(), runtimeStateSize, runtimeStateOffset);
    bufferManager.uploadColorData(stagingEntities.colors.data(), colorSize, colorOffset);
    bufferManager.uploadModelMatrixData(stagingEntities.modelMatrices.data(), modelMatrixSize, modelMatrixOffset);
    bufferManager.uploadControlParamsData(stagingEntities.controlParams.data(), controlParamsSize, controlParamsOffset);
    
    // Initialize position buffers with spawn positions
    std::vector<glm::vec4> initialPositions;
    initialPositions.reserve(entityCount);
    
    for (size_t i = 0; i < stagingEntities.modelMatrices.size(); ++i) {
        const auto& modelMatrix = stagingEntities.modelMatrices[i];
        // Extract position from modelMatrix (4th column contains translation)
        glm::vec3 spawnPosition = glm::vec3(modelMatrix[3]);
        initialPositions.emplace_back(spawnPosition, 1.0f);
        
        // Debug first few positions to verify data
        if (i < 5) {
            std::cout << "Entity " << i << " spawn position: (" 
                      << spawnPosition.x << ", " << spawnPosition.y << ", " << spawnPosition.z << ")" << std::endl;
        }
    }
    
    // Initialize ALL position buffers so graphics and physics can read from any of them
    VkDeviceSize positionUploadSize = initialPositions.size() * sizeof(glm::vec4);
    VkDeviceSize positionOffset = activeEntityCount * sizeof(glm::vec4);
    
    bufferManager.uploadPositionDataToAllBuffers(initialPositions.data(), positionUploadSize, positionOffset);
    
    activeEntityCount += entityCount;
    stagingEntities.clear();
    
    std::cout << "GPUEntityManager: Uploaded " << entityCount << " entities to GPU-local memory (SoA), total: " << activeEntityCount << std::endl;
}

void GPUEntityManager::clearAllEntities() {
    stagingEntities.clear();
    activeEntityCount = 0;
}

// Core entity logic now clearly visible - descriptor management delegated to EntityDescriptorManager

flecs::entity GPUEntityManager::getECSEntityFromGPUIndex(uint32_t gpuIndex) const {
    if (gpuIndex < gpuIndexToECSEntity.size()) {
        return gpuIndexToECSEntity[gpuIndex];
    }
    return flecs::entity{}; // Invalid entity
}

bool GPUEntityManager::updateVelocityForEntity(uint32_t gpuIndex, const glm::vec2& velocity, float damping, bool manualControl) {
    if (gpuIndex >= activeEntityCount) {
        return false;
    }

    glm::vec4 velocityData{velocity.x, velocity.y, damping, manualControl ? 1.0f : 0.0f};
    VkDeviceSize offset = static_cast<VkDeviceSize>(gpuIndex) * sizeof(glm::vec4);
    return bufferManager.uploadVelocityData(&velocityData, sizeof(velocityData), offset);
}

bool GPUEntityManager::updateMovementParamsForEntity(uint32_t gpuIndex, const glm::vec4& params) {
    if (gpuIndex >= activeEntityCount) {
        return false;
    }

    VkDeviceSize offset = static_cast<VkDeviceSize>(gpuIndex) * sizeof(glm::vec4);
    return bufferManager.uploadMovementParamsData(&params, sizeof(params), offset);
}

bool GPUEntityManager::updateRuntimeStateForEntity(uint32_t gpuIndex, const glm::vec4& state) {
    if (gpuIndex >= activeEntityCount) {
        return false;
    }

    VkDeviceSize offset = static_cast<VkDeviceSize>(gpuIndex) * sizeof(glm::vec4);
    return bufferManager.uploadRuntimeStateData(&state, sizeof(state), offset);
}

bool GPUEntityManager::updateControlParamsForEntity(uint32_t gpuIndex, const glm::vec4& params) {
    if (gpuIndex >= activeEntityCount) {
        return false;
    }

    VkDeviceSize offset = static_cast<VkDeviceSize>(gpuIndex) * sizeof(glm::vec4);
    return bufferManager.uploadControlParamsData(&params, sizeof(params), offset);
}
