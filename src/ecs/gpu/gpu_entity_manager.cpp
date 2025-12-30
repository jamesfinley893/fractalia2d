#include "gpu_entity_manager.h"
#include "../../vulkan/core/vulkan_context.h"
#include "../../vulkan/core/vulkan_sync.h"
#include "../../vulkan/resources/core/resource_coordinator.h"
#include "../../vulkan/core/vulkan_function_loader.h"
#include "../../vulkan/core/vulkan_utils.h"
#include "soft_body_constants.h"
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
    stagingFEM.reserve(stagingEntities.size() + entities.size());
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
                float renderScale = std::max(transform->scale.x, transform->scale.y);
                stagingEntities.controlParams.back() = glm::vec4(0.0f, 0.0f, 1.0f, renderScale);
            }
            entity.set<GPUIndex>({gpuIndex});

            // Build FEM grid (3x3 nodes, 8 triangles)
            constexpr int gridSize = 3;
            constexpr float halfExtent = 1.0f;
            constexpr float spacing = (halfExtent * 2.0f) / float(gridSize - 1);
            glm::vec3 localVerts[SoftBodyConstants::kParticlesPerBody];
            uint32_t v = 0;
            for (int y = 0; y < gridSize; ++y) {
                for (int x = 0; x < gridSize; ++x) {
                    float lx = -halfExtent + spacing * float(x);
                    float ly = -halfExtent + spacing * float(y);
                    localVerts[v++] = glm::vec3(lx, ly, 0.0f);
                }
            }

            glm::mat4 model = transform->getMatrix();
            glm::vec3 worldVerts[SoftBodyConstants::kParticlesPerBody];
            for (uint32_t i = 0; i < SoftBodyConstants::kParticlesPerBody; ++i) {
                worldVerts[i] = glm::vec3(model * glm::vec4(localVerts[i], 1.0f));
            }

            uint32_t nodeOffset = gpuIndex * SoftBodyConstants::kParticlesPerBody;
            uint32_t triOffset = gpuIndex * SoftBodyConstants::kTrianglesPerBody;
            for (uint32_t i = 0; i < SoftBodyConstants::kParticlesPerBody; ++i) {
                stagingFEM.nodePositions.emplace_back(worldVerts[i], 1.0f);
                stagingFEM.nodeVelocities.emplace_back(0.0f, 0.0f, 0.0f, 0.0f);
                stagingFEM.nodeRestPositions.emplace_back(worldVerts[i], 1.0f);
            }

            float restArea = 0.0f;
            for (int y = 0; y < gridSize - 1; ++y) {
                for (int x = 0; x < gridSize - 1; ++x) {
                    uint32_t i0 = nodeOffset + y * gridSize + x;
                    uint32_t i1 = nodeOffset + y * gridSize + (x + 1);
                    uint32_t i2 = nodeOffset + (y + 1) * gridSize + x;
                    uint32_t i3 = nodeOffset + (y + 1) * gridSize + (x + 1);

                    stagingFEM.triIndices.emplace_back(i0, i1, i2, 0);
                    stagingFEM.triIndices.emplace_back(i1, i3, i2, 0);

                    glm::vec2 a = glm::vec2(worldVerts[y * gridSize + x]);
                    glm::vec2 b = glm::vec2(worldVerts[y * gridSize + (x + 1)]);
                    glm::vec2 c = glm::vec2(worldVerts[(y + 1) * gridSize + x]);
                    glm::vec2 d = glm::vec2(worldVerts[(y + 1) * gridSize + (x + 1)]);

                    restArea += 0.5f * std::abs((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x));
                    restArea += 0.5f * std::abs((d.x - b.x) * (c.y - b.y) - (d.y - b.y) * (c.x - b.x));
                }
            }
            float nodeMass = (restArea > 0.0f) ? (restArea / float(SoftBodyConstants::kParticlesPerBody)) : 1.0f;
            float invMass = (nodeMass > 0.0f) ? (1.0f / nodeMass) : 0.0f;
            for (uint32_t i = 0; i < SoftBodyConstants::kParticlesPerBody; ++i) {
                stagingFEM.nodeInvMass.emplace_back(invMass);
            }

            stagingFEM.bodyData.emplace_back(
                nodeOffset,
                SoftBodyConstants::kParticlesPerBody,
                triOffset,
                SoftBodyConstants::kTrianglesPerBody
            );
            stagingFEM.bodyParams.emplace_back(
                6.0f,   // stiffness
                0.002f, // compliance (softer deformation)
                0.55f,  // restitution
                0.4f    // friction
            );
            stagingFEM.triRestData.emplace_back(0.0f);
            stagingFEM.triRestArea.emplace_back(restArea);
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
    
    // Initialize FEM node buffers
    VkDeviceSize nodeOffset = activeEntityCount * SoftBodyConstants::kParticlesPerBody * sizeof(glm::vec4);
    VkDeviceSize nodeCount = stagingFEM.nodePositions.size();
    VkDeviceSize nodeSize = nodeCount * sizeof(glm::vec4);

    bufferManager.uploadPositionDataToAllBuffers(stagingFEM.nodePositions.data(), nodeSize, nodeOffset);
    bufferManager.uploadNodeVelocityData(stagingFEM.nodeVelocities.data(), nodeCount * sizeof(glm::vec4), nodeOffset);
    bufferManager.uploadNodeInvMassData(stagingFEM.nodeInvMass.data(), nodeCount * sizeof(float), activeEntityCount * SoftBodyConstants::kParticlesPerBody * sizeof(float));
    bufferManager.uploadNodeRestData(stagingFEM.nodeRestPositions.data(), nodeCount * sizeof(glm::vec4), nodeOffset);

    VkDeviceSize bodyOffset = activeEntityCount * sizeof(glm::uvec4);
    VkDeviceSize bodyParamsOffset = activeEntityCount * sizeof(glm::vec4);
    VkDeviceSize bodyCount = stagingFEM.bodyData.size();
    bufferManager.uploadBodyData(stagingFEM.bodyData.data(), bodyCount * sizeof(glm::uvec4), bodyOffset);
    bufferManager.uploadBodyParamsData(stagingFEM.bodyParams.data(), bodyCount * sizeof(glm::vec4), bodyParamsOffset);
    bufferManager.uploadTriangleRestData(stagingFEM.triRestData.data(), bodyCount * sizeof(glm::vec4), activeEntityCount * sizeof(glm::vec4));
    bufferManager.uploadTriangleAreaData(stagingFEM.triRestArea.data(), bodyCount * sizeof(float), activeEntityCount * sizeof(float));
    bufferManager.uploadTriangleIndexData(stagingFEM.triIndices.data(), stagingFEM.triIndices.size() * sizeof(glm::uvec4), activeEntityCount * SoftBodyConstants::kTrianglesPerBody * sizeof(glm::uvec4));
    
    activeEntityCount += entityCount;
    stagingEntities.clear();
    stagingFEM.clear();
    
    std::cout << "GPUEntityManager: Uploaded " << entityCount << " entities to GPU-local memory (SoA), total: " << activeEntityCount << std::endl;
}

void GPUEntityManager::clearAllEntities() {
    stagingEntities.clear();
    stagingFEM.clear();
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
