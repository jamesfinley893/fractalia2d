#include "entity_buffer_manager.h"
#include "../../vulkan/core/vulkan_context.h"
#include "../../vulkan/resources/core/resource_coordinator.h"
#include "../../vulkan/resources/core/command_executor.h"
#include "../../vulkan/core/vulkan_function_loader.h"
#include <iostream>
#include <cstring>
#include <limits>
#include <algorithm>

EntityBufferManager::EntityBufferManager() {
}

EntityBufferManager::~EntityBufferManager() {
    cleanup();
}

bool EntityBufferManager::initialize(const VulkanContext& context, ResourceCoordinator* resourceCoordinator, uint32_t maxEntities) {
    this->maxEntities = maxEntities;
    this->context = &context;
    maxNodes = maxEntities * SoftBodyConstants::kParticlesPerBody;
    maxTriangles = maxEntities * SoftBodyConstants::kTrianglesPerBody;
    
    // Initialize upload service
    if (!uploadService.initialize(resourceCoordinator)) {
        std::cerr << "EntityBufferManager: Failed to initialize upload service" << std::endl;
        return false;
    }
    
    // Initialize specialized buffers
    if (!velocityBuffer.initialize(context, resourceCoordinator, maxEntities)) {
        std::cerr << "EntityBufferManager: Failed to initialize velocity buffer" << std::endl;
        return false;
    }
    
    if (!movementParamsBuffer.initialize(context, resourceCoordinator, maxEntities)) {
        std::cerr << "EntityBufferManager: Failed to initialize movement params buffer" << std::endl;
        return false;
    }
    
    if (!runtimeStateBuffer.initialize(context, resourceCoordinator, maxEntities)) {
        std::cerr << "EntityBufferManager: Failed to initialize runtime state buffer" << std::endl;
        return false;
    }
    
    if (!colorBuffer.initialize(context, resourceCoordinator, maxEntities)) {
        std::cerr << "EntityBufferManager: Failed to initialize color buffer" << std::endl;
        return false;
    }
    
    if (!modelMatrixBuffer.initialize(context, resourceCoordinator, maxEntities)) {
        std::cerr << "EntityBufferManager: Failed to initialize model matrix buffer" << std::endl;
        return false;
    }
    
    if (!spatialMapBuffer.initialize(context, resourceCoordinator, 4096)) { // 64x64 grid
        std::cerr << "EntityBufferManager: Failed to initialize spatial map buffer" << std::endl;
        return false;
    }
    
    // Initialize spatial map buffer with NULL values (0xFFFFFFFF)
    if (!initializeSpatialMapBuffer()) {
        std::cerr << "EntityBufferManager: Failed to clear spatial map buffer" << std::endl;
        return false;
    }
    
    // Initialize position buffer coordinator for node positions
    if (!positionCoordinator.initialize(context, resourceCoordinator, maxNodes)) {
        std::cerr << "EntityBufferManager: Failed to initialize position coordinator" << std::endl;
        return false;
    }

    if (!controlParamsBuffer.initialize(context, resourceCoordinator, maxEntities)) {
        std::cerr << "EntityBufferManager: Failed to initialize control params buffer" << std::endl;
        return false;
    }

    if (!spatialNextBuffer.initialize(context, resourceCoordinator, maxEntities)) {
        std::cerr << "EntityBufferManager: Failed to initialize spatial next buffer" << std::endl;
        return false;
    }
    
    if (!nodeVelocityBuffer.initialize(context, resourceCoordinator, maxNodes)) {
        std::cerr << "EntityBufferManager: Failed to initialize node velocity buffer" << std::endl;
        return false;
    }

    if (!nodeInvMassBuffer.initialize(context, resourceCoordinator, maxNodes)) {
        std::cerr << "EntityBufferManager: Failed to initialize node inv mass buffer" << std::endl;
        return false;
    }

    if (!bodyDataBuffer.initialize(context, resourceCoordinator, maxEntities)) {
        std::cerr << "EntityBufferManager: Failed to initialize body data buffer" << std::endl;
        return false;
    }

    if (!bodyParamsBuffer.initialize(context, resourceCoordinator, maxEntities)) {
        std::cerr << "EntityBufferManager: Failed to initialize body params buffer" << std::endl;
        return false;
    }

    if (!triangleRestBuffer.initialize(context, resourceCoordinator, maxEntities)) {
        std::cerr << "EntityBufferManager: Failed to initialize triangle rest buffer" << std::endl;
        return false;
    }

    if (!triangleAreaBuffer.initialize(context, resourceCoordinator, maxEntities)) {
        std::cerr << "EntityBufferManager: Failed to initialize triangle area buffer" << std::endl;
        return false;
    }

    if (!nodeForceBuffer.initialize(context, resourceCoordinator, maxNodes)) {
        std::cerr << "EntityBufferManager: Failed to initialize node force buffer" << std::endl;
        return false;
    }
    
    if (!nodeRestBuffer.initialize(context, resourceCoordinator, maxNodes)) {
        std::cerr << "EntityBufferManager: Failed to initialize node rest buffer" << std::endl;
        return false;
    }
    
    if (!triangleIndexBuffer.initialize(context, resourceCoordinator, maxTriangles)) {
        std::cerr << "EntityBufferManager: Failed to initialize triangle index buffer" << std::endl;
        return false;
    }
    
    std::cout << "EntityBufferManager: Initialized successfully for " << maxEntities << " entities using SRP-compliant design" << std::endl;
    return true;
}

void EntityBufferManager::cleanup() {
    // Cleanup specialized components
    positionCoordinator.cleanup();
    spatialMapBuffer.cleanup();
    modelMatrixBuffer.cleanup();
    colorBuffer.cleanup();
    runtimeStateBuffer.cleanup();
    movementParamsBuffer.cleanup();
    velocityBuffer.cleanup();
    controlParamsBuffer.cleanup();
    spatialNextBuffer.cleanup();
    nodeVelocityBuffer.cleanup();
    nodeInvMassBuffer.cleanup();
    bodyDataBuffer.cleanup();
    bodyParamsBuffer.cleanup();
    triangleRestBuffer.cleanup();
    triangleAreaBuffer.cleanup();
    nodeForceBuffer.cleanup();
    nodeRestBuffer.cleanup();
    triangleIndexBuffer.cleanup();
    uploadService.cleanup();
    
    maxEntities = 0;
    maxNodes = 0;
    maxTriangles = 0;
}


bool EntityBufferManager::uploadVelocityData(const void* data, VkDeviceSize size, VkDeviceSize offset) {
    return uploadService.upload(velocityBuffer, data, size, offset);
}

bool EntityBufferManager::uploadMovementParamsData(const void* data, VkDeviceSize size, VkDeviceSize offset) {
    return uploadService.upload(movementParamsBuffer, data, size, offset);
}

bool EntityBufferManager::uploadRuntimeStateData(const void* data, VkDeviceSize size, VkDeviceSize offset) {
    return uploadService.upload(runtimeStateBuffer, data, size, offset);
}

bool EntityBufferManager::uploadColorData(const void* data, VkDeviceSize size, VkDeviceSize offset) {
    return uploadService.upload(colorBuffer, data, size, offset);
}

bool EntityBufferManager::uploadModelMatrixData(const void* data, VkDeviceSize size, VkDeviceSize offset) {
    return uploadService.upload(modelMatrixBuffer, data, size, offset);
}

bool EntityBufferManager::uploadSpatialMapData(const void* data, VkDeviceSize size, VkDeviceSize offset) {
    return uploadService.upload(spatialMapBuffer, data, size, offset);
}

bool EntityBufferManager::uploadControlParamsData(const void* data, VkDeviceSize size, VkDeviceSize offset) {
    return uploadService.upload(controlParamsBuffer, data, size, offset);
}

bool EntityBufferManager::uploadSpatialNextData(const void* data, VkDeviceSize size, VkDeviceSize offset) {
    return uploadService.upload(spatialNextBuffer, data, size, offset);
}

bool EntityBufferManager::uploadNodeVelocityData(const void* data, VkDeviceSize size, VkDeviceSize offset) {
    return uploadService.upload(nodeVelocityBuffer, data, size, offset);
}

bool EntityBufferManager::uploadNodeInvMassData(const void* data, VkDeviceSize size, VkDeviceSize offset) {
    return uploadService.upload(nodeInvMassBuffer, data, size, offset);
}

bool EntityBufferManager::uploadBodyData(const void* data, VkDeviceSize size, VkDeviceSize offset) {
    return uploadService.upload(bodyDataBuffer, data, size, offset);
}

bool EntityBufferManager::uploadBodyParamsData(const void* data, VkDeviceSize size, VkDeviceSize offset) {
    return uploadService.upload(bodyParamsBuffer, data, size, offset);
}

bool EntityBufferManager::uploadTriangleRestData(const void* data, VkDeviceSize size, VkDeviceSize offset) {
    return uploadService.upload(triangleRestBuffer, data, size, offset);
}

bool EntityBufferManager::uploadTriangleAreaData(const void* data, VkDeviceSize size, VkDeviceSize offset) {
    return uploadService.upload(triangleAreaBuffer, data, size, offset);
}

bool EntityBufferManager::uploadNodeForceData(const void* data, VkDeviceSize size, VkDeviceSize offset) {
    return uploadService.upload(nodeForceBuffer, data, size, offset);
}

bool EntityBufferManager::uploadNodeRestData(const void* data, VkDeviceSize size, VkDeviceSize offset) {
    return uploadService.upload(nodeRestBuffer, data, size, offset);
}

bool EntityBufferManager::uploadTriangleIndexData(const void* data, VkDeviceSize size, VkDeviceSize offset) {
    return uploadService.upload(triangleIndexBuffer, data, size, offset);
}

bool EntityBufferManager::uploadPositionDataToAllBuffers(const void* data, VkDeviceSize size, VkDeviceSize offset) {
    return positionCoordinator.uploadToAllBuffers(data, size, offset);
}

// Helper method to create staging buffer and read GPU data
bool EntityBufferManager::readGPUBuffer(VkBuffer srcBuffer, void* dstData, VkDeviceSize size, VkDeviceSize offset) const {
    // Access resourceCoordinator through uploadService
    auto* resourceCoordinator = uploadService.getResourceCoordinator();
    if (!resourceCoordinator) {
        return false;
    }
    
    const auto* context = resourceCoordinator->getContext();
    if (!context) {
        return false;
    }
    
    // Create staging buffer for readback
    auto stagingHandle = resourceCoordinator->createBuffer(
        size,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );
    
    if (!stagingHandle.buffer.get()) {
        std::cerr << "Failed to create staging buffer for readback" << std::endl;
        return false;
    }
    
    // Get command executor for the copy operation
    auto* commandExecutor = resourceCoordinator->getCommandExecutor();
    if (!commandExecutor) {
        resourceCoordinator->destroyResource(stagingHandle);
        return false;
    }
    
    // Use synchronous buffer copy (automatically handles command buffer creation/submission)
    commandExecutor->copyBufferToBuffer(srcBuffer, stagingHandle.buffer.get(), size, offset, 0);
    
    // Copy data from staging buffer to host memory
    if (stagingHandle.mappedData) {
        std::memcpy(dstData, stagingHandle.mappedData, size);
    } else {
        // Map, copy, unmap
        const auto& loader = context->getLoader();
        VkDevice device = context->getDevice();
        
        void* mappedData;
        VkResult result = loader.vkMapMemory(device, stagingHandle.memory.get(), 0, size, 0, &mappedData);
        if (result != VK_SUCCESS) {
            resourceCoordinator->destroyResource(stagingHandle);
            return false;
        }
        
        std::memcpy(dstData, mappedData, size);
        loader.vkUnmapMemory(device, stagingHandle.memory.get());
    }
    
    // Cleanup staging buffer
    resourceCoordinator->destroyResource(stagingHandle);
    return true;
}

// Debug readback implementations
bool EntityBufferManager::readbackEntityAtPosition(glm::vec2 worldPos, EntityDebugInfo& info) const {
    // Calculate clicked spatial cell using same logic as GPU shader
    const float CELL_SIZE = 2.0f;
    const uint32_t GRID_WIDTH = 64;
    const uint32_t GRID_HEIGHT = 64;
    
    // Convert world position to grid coordinates (same as GPU)
    glm::ivec2 gridCoord = glm::ivec2(glm::floor(worldPos / CELL_SIZE));
    uint32_t clickedX = static_cast<uint32_t>(gridCoord.x) & (GRID_WIDTH - 1);
    uint32_t clickedY = static_cast<uint32_t>(gridCoord.y) & (GRID_HEIGHT - 1);
    uint32_t clickedCellIndex = clickedX + clickedY * GRID_WIDTH;
    
    std::cout << "=== ENTITY SEARCH DEBUG ===" << std::endl;
    std::cout << "Click position: (" << worldPos.x << ", " << worldPos.y << ")" << std::endl;
    std::cout << "CELL_SIZE: " << CELL_SIZE << ", GRID_WIDTH: " << GRID_WIDTH << ", GRID_HEIGHT: " << GRID_HEIGHT << std::endl;
    std::cout << "Raw grid coord: (" << gridCoord.x << ", " << gridCoord.y << ")" << std::endl;
    std::cout << "Wrapped grid coord: (" << clickedX << ", " << clickedY << ")" << std::endl;
    std::cout << "Clicked cell: " << clickedCellIndex << " (formula: " << clickedX << " + " << clickedY << " * " << GRID_WIDTH << ")" << std::endl;
    
    // Search nearby spatial cells for entities (much more efficient!)
    uint32_t closestEntity = 0;
    float closestDistance = std::numeric_limits<float>::max();
    glm::vec4 closestPosition;
    uint32_t cellsChecked = 0;
    uint32_t entitiesFound = 0;
    
    // Search in a 5x5 grid around the clicked cell (25 cells total)
    const int SEARCH_RADIUS = 2; // cells in each direction
    
    for (int dy = -SEARCH_RADIUS; dy <= SEARCH_RADIUS; ++dy) {
        for (int dx = -SEARCH_RADIUS; dx <= SEARCH_RADIUS; ++dx) {
            // Calculate neighboring cell coordinates
            int neighborX = (int)clickedX + dx;
            int neighborY = (int)clickedY + dy;
            
            // Wrap around grid boundaries (handle negative coordinates properly)
            uint32_t searchX = ((uint32_t)(neighborX % (int)GRID_WIDTH + GRID_WIDTH)) & (GRID_WIDTH - 1);
            uint32_t searchY = ((uint32_t)(neighborY % (int)GRID_HEIGHT + GRID_HEIGHT)) & (GRID_HEIGHT - 1);
            uint32_t searchCellIndex = searchX + searchY * GRID_WIDTH;
            
            cellsChecked++;
            
            // Check entities in this cell
            std::vector<uint32_t> entitiesInCell;
            std::cout << "Searching cell " << searchCellIndex << " (grid: " << searchX << ", " << searchY << ")" << std::endl;
            if (readbackSpatialCell(searchCellIndex, entitiesInCell)) {
                for (uint32_t entityId : entitiesInCell) {
                    if (entityId >= maxEntities) continue;
                    entitiesFound++;
                    
                    // Read position for this entity
                    glm::vec4 entityPosition;
                    if (readGPUBuffer(positionCoordinator.getPrimaryBuffer(), 
                                     &entityPosition, sizeof(glm::vec4), 
                                     entityId * sizeof(glm::vec4))) {
                        
                        // Calculate entity's spatial cell using same logic as shader
                        glm::vec2 entityPos2D = glm::vec2(entityPosition);
                        glm::ivec2 entityGridCoord = glm::ivec2(glm::floor(entityPos2D / CELL_SIZE));
                        uint32_t entityCellX = static_cast<uint32_t>(entityGridCoord.x) & (GRID_WIDTH - 1);
                        uint32_t entityCellY = static_cast<uint32_t>(entityGridCoord.y) & (GRID_HEIGHT - 1);
                        uint32_t entityActualCell = entityCellX + entityCellY * GRID_WIDTH;
                        
                        float distance = glm::distance(worldPos, glm::vec2(entityPosition));
                        std::cout << "  Entity " << entityId << " at (" << entityPosition.x << ", " << entityPosition.y 
                                  << ") in search cell " << searchCellIndex << " but actual cell " << entityActualCell 
                                  << " distance " << distance << std::endl;
                        
                        if (distance < closestDistance) {
                            closestDistance = distance;
                            closestEntity = entityId;
                            closestPosition = entityPosition;
                        }
                    }
                }
            }
        }
    }
    
    std::cout << "Cells searched: " << cellsChecked << ", Entities found: " << entitiesFound << std::endl;
    
    if (entitiesFound == 0) {
        std::cout << "No entities found in nearby cells" << std::endl;
        return false;
    }
    
    // Calculate which cell the closest entity is actually in
    glm::vec2 entityPos2D = glm::vec2(closestPosition);
    glm::ivec2 entityGridCoord = glm::ivec2(glm::floor(entityPos2D / CELL_SIZE));
    uint32_t entityX = static_cast<uint32_t>(entityGridCoord.x) & (GRID_WIDTH - 1);
    uint32_t entityY = static_cast<uint32_t>(entityGridCoord.y) & (GRID_WIDTH - 1);
    uint32_t entityCellIndex = entityX + entityY * GRID_WIDTH;
    
    // Fill in the debug info
    info.entityId = closestEntity;
    info.position = closestPosition;
    info.spatialCell = entityCellIndex;
    
    // Read velocity for the closest entity
    if (!readGPUBuffer(velocityBuffer.getBuffer(), 
                      &info.velocity, sizeof(glm::vec4), 
                      closestEntity * sizeof(glm::vec4))) {
        info.velocity = glm::vec4(0.0f); // Fallback
    }
    
    std::cout << "Closest entity: " << closestEntity << " at distance " << closestDistance << std::endl;
    std::cout << "Entity position: (" << closestPosition.x << ", " << closestPosition.y << ")" << std::endl;
    std::cout << "Entity cell: " << entityCellIndex << " (grid: " << entityX << ", " << entityY << ")" << std::endl;
    std::cout << "Cell difference: clicked=" << clickedCellIndex << ", entity=" << entityCellIndex 
              << " (diff=" << (int)entityCellIndex - (int)clickedCellIndex << ")" << std::endl;
    
    return true;
}

bool EntityBufferManager::readbackEntityById(uint32_t entityId, EntityDebugInfo& info) const {
    if (entityId >= maxEntities) {
        return false;
    }
    
    info.entityId = entityId;
    
    // Read position
    if (!readGPUBuffer(positionCoordinator.getPrimaryBuffer(), 
                      &info.position, sizeof(glm::vec4), 
                      entityId * sizeof(glm::vec4))) {
        return false;
    }
    
    // Read velocity
    if (!readGPUBuffer(velocityBuffer.getBuffer(), 
                      &info.velocity, sizeof(glm::vec4), 
                      entityId * sizeof(glm::vec4))) {
        info.velocity = glm::vec4(0.0f);
    }
    
    // Calculate spatial cell from position (same logic as GPU)
    const float CELL_SIZE = 2.0f;
    const uint32_t GRID_WIDTH = 64;
    
    glm::vec2 pos2D = glm::vec2(info.position);
    glm::ivec2 gridCoord = glm::ivec2(glm::floor(pos2D / CELL_SIZE));
    uint32_t x = static_cast<uint32_t>(gridCoord.x) & (GRID_WIDTH - 1);
    uint32_t y = static_cast<uint32_t>(gridCoord.y) & (GRID_WIDTH - 1);
    info.spatialCell = x + y * GRID_WIDTH;
    
    return true;
}

bool EntityBufferManager::readbackSpatialCell(uint32_t cellIndex, std::vector<uint32_t>& entityIds) const {
    const uint32_t SPATIAL_MAP_SIZE = 4096; // 64x64
    if (cellIndex >= SPATIAL_MAP_SIZE) {
        return false;
    }
    
    entityIds.clear();
    
    // Read the spatial cell entry (uvec2: entityId, nextIndex)
    glm::uvec2 cellData;
    if (!readGPUBuffer(spatialMapBuffer.getBuffer(), 
                      &cellData, sizeof(glm::uvec2), 
                      cellIndex * sizeof(glm::uvec2))) {
        return false;
    }
    
    std::cout << "DEBUG: Cell " << cellIndex << " raw data: entityId=" << cellData.x 
              << ", next=" << cellData.y << std::endl;
    
    // Check if there's an entity in this cell
    const uint32_t NULL_INDEX = 0xFFFFFFFF;
    if (cellData.x == NULL_INDEX || cellData.x == 0) {
        std::cout << "DEBUG: Empty cell (no entities)" << std::endl;
        return true; // Empty cell
    }
    
    // Looking at the GPU shader more carefully:
    // atomicExchange(spatialMap.spatialCells[cellIndex].x, entityIndex) 
    // spatialMap.spatialCells[cellIndex].y = oldHead
    //
    // This means:
    // - .x contains the most recent entity ID added to this cell
    // - .y contains the previous head (which could be another entity or NULL_INDEX)
    //
    // But there's a race condition issue: the .y gets overwritten each time
    // So we can only reliably get the most recent entity in each cell
    
    uint32_t headEntity = cellData.x;
    if (headEntity != NULL_INDEX && headEntity < maxEntities) {
        std::cout << "DEBUG: Found head entity " << headEntity << " in cell " << cellIndex << std::endl;
        entityIds.push_back(headEntity);
        
        // Try to follow the chain, but this might not work reliably due to race conditions
        uint32_t nextEntity = cellData.y;
        if (nextEntity != NULL_INDEX && nextEntity < maxEntities && nextEntity != headEntity) {
            std::cout << "DEBUG: Found next entity " << nextEntity << " in cell " << cellIndex << std::endl;
            entityIds.push_back(nextEntity);
        }
    }
    
    return true;
}

bool EntityBufferManager::initializeSpatialMapBuffer() {
    const uint32_t SPATIAL_MAP_SIZE = 4096; // 64x64 grid
    const uint32_t NULL_INDEX = 0xFFFFFFFF;
    
    // Create initialization data with NULL values
    std::vector<glm::uvec2> initData(SPATIAL_MAP_SIZE);
    for (auto& cell : initData) {
        cell.x = NULL_INDEX; // entityId = NULL
        cell.y = NULL_INDEX; // nextIndex = NULL
    }
    
    // Upload the NULL initialization data
    VkDeviceSize uploadSize = SPATIAL_MAP_SIZE * sizeof(glm::uvec2);
    bool success = uploadService.upload(spatialMapBuffer, initData.data(), uploadSize, 0);
    
    if (success) {
        std::cout << "EntityBufferManager: Spatial map buffer initialized with NULL values (" 
                  << SPATIAL_MAP_SIZE << " cells)" << std::endl;
    }
    
    return success;
}

bool EntityBufferManager::readbackEntityAtPositionSafe(glm::vec2 worldPos, EntityDebugInfo& info) const {
    if (!context) {
        std::cerr << "EntityBufferManager::readbackEntityAtPositionSafe - No Vulkan context available" << std::endl;
        return false;
    }
    
    // Wait for GPU to complete all operations to ensure spatial map is consistent
    const auto& vk = context->getLoader();
    const VkDevice device = context->getDevice();
    
    VkResult result = vk.vkDeviceWaitIdle(device);
    if (result != VK_SUCCESS) {
        std::cerr << "EntityBufferManager::readbackEntityAtPositionSafe - Failed to wait for GPU idle: " << result << std::endl;
        return false;
    }
    
    std::cout << "GPU synchronized - spatial map data should be consistent" << std::endl;
    
    // Now perform the readback with guaranteed consistent data
    return readbackEntityAtPosition(worldPos, info);
}

