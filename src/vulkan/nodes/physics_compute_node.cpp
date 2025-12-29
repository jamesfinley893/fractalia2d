#include "physics_compute_node.h"
#include "../pipelines/compute_pipeline_manager.h"
#include "../../ecs/gpu/gpu_entity_manager.h"
#include "../core/vulkan_context.h"
#include "../core/vulkan_function_loader.h"
#include "../core/vulkan_constants.h"
#include "../pipelines/descriptor_layout_manager.h"
#include "../monitoring/gpu_timeout_detector.h"
#include <iostream>
#include <array>
#include <glm/glm.hpp>
#include <stdexcept>
#include <memory>

namespace {
    struct DispatchParams {
        uint32_t totalWorkgroups;
        uint32_t maxWorkgroupsPerChunk;
        bool useChunking;
    };
    
    DispatchParams calculateDispatchParams(uint32_t entityCount, uint32_t maxWorkgroups, bool forceChunking) {
        // Calculate workgroups needed for both spatial map clearing and entity processing
        const uint32_t SPATIAL_MAP_SIZE = 4096;
        const uint32_t spatialClearWorkgroups = (SPATIAL_MAP_SIZE + THREADS_PER_WORKGROUP - 1) / THREADS_PER_WORKGROUP;
        const uint32_t entityWorkgroups = (entityCount + THREADS_PER_WORKGROUP - 1) / THREADS_PER_WORKGROUP;
        
        // Use maximum of both requirements
        const uint32_t totalWorkgroups = std::max(spatialClearWorkgroups, entityWorkgroups);
        
        return {
            totalWorkgroups,
            maxWorkgroups,
            totalWorkgroups > maxWorkgroups || forceChunking
        };
    }
}

PhysicsComputeNode::PhysicsComputeNode(
    FrameGraphTypes::ResourceId velocityBuffer,
    FrameGraphTypes::ResourceId runtimeStateBuffer,
    FrameGraphTypes::ResourceId spatialMapBuffer,
    FrameGraphTypes::ResourceId positionBuffer,
    FrameGraphTypes::ResourceId currentPositionBuffer,
    FrameGraphTypes::ResourceId targetPositionBuffer,
    ComputePipelineManager* computeManager,
    GPUEntityManager* gpuEntityManager,
    std::shared_ptr<GPUTimeoutDetector> timeoutDetector
) : velocityBufferId(velocityBuffer)
  , runtimeStateBufferId(runtimeStateBuffer)
  , spatialMapBufferId(spatialMapBuffer)
  , positionBufferId(positionBuffer)
  , currentPositionBufferId(currentPositionBuffer)
  , targetPositionBufferId(targetPositionBuffer)
  , computeManager(computeManager)
  , gpuEntityManager(gpuEntityManager)
  , timeoutDetector(timeoutDetector) {
    
    // Validate dependencies during construction for fail-fast behavior
    if (!computeManager) {
        throw std::invalid_argument("PhysicsComputeNode: computeManager cannot be null");
    }
    if (!gpuEntityManager) {
        throw std::invalid_argument("PhysicsComputeNode: gpuEntityManager cannot be null");
    }
}

std::vector<ResourceDependency> PhysicsComputeNode::getInputs() const {
    return {
        {velocityBufferId, ResourceAccess::ReadWrite, PipelineStage::ComputeShader},
        {runtimeStateBufferId, ResourceAccess::ReadWrite, PipelineStage::ComputeShader},
        {currentPositionBufferId, ResourceAccess::ReadWrite, PipelineStage::ComputeShader},
        {spatialMapBufferId, ResourceAccess::ReadWrite, PipelineStage::ComputeShader},
    };
}

std::vector<ResourceDependency> PhysicsComputeNode::getOutputs() const {
    return {
        {velocityBufferId, ResourceAccess::Write, PipelineStage::ComputeShader},
        {runtimeStateBufferId, ResourceAccess::Write, PipelineStage::ComputeShader},
        {positionBufferId, ResourceAccess::Write, PipelineStage::ComputeShader},
        {currentPositionBufferId, ResourceAccess::Write, PipelineStage::ComputeShader},
        {spatialMapBufferId, ResourceAccess::Write, PipelineStage::ComputeShader},
    };
}

void PhysicsComputeNode::execute(VkCommandBuffer commandBuffer, const FrameGraph& frameGraph) {
    // Validate dependencies are still valid
    if (!computeManager || !gpuEntityManager) {
        std::cerr << "PhysicsComputeNode: Critical error - dependencies became null during execution" << std::endl;
        return;
    }
    
    const uint32_t entityCount = gpuEntityManager->getEntityCount();
    if (entityCount == 0) {
        FRAME_GRAPH_DEBUG_LOG_THROTTLED(debugCounter, 1800, "PhysicsComputeNode: No entities to process");
        return;
    }
    
    // Create compute pipeline state for physics
    auto layoutSpec = DescriptorLayoutPresets::createEntityComputeLayout();
    VkDescriptorSetLayout descriptorLayout = computeManager->getLayoutManager()->getLayout(layoutSpec);
    ComputePipelineState pipelineState = ComputePipelinePresets::createPhysicsState(descriptorLayout);
    
    // Set frame counter from FrameGraph for compute shader consistency
    pushConstants.frame = frameGraph.getGlobalFrameCounter();
    
    // Create compute dispatch
    ComputeDispatch dispatch{};
    dispatch.pipeline = computeManager->getPipeline(pipelineState);
    dispatch.layout = computeManager->getPipelineLayout(pipelineState);
    
    if (dispatch.pipeline == VK_NULL_HANDLE || dispatch.layout == VK_NULL_HANDLE) {
        std::cerr << "PhysicsComputeNode: Failed to get physics compute pipeline or layout" << std::endl;
        return;
    }
    
    // Set up descriptor sets
    VkDescriptorSet computeDescriptorSet = gpuEntityManager->getDescriptorManager().getComputeDescriptorSet();
    
    if (computeDescriptorSet != VK_NULL_HANDLE) {
        dispatch.descriptorSets.push_back(computeDescriptorSet);
    } else {
        std::cerr << "PhysicsComputeNode: ERROR - Missing compute descriptor set!" << std::endl;
        return;
    }
    
    // Configure push constants and dispatch
    pushConstants.entityCount = entityCount;
    dispatch.pushConstantData = &pushConstants;
    dispatch.pushConstantSize = sizeof(PhysicsPushConstants);
    dispatch.pushConstantStages = VK_SHADER_STAGE_COMPUTE_BIT;
    dispatch.calculateOptimalDispatch(entityCount, glm::uvec3(THREADS_PER_WORKGROUP, 1, 1));
    
    // Apply adaptive workload management
    uint32_t maxWorkgroupsPerDispatch = adaptiveMaxWorkgroups;
    bool shouldForceChunking = forceChunkedDispatch;
    
    if (timeoutDetector) {
        auto recommendation = timeoutDetector->getRecoveryRecommendation();
        if (recommendation.shouldReduceWorkload) {
            maxWorkgroupsPerDispatch = std::min(maxWorkgroupsPerDispatch, recommendation.recommendedMaxWorkgroups);
        }
        if (recommendation.shouldSplitDispatches) {
            shouldForceChunking = true;
        }
        if (!timeoutDetector->isGPUHealthy()) {
            std::cerr << "PhysicsComputeNode: GPU not healthy, reducing workload" << std::endl;
            maxWorkgroupsPerDispatch = std::min(maxWorkgroupsPerDispatch, 512u);
        }
    }
    
    // Calculate dispatch parameters
    auto dispatchParams = calculateDispatchParams(entityCount, maxWorkgroupsPerDispatch, shouldForceChunking);
    
    // Validate dispatch limits
    if (dispatchParams.totalWorkgroups > 65535) {
        std::cerr << "ERROR: Workgroup count " << dispatchParams.totalWorkgroups << " exceeds Vulkan limit!" << std::endl;
        return;
    }
    
    // Debug logging (thread-safe) - once every 30 seconds
    FRAME_GRAPH_DEBUG_LOG_THROTTLED(debugCounter, 1800, "PhysicsComputeNode: " << entityCount << " entities → " << dispatchParams.totalWorkgroups << " workgroups");
    
    const VulkanContext* context = frameGraph.getContext();
    if (!context) {
        std::cerr << "PhysicsComputeNode: Cannot get Vulkan context" << std::endl;
        return;
    }
    
    // Cache loader reference for performance
    const auto& vk = context->getLoader();
    
    // Bind pipeline and descriptor sets once
    vk.vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, dispatch.pipeline);
    
    vk.vkCmdBindDescriptorSets(
        commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, dispatch.layout,
        0, 1, &dispatch.descriptorSets[0], 0, nullptr);
    
    if (!dispatchParams.useChunking) {
        // Single dispatch execution
        std::cout << "PhysicsComputeNode: Starting single dispatch execution..." << std::endl;
        if (timeoutDetector) {
            timeoutDetector->beginComputeDispatch("Physics", dispatchParams.totalWorkgroups);
        }
        
        vk.vkCmdPushConstants(
            commandBuffer, dispatch.layout, VK_SHADER_STAGE_COMPUTE_BIT,
            0, sizeof(PhysicsPushConstants), &pushConstants);
        
        vk.vkCmdDispatch(commandBuffer, dispatchParams.totalWorkgroups, 1, 1);
        
        if (timeoutDetector) {
            timeoutDetector->endComputeDispatch();
        }
        
        // Memory barrier for compute→graphics synchronization
        VkMemoryBarrier memoryBarrier{};
        memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        memoryBarrier.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
        
        vk.vkCmdPipelineBarrier(
            commandBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
            0, 1, &memoryBarrier, 0, nullptr, 0, nullptr);
    } else {
        executeChunkedDispatch(commandBuffer, context, dispatch, 
                              dispatchParams.totalWorkgroups, dispatchParams.maxWorkgroupsPerChunk, entityCount);
    }
}

void PhysicsComputeNode::executeChunkedDispatch(
    VkCommandBuffer commandBuffer, 
    const VulkanContext* context, 
    const ComputeDispatch& dispatch,
    uint32_t totalWorkgroups,
    uint32_t maxWorkgroupsPerChunk,
    uint32_t entityCount) {
    
    // Cache loader reference for performance
    const auto& vk = context->getLoader();
    
    uint32_t processedWorkgroups = 0;
    uint32_t chunkCount = 0;
    
    while (processedWorkgroups < totalWorkgroups) {
        uint32_t currentChunkSize = std::min(maxWorkgroupsPerChunk, totalWorkgroups - processedWorkgroups);
        uint32_t baseEntityOffset = processedWorkgroups * THREADS_PER_WORKGROUP;
        
        if (entityCount <= baseEntityOffset) break; // No more entities to process
        
        // Monitor chunk execution
        if (timeoutDetector) {
            std::string chunkName = "Physics_Chunk" + std::to_string(chunkCount);
            timeoutDetector->beginComputeDispatch(chunkName.c_str(), currentChunkSize);
        }
        
        // Update push constants for this chunk
        PhysicsPushConstants chunkPushConstants = pushConstants;
        chunkPushConstants.entityOffset = baseEntityOffset;
        
        vk.vkCmdPushConstants(
            commandBuffer, dispatch.layout, VK_SHADER_STAGE_COMPUTE_BIT,
            0, sizeof(PhysicsPushConstants), &chunkPushConstants);
        
        vk.vkCmdDispatch(commandBuffer, currentChunkSize, 1, 1);
        
        if (timeoutDetector) {
            timeoutDetector->endComputeDispatch();
        }
        
        // Inter-chunk memory barrier
        if (processedWorkgroups + currentChunkSize < totalWorkgroups) {
            VkMemoryBarrier memoryBarrier{};
            memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            
            vk.vkCmdPipelineBarrier(
                commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memoryBarrier, 0, nullptr, 0, nullptr);
        }
        
        processedWorkgroups += currentChunkSize;
        chunkCount++;
    }
    
    // Final memory barrier for compute→graphics synchronization
    VkMemoryBarrier finalMemoryBarrier{};
    finalMemoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    finalMemoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    finalMemoryBarrier.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
    
    vk.vkCmdPipelineBarrier(
        commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, 0, 1, &finalMemoryBarrier, 0, nullptr, 0, nullptr);
    
    // Debug statistics logging (thread-safe)
    if constexpr (FRAME_GRAPH_DEBUG_ENABLED) {
        uint32_t chunkLogCounter = FrameGraphDebug::incrementCounter(debugCounter);
        if (chunkLogCounter % 300 == 0) {
            std::cout << "[FrameGraph Debug] PhysicsComputeNode: Split dispatch into " << chunkCount 
                      << " chunks (" << maxWorkgroupsPerChunk << " max) for " << entityCount << " entities (occurrence #" << chunkLogCounter << ")" << std::endl;
            
            if (timeoutDetector) {
                auto stats = timeoutDetector->getStats();
                std::cout << "  GPU Stats: avg=" << stats.averageDispatchTimeMs 
                          << "ms, peak=" << stats.peakDispatchTimeMs << "ms"
                          << ", warnings=" << stats.warningCount 
                          << ", critical=" << stats.criticalCount << std::endl;
            }
        }
    }
}

// Node lifecycle implementation
bool PhysicsComputeNode::initializeNode(const FrameGraph& frameGraph) {
    // One-time initialization - validate dependencies
    if (!computeManager) {
        std::cerr << "PhysicsComputeNode: ComputePipelineManager is null" << std::endl;
        return false;
    }
    if (!gpuEntityManager) {
        std::cerr << "PhysicsComputeNode: GPUEntityManager is null" << std::endl;
        return false;
    }
    return true;
}

void PhysicsComputeNode::prepareFrame(uint32_t frameIndex, float time, float deltaTime) {
    // Store timing data for execution
    currentTime = time;
    currentDeltaTime = deltaTime;
    
    // Update push constants with timing data - frame counter will be set in execute()
    pushConstants.time = time;
    pushConstants.deltaTime = deltaTime;
}

void PhysicsComputeNode::releaseFrame(uint32_t frameIndex) {
    // Per-frame cleanup - nothing to clean up for physics compute node
}

