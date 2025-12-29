#include "physics_compute_node.h"
#include "../pipelines/compute_pipeline_manager.h"
#include "../../ecs/gpu/gpu_entity_manager.h"
#include "../core/vulkan_context.h"
#include "../core/vulkan_function_loader.h"
#include "../core/vulkan_constants.h"
#include "../pipelines/descriptor_layout_manager.h"
#include "../monitoring/gpu_timeout_detector.h"
#include "../../ecs/gpu/soft_body_constants.h"
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
    
    DispatchParams calculateDispatchParams(uint32_t elementCount, uint32_t maxWorkgroups, bool forceChunking) {
        const uint32_t totalWorkgroups = (elementCount + THREADS_PER_WORKGROUP - 1) / THREADS_PER_WORKGROUP;
        return {
            totalWorkgroups,
            maxWorkgroups,
            totalWorkgroups > maxWorkgroups || forceChunking
        };
    }
}

PhysicsComputeNode::PhysicsComputeNode(const Data& data)
    : data(data) {
    
    // Validate dependencies during construction for fail-fast behavior
    if (!this->data.computeManager) {
        throw std::invalid_argument("PhysicsComputeNode: computeManager cannot be null");
    }
    if (!this->data.gpuEntityManager) {
        throw std::invalid_argument("PhysicsComputeNode: gpuEntityManager cannot be null");
    }
}

std::vector<ResourceDependency> PhysicsComputeNode::getInputs() const {
    return {
        {data.velocityBufferId, ResourceAccess::ReadWrite, PipelineStage::ComputeShader},
        {data.runtimeStateBufferId, ResourceAccess::ReadWrite, PipelineStage::ComputeShader},
        {data.positionBufferId, ResourceAccess::ReadWrite, PipelineStage::ComputeShader},
        {data.currentPositionBufferId, ResourceAccess::ReadWrite, PipelineStage::ComputeShader},
        {data.spatialMapBufferId, ResourceAccess::ReadWrite, PipelineStage::ComputeShader},
        {data.controlParamsBufferId, ResourceAccess::Read, PipelineStage::ComputeShader},
        {data.spatialNextBufferId, ResourceAccess::ReadWrite, PipelineStage::ComputeShader},
        {data.particleVelocityBufferId, ResourceAccess::ReadWrite, PipelineStage::ComputeShader},
        {data.particleInvMassBufferId, ResourceAccess::Read, PipelineStage::ComputeShader},
        {data.particleBodyBufferId, ResourceAccess::Read, PipelineStage::ComputeShader},
        {data.bodyDataBufferId, ResourceAccess::Read, PipelineStage::ComputeShader},
        {data.bodyParamsBufferId, ResourceAccess::Read, PipelineStage::ComputeShader},
        {data.distanceConstraintBufferId, ResourceAccess::Read, PipelineStage::ComputeShader},
    };
}

std::vector<ResourceDependency> PhysicsComputeNode::getOutputs() const {
    return {
        {data.velocityBufferId, ResourceAccess::Write, PipelineStage::ComputeShader},
        {data.runtimeStateBufferId, ResourceAccess::Write, PipelineStage::ComputeShader},
        {data.positionBufferId, ResourceAccess::Write, PipelineStage::ComputeShader},
        {data.currentPositionBufferId, ResourceAccess::Write, PipelineStage::ComputeShader},
        {data.spatialMapBufferId, ResourceAccess::Write, PipelineStage::ComputeShader},
        {data.particleVelocityBufferId, ResourceAccess::Write, PipelineStage::ComputeShader},
    };
}

bool PhysicsComputeNode::ensurePipeline() {
    if (!data.computeManager) {
        return false;
    }

    auto layoutSpec = DescriptorLayoutPresets::createEntityComputeLayout();
    VkDescriptorSetLayout descriptorLayout = data.computeManager->getLayoutManager()->getLayout(layoutSpec);
    if (descriptorLayout == VK_NULL_HANDLE) {
        return false;
    }
    if (descriptorLayout != cachedDescriptorLayout) {
        pipelineDirty = true;
    }

    if (!pipelineDirty) {
        return cachedPipeline != VK_NULL_HANDLE && cachedPipelineLayout != VK_NULL_HANDLE;
    }

    ComputePipelineState pipelineState = ComputePipelinePresets::createPhysicsState(descriptorLayout);
    VkPipeline pipeline = data.computeManager->getPipeline(pipelineState);
    VkPipelineLayout pipelineLayout = data.computeManager->getPipelineLayout(pipelineState);
    if (pipeline == VK_NULL_HANDLE || pipelineLayout == VK_NULL_HANDLE) {
        return false;
    }

    cachedDescriptorLayout = descriptorLayout;
    cachedPipeline = pipeline;
    cachedPipelineLayout = pipelineLayout;
    pipelineDirty = false;
    return true;
}

void PhysicsComputeNode::execute(VkCommandBuffer commandBuffer, const FrameGraph& frameGraph) {
    // Validate dependencies are still valid
    if (!data.computeManager || !data.gpuEntityManager) {
        std::cerr << "PhysicsComputeNode: Critical error - dependencies became null during execution" << std::endl;
        return;
    }
    
    const uint32_t bodyCount = data.gpuEntityManager->getEntityCount();
    if (bodyCount == 0) {
        FRAME_GRAPH_DEBUG_LOG_THROTTLED(debugCounter, 1800, "PhysicsComputeNode: No entities to process");
        return;
    }
    
    // Set frame counter from FrameGraph for compute shader consistency
    pushConstants.frame = frameGraph.getGlobalFrameCounter();
    
    if (!ensurePipeline()) {
        std::cerr << "PhysicsComputeNode: Failed to ensure physics compute pipeline or layout" << std::endl;
        return;
    }
    
    // Create compute dispatch
    ComputeDispatch dispatch{};
    dispatch.pipeline = cachedPipeline;
    dispatch.layout = cachedPipelineLayout;
    
    // Set up descriptor sets
    VkDescriptorSet computeDescriptorSet = data.gpuEntityManager->getDescriptorManager().getComputeDescriptorSet();
    
    if (computeDescriptorSet != VK_NULL_HANDLE) {
        dispatch.descriptorSets.push_back(computeDescriptorSet);
    } else {
        std::cerr << "PhysicsComputeNode: ERROR - Missing compute descriptor set!" << std::endl;
        return;
    }
    
    const uint32_t particleCount = bodyCount * SoftBodyConstants::kParticlesPerBody;
    const uint32_t constraintCount = bodyCount * SoftBodyConstants::kConstraintsPerBody;

    // Configure push constants and dispatch
    pushConstants.bodyCount = bodyCount;
    pushConstants.particleCount = particleCount;
    pushConstants.constraintCount = constraintCount;
    dispatch.pushConstantData = &pushConstants;
    dispatch.pushConstantSize = sizeof(PBDPushConstants);
    dispatch.pushConstantStages = VK_SHADER_STAGE_COMPUTE_BIT;
    
    // Apply adaptive workload management
    uint32_t maxWorkgroupsPerDispatch = adaptiveMaxWorkgroups;
    bool shouldForceChunking = forceChunkedDispatch;
    
    if (data.timeoutDetector) {
        auto recommendation = data.timeoutDetector->getRecoveryRecommendation();
        if (recommendation.shouldReduceWorkload) {
            maxWorkgroupsPerDispatch = std::min(maxWorkgroupsPerDispatch, recommendation.recommendedMaxWorkgroups);
        }
        if (recommendation.shouldSplitDispatches) {
            shouldForceChunking = true;
        }
        if (!data.timeoutDetector->isGPUHealthy()) {
            std::cerr << "PhysicsComputeNode: GPU not healthy, reducing workload" << std::endl;
            maxWorkgroupsPerDispatch = std::min(maxWorkgroupsPerDispatch, 512u);
        }
    }
    
    // Calculate dispatch parameters
    // Debug logging (thread-safe) - once every 30 seconds
    FRAME_GRAPH_DEBUG_LOG_THROTTLED(debugCounter, 1800, "PhysicsComputeNode: " << bodyCount << " bodies, " << particleCount << " particles");
    
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
    
    auto dispatchPass = [&](const char* name, uint32_t mode, uint32_t elementCount, bool finalToGraphics) {
        if (elementCount == 0) {
            return;
        }
        pushConstants.mode = mode;
        pushConstants.elementOffset = 0;

        auto dispatchParams = calculateDispatchParams(elementCount, maxWorkgroupsPerDispatch, shouldForceChunking);
        if (dispatchParams.totalWorkgroups > 65535) {
            std::cerr << "ERROR: Workgroup count " << dispatchParams.totalWorkgroups << " exceeds Vulkan limit!" << std::endl;
            return;
        }

        if (!dispatchParams.useChunking) {
            if (data.timeoutDetector) {
                data.timeoutDetector->beginComputeDispatch(name, dispatchParams.totalWorkgroups);
            }

            vk.vkCmdPushConstants(
                commandBuffer, dispatch.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                0, sizeof(PBDPushConstants), &pushConstants);

            vk.vkCmdDispatch(commandBuffer, dispatchParams.totalWorkgroups, 1, 1);

            if (data.timeoutDetector) {
                data.timeoutDetector->endComputeDispatch();
            }

            VkMemoryBarrier memoryBarrier{};
            memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            memoryBarrier.dstAccessMask = finalToGraphics
                ? VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT
                : (VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

            vk.vkCmdPipelineBarrier(
                commandBuffer,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                finalToGraphics ? VK_PIPELINE_STAGE_VERTEX_INPUT_BIT : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 1, &memoryBarrier, 0, nullptr, 0, nullptr);
        } else {
            executeChunkedDispatch(commandBuffer, context, dispatch,
                                   dispatchParams.totalWorkgroups, dispatchParams.maxWorkgroupsPerChunk,
                                   elementCount, finalToGraphics);
        }
    };

    dispatchPass("PBD_Integrate", 0u, particleCount, false);
    dispatchPass("PBD_ClearSpatial", 1u, 4096u, false);
    dispatchPass("PBD_BuildSpatial", 2u, particleCount, false);

    constexpr uint32_t solverIterations = 8;
    for (uint32_t iter = 0; iter < solverIterations; ++iter) {
        dispatchPass("PBD_SolveDistance", 3u, constraintCount, false);
        dispatchPass("PBD_SolveArea", 4u, bodyCount, false);
        dispatchPass("PBD_Collide", 5u, particleCount, false);
    }

    dispatchPass("PBD_Finalize", 6u, particleCount, true);
}

void PhysicsComputeNode::executeChunkedDispatch(
    VkCommandBuffer commandBuffer, 
    const VulkanContext* context, 
    const ComputeDispatch& dispatch,
    uint32_t totalWorkgroups,
    uint32_t maxWorkgroupsPerChunk,
    uint32_t elementCount,
    bool finalToGraphics) {
    
    // Cache loader reference for performance
    const auto& vk = context->getLoader();
    
    uint32_t processedWorkgroups = 0;
    uint32_t chunkCount = 0;
    
    while (processedWorkgroups < totalWorkgroups) {
        uint32_t currentChunkSize = std::min(maxWorkgroupsPerChunk, totalWorkgroups - processedWorkgroups);
        uint32_t baseElementOffset = processedWorkgroups * THREADS_PER_WORKGROUP;
        
        if (elementCount <= baseElementOffset) break; // No more elements to process
        
        // Monitor chunk execution
        if (data.timeoutDetector) {
            std::string chunkName = "Physics_Chunk" + std::to_string(chunkCount);
            data.timeoutDetector->beginComputeDispatch(chunkName.c_str(), currentChunkSize);
        }
        
        // Update push constants for this chunk
        PBDPushConstants chunkPushConstants = pushConstants;
        chunkPushConstants.elementOffset = baseElementOffset;
        
        vk.vkCmdPushConstants(
            commandBuffer, dispatch.layout, VK_SHADER_STAGE_COMPUTE_BIT,
            0, sizeof(PBDPushConstants), &chunkPushConstants);
        
        vk.vkCmdDispatch(commandBuffer, currentChunkSize, 1, 1);
        
        if (data.timeoutDetector) {
            data.timeoutDetector->endComputeDispatch();
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
    
    // Final memory barrier for compute→compute or compute→graphics synchronization
    VkMemoryBarrier finalMemoryBarrier{};
    finalMemoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    finalMemoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    finalMemoryBarrier.dstAccessMask = finalToGraphics
        ? VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT
        : (VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    
    vk.vkCmdPipelineBarrier(
        commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        finalToGraphics ? VK_PIPELINE_STAGE_VERTEX_INPUT_BIT : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &finalMemoryBarrier, 0, nullptr, 0, nullptr);
    
    // Debug statistics logging (thread-safe)
    if constexpr (FRAME_GRAPH_DEBUG_ENABLED) {
        uint32_t chunkLogCounter = FrameGraphDebug::incrementCounter(debugCounter);
        if (chunkLogCounter % 300 == 0) {
            std::cout << "[FrameGraph Debug] PhysicsComputeNode: Split dispatch into " << chunkCount 
                      << " chunks (" << maxWorkgroupsPerChunk << " max) for " << elementCount << " elements (occurrence #" << chunkLogCounter << ")" << std::endl;
            
            if (data.timeoutDetector) {
                auto stats = data.timeoutDetector->getStats();
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
    if (!data.computeManager) {
        std::cerr << "PhysicsComputeNode: ComputePipelineManager is null" << std::endl;
        return false;
    }
    if (!data.gpuEntityManager) {
        std::cerr << "PhysicsComputeNode: GPUEntityManager is null" << std::endl;
        return false;
    }
    return ensurePipeline();
}

void PhysicsComputeNode::prepareFrame(const FrameContext& frameContext) {
    // Store timing data for execution
    currentTime = frameContext.time;
    currentDeltaTime = frameContext.deltaTime;
    
    // Update push constants with timing data - frame counter will be set in execute()
    pushConstants.time = frameContext.time;
    pushConstants.deltaTime = frameContext.deltaTime;
}

void PhysicsComputeNode::releaseFrame(uint32_t frameIndex) {
    // Per-frame cleanup - nothing to clean up for physics compute node
}

