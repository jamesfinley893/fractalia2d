#pragma once

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include "../rendering/frame_graph.h"
#include "../rendering/frame_graph_debug.h"
#include "../core/vulkan_constants.h"
#include <memory>

// Forward declarations
class ComputePipelineManager;
class GPUEntityManager;
class GPUTimeoutDetector;

class PhysicsComputeNode : public FrameGraphNode {
    DECLARE_FRAME_GRAPH_NODE(PhysicsComputeNode)
    
public:
    PhysicsComputeNode(
        FrameGraphTypes::ResourceId velocityBuffer,
        FrameGraphTypes::ResourceId runtimeStateBuffer,
        FrameGraphTypes::ResourceId spatialMapBuffer,
        FrameGraphTypes::ResourceId positionBuffer,
        FrameGraphTypes::ResourceId currentPositionBuffer,
        FrameGraphTypes::ResourceId targetPositionBuffer,
        ComputePipelineManager* computeManager,
        GPUEntityManager* gpuEntityManager,
        std::shared_ptr<GPUTimeoutDetector> timeoutDetector = nullptr
    );
    
    // FrameGraphNode interface
    std::vector<ResourceDependency> getInputs() const override;
    std::vector<ResourceDependency> getOutputs() const override;
    void execute(VkCommandBuffer commandBuffer, const FrameGraph& frameGraph) override;
    
    // Queue requirements
    bool needsComputeQueue() const override { return true; }
    bool needsGraphicsQueue() const override { return false; }
    
    // Node lifecycle - standardized pattern
    bool initializeNode(const FrameGraph& frameGraph) override;
    void prepareFrame(const FrameContext& frameContext) override;
    void releaseFrame(uint32_t frameIndex) override;

    // Pipeline cache invalidation (e.g., layout cache reset)
    void invalidatePipelineCache() { pipelineDirty = true; }

private:
    bool ensurePipeline();
    // Helper method for chunked dispatch execution
    void executeChunkedDispatch(
        VkCommandBuffer commandBuffer, 
        const VulkanContext* context, 
        const class ComputeDispatch& dispatch,
        uint32_t totalWorkgroups,
        uint32_t maxWorkgroupsPerChunk,
        uint32_t entityCount);
    
    FrameGraphTypes::ResourceId velocityBufferId;
    FrameGraphTypes::ResourceId runtimeStateBufferId;
    FrameGraphTypes::ResourceId spatialMapBufferId;
    FrameGraphTypes::ResourceId positionBufferId;
    FrameGraphTypes::ResourceId currentPositionBufferId;
    FrameGraphTypes::ResourceId targetPositionBufferId;
    
    // External dependencies (not owned) - validated during execution
    ComputePipelineManager* computeManager;
    GPUEntityManager* gpuEntityManager;
    std::shared_ptr<GPUTimeoutDetector> timeoutDetector;
    
    // Adaptive dispatch parameters
    uint32_t adaptiveMaxWorkgroups = MAX_WORKGROUPS_PER_CHUNK;
    bool forceChunkedDispatch = true;     // Always use chunking for stability
    
    // Debug counter - zero overhead in release builds
    mutable FrameGraphDebug::DebugCounter debugCounter{};
    
    // Frame timing data for new lifecycle
    float currentTime = 0.0f;
    float currentDeltaTime = 0.0f;

    VkDescriptorSetLayout cachedDescriptorLayout = VK_NULL_HANDLE;
    VkPipeline cachedPipeline = VK_NULL_HANDLE;
    VkPipelineLayout cachedPipelineLayout = VK_NULL_HANDLE;
    bool pipelineDirty = true;
    
    // Frame data for compute shader
    struct PhysicsPushConstants {
        float time;
        float deltaTime;
        uint32_t entityCount;
        uint32_t frame;
        uint32_t entityOffset;  // For chunked dispatches
        uint32_t padding[3];    // Ensure 16-byte alignment
    } pushConstants{};
};
