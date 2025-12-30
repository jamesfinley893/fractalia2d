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
    struct Data {
        FrameGraphTypes::ResourceId velocityBufferId = 0;
        FrameGraphTypes::ResourceId runtimeStateBufferId = 0;
        FrameGraphTypes::ResourceId spatialMapBufferId = 0;
        FrameGraphTypes::ResourceId positionBufferId = 0;
        FrameGraphTypes::ResourceId currentPositionBufferId = 0;
        FrameGraphTypes::ResourceId targetPositionBufferId = 0;
        FrameGraphTypes::ResourceId controlParamsBufferId = 0;
        FrameGraphTypes::ResourceId spatialNextBufferId = 0;
        FrameGraphTypes::ResourceId bodyDataBufferId = 0;
        FrameGraphTypes::ResourceId bodyParamsBufferId = 0;
        FrameGraphTypes::ResourceId nodeVelocityBufferId = 0;
        FrameGraphTypes::ResourceId nodeInvMassBufferId = 0;
        FrameGraphTypes::ResourceId triangleRestBufferId = 0;
        FrameGraphTypes::ResourceId triangleAreaBufferId = 0;
        FrameGraphTypes::ResourceId nodeForceBufferId = 0;
        FrameGraphTypes::ResourceId nodeRestBufferId = 0;
        FrameGraphTypes::ResourceId triangleIndexBufferId = 0;
        ComputePipelineManager* computeManager = nullptr;
        GPUEntityManager* gpuEntityManager = nullptr;
        std::shared_ptr<GPUTimeoutDetector> timeoutDetector = nullptr;
    };

    explicit PhysicsComputeNode(const Data& data);
    
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
        uint32_t elementCount,
        bool finalToGraphics);
    
    Data data;
    
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
    struct FEMPushConstants {
        float time;
        float deltaTime;
        uint32_t bodyCount;
        uint32_t nodeCount;
        uint32_t triangleCount;
        uint32_t frame;
        uint32_t elementOffset;
        uint32_t mode;
        uint32_t padding;
    } pushConstants{};
};
