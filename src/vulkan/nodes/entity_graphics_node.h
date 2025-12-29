#pragma once

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include "../rendering/frame_graph.h"
#include "../rendering/frame_graph_debug.h"
#include <flecs.h>
#include <cstdint>
#include <glm/glm.hpp>
#include <memory>

// Forward declarations
class GraphicsPipelineManager;
class VulkanSwapchain;
class ResourceCoordinator;
class GPUEntityManager;

class EntityGraphicsNode : public FrameGraphNode {
    DECLARE_FRAME_GRAPH_NODE(EntityGraphicsNode)
    
public:
    struct Data {
        FrameGraphTypes::ResourceId positionBufferId = 0;
        FrameGraphTypes::ResourceId movementParamsBufferId = 0;
        FrameGraphTypes::ResourceId colorTargetId = 0;
        GraphicsPipelineManager* graphicsManager = nullptr;
        VulkanSwapchain* swapchain = nullptr;
        ResourceCoordinator* resourceCoordinator = nullptr;
        GPUEntityManager* gpuEntityManager = nullptr;
    };

    explicit EntityGraphicsNode(const Data& data);
    
    // FrameGraphNode interface
    std::vector<ResourceDependency> getInputs() const override;
    std::vector<ResourceDependency> getOutputs() const override;
    void execute(VkCommandBuffer commandBuffer, const FrameGraph& frameGraph) override;
    
    // Queue requirements
    bool needsComputeQueue() const override { return false; }
    bool needsGraphicsQueue() const override { return true; }
    
    // Update swapchain image index for current frame
    void setImageIndex(uint32_t imageIndex) { this->imageIndex = imageIndex; }
    
    // Set current frame's swapchain image resource ID (called each frame)
    void setCurrentSwapchainImageId(FrameGraphTypes::ResourceId currentImageId) { this->currentSwapchainImageId = currentImageId; }
    
    // Node lifecycle - standardized pattern
    bool initializeNode(const FrameGraph& frameGraph) override;
    void prepareFrame(const FrameContext& frameContext) override;
    void releaseFrame(uint32_t frameIndex) override;
    
    // Set world reference for camera matrix access
    void setWorld(flecs::world* world) { this->world = world; }
    
    // Force uniform buffer update on next frame (call when camera changes)
    void markUniformBufferDirty() { uniformBufferDirty = true; }

    // Pipeline cache invalidation (e.g., swapchain recreation)
    void invalidatePipelineCache() { pipelineDirty = true; }

private:
    // Pipeline cache management
    bool ensurePipeline();

    // Internal uniform buffer update
    void updateUniformBuffer();
    
    // Uniform buffer optimization - cache and dirty tracking
    struct CachedUBO {
        glm::mat4 view;
        glm::mat4 proj;
    } cachedUBO{};
    
    // Helper methods for camera matrix management
    CachedUBO getCameraMatrices();
    bool updateUniformBufferData(const CachedUBO& ubo);
    
    // Resources
    Data data;
    FrameGraphTypes::ResourceId currentSwapchainImageId = 0; // Dynamic per-frame ID
    
    // Current frame state
    uint32_t imageIndex = 0;
    float frameTime = 0.0f;
    float frameDeltaTime = 0.0f;
    uint32_t currentFrameIndex = 0;
    
    // ECS world reference for camera matrices
    flecs::world* world = nullptr;
    
    bool uniformBufferDirty = true;  // Force update on first frame
    uint32_t lastUpdatedFrameIndex = UINT32_MAX; // Track which frame index was last updated

    // Cached pipeline state (rebuilt when invalidated)
    VkDescriptorSetLayout cachedDescriptorLayout = VK_NULL_HANDLE;
    VkRenderPass cachedRenderPass = VK_NULL_HANDLE;
    VkPipeline cachedPipeline = VK_NULL_HANDLE;
    VkPipelineLayout cachedPipelineLayout = VK_NULL_HANDLE;
    VkFormat cachedSwapchainFormat = VK_FORMAT_UNDEFINED;
    bool pipelineDirty = true;
    
    // Debug counters - zero overhead in release builds
    mutable FrameGraphDebug::DebugCounter debugCounter{};
    mutable FrameGraphDebug::DebugCounter noEntitiesCounter{};
    mutable FrameGraphDebug::DebugCounter drawCounter{};
    mutable FrameGraphDebug::DebugCounter updateCounter{};
};
