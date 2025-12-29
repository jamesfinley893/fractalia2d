#pragma once

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include "../rendering/frame_graph.h"
#include <memory>

// Forward declarations
class VulkanSwapchain;

class SwapchainPresentNode : public FrameGraphNode {
    DECLARE_FRAME_GRAPH_NODE(SwapchainPresentNode)
    
public:
    SwapchainPresentNode(
        FrameGraphTypes::ResourceId colorTarget,
        VulkanSwapchain* swapchain
    );
    
    // FrameGraphNode interface
    std::vector<ResourceDependency> getInputs() const override;
    std::vector<ResourceDependency> getOutputs() const override;
    void execute(VkCommandBuffer commandBuffer, const FrameGraph& frameGraph) override;
    
    // Node lifecycle - standardized pattern
    bool initializeNode(const FrameGraph& frameGraph) override;
    void prepareFrame(const FrameContext& frameContext) override;
    void releaseFrame(uint32_t frameIndex) override;
    
    // Queue requirements - presentation happens on graphics queue
    bool needsComputeQueue() const override { return false; }
    bool needsGraphicsQueue() const override { return true; }
    
    // Update swapchain image index for current frame
    void setImageIndex(uint32_t imageIndex) { this->imageIndex = imageIndex; }
    
    // Set current frame's swapchain image resource ID (called each frame)
    void setCurrentSwapchainImageId(FrameGraphTypes::ResourceId currentImageId) { this->currentSwapchainImageId = currentImageId; }
    
    // Get current image index (for debugging/validation)
    uint32_t getImageIndex() const { return imageIndex; }

private:
    FrameGraphTypes::ResourceId colorTargetId; // Static placeholder - not used  
    FrameGraphTypes::ResourceId currentSwapchainImageId = 0; // Dynamic per-frame ID
    
    // External dependencies (not owned) - validated during execution
    VulkanSwapchain* swapchain;
    
    // Current frame state
    uint32_t imageIndex = 0;
};
