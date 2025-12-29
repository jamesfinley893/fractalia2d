#pragma once

#include <vulkan/vulkan.h>

// Forward declarations
class VulkanContext;
class VulkanSwapchain;
class GraphicsPipelineManager;
class GPUSynchronizationService;
class VulkanSync;

struct SurfaceAcquisitionResult {
    bool success = false;
    uint32_t imageIndex = 0;
    bool recreationNeeded = false;
    VkResult result = VK_SUCCESS;
};

class PresentationSurface {
public:
    PresentationSurface();
    ~PresentationSurface();

    bool initialize(
        VulkanContext* context,
        VulkanSwapchain* swapchain,
        GraphicsPipelineManager* graphicsManager,
        GPUSynchronizationService* syncManager,
        VulkanSync* sync
    );
    
    void cleanup();

    // Main coordination methods
    SurfaceAcquisitionResult acquireNextImage(uint32_t currentFrame);
    bool recreateSwapchain();

    // Framebuffer resize handling
    void setFramebufferResized(bool resized) { framebufferResized = resized; }
    bool isFramebufferResized() const { return framebufferResized; }

private:
    // Dependencies
    VulkanContext* context = nullptr;
    VulkanSwapchain* swapchain = nullptr;
    GraphicsPipelineManager* graphicsManager = nullptr;
    GPUSynchronizationService* syncManager = nullptr;
    VulkanSync* sync = nullptr;
    
    // Current render pass tracking for swapchain recreation
    VkRenderPass currentRenderPass = VK_NULL_HANDLE;

    // State tracking
    bool recreationInProgress = false;
    bool framebufferResized = false;
    bool acquisitionInProgress = false;
};
