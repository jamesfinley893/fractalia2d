#pragma once

#include <vulkan/vulkan.h>
#include "../core/vulkan_constants.h"
#include "../rendering/frame_graph.h"

// Forward declarations
class VulkanContext;
class VulkanSync;
class VulkanSwapchain;
class QueueManager;

struct SubmissionResult {
    bool success = false;
    bool swapchainRecreationNeeded = false;
    VkResult lastResult = VK_SUCCESS;
};

class CommandSubmissionService {
public:
    CommandSubmissionService();
    ~CommandSubmissionService();

    bool initialize(VulkanContext* context, VulkanSync* sync, VulkanSwapchain* swapchain, QueueManager* queueManager);
    void cleanup();

    // Main submission methods
    SubmissionResult submitFrame(
        uint32_t currentFrame,
        uint32_t imageIndex,
        const FrameGraph::ExecutionResult& executionResult,
        bool framebufferResized
    );

private:
    // Dependencies
    VulkanContext* context = nullptr;
    VulkanSync* sync = nullptr;
    VulkanSwapchain* swapchain = nullptr;
    QueueManager* queueManager = nullptr;

    // Helper methods
    SubmissionResult submitComputeWorkAsync(uint32_t computeFrame);
    SubmissionResult submitGraphicsWork(uint32_t currentFrame, bool waitForCompute);
    SubmissionResult presentFrame(uint32_t currentFrame, uint32_t imageIndex, bool framebufferResized);

    // Fence management helpers
    bool resetAndSubmitCompute(uint32_t currentFrame, VkFence fence, VkCommandBuffer commandBuffer);
    bool resetAndSubmitGraphics(uint32_t currentFrame, VkFence fence, VkCommandBuffer commandBuffer);
};
