#include "swapchain_present_node.h"
#include "../core/vulkan_swapchain.h"
#include "../core/vulkan_context.h"
#include "../core/vulkan_function_loader.h"
#include <iostream>
#include <stdexcept>
#include <memory>

namespace {
    // Validation helper for presentation node
    class PresentationValidator {
    public:
        static bool validateDependencies(VulkanSwapchain* swapchain, const char* context) {
            if (!swapchain) {
                std::cerr << context << ": Missing swapchain dependency" << std::endl;
                return false;
            }
            return true;
        }
        
        static bool validateContext(const VulkanContext* context, const char* nodeContext) {
            if (!context) {
                std::cerr << nodeContext << ": Missing Vulkan context from frame graph" << std::endl;
                return false;
            }
            return true;
        }
    };
}

SwapchainPresentNode::SwapchainPresentNode(
    FrameGraphTypes::ResourceId colorTarget,
    VulkanSwapchain* swapchain
) : colorTargetId(colorTarget)
  , swapchain(swapchain) {
    
    // Validate constructor parameters for fail-fast behavior
    if (!swapchain) {
        throw std::invalid_argument("SwapchainPresentNode: swapchain cannot be null");
    }
}

std::vector<ResourceDependency> SwapchainPresentNode::getInputs() const {
    // ELEGANT SOLUTION: Use dynamic swapchain image ID resolved each frame
    return {
        {currentSwapchainImageId, ResourceAccess::Read, PipelineStage::ColorAttachment},
    };
}

std::vector<ResourceDependency> SwapchainPresentNode::getOutputs() const {
    // Present node doesn't produce frame graph resources, it presents to swapchain
    return {};
}

void SwapchainPresentNode::execute(VkCommandBuffer commandBuffer, const FrameGraph& frameGraph) {
    // Validate runtime dependencies
    if (!swapchain) {
        std::cerr << "SwapchainPresentNode::execute: Critical error - swapchain became null during execution" << std::endl;
        return;
    }
    
    const VulkanContext* context = frameGraph.getContext();
    if (!PresentationValidator::validateContext(context, "SwapchainPresentNode::execute")) {
        return;
    }
    
    // Validate image index bounds
    uint32_t imageCount = swapchain->getImages().size();
    if (imageIndex >= imageCount) {
        std::cerr << "SwapchainPresentNode: Invalid image index " << imageIndex 
                  << " (max: " << imageCount << ")" << std::endl;
        return;
    }
    
    // Note: This node doesn't perform command buffer operations.
    // Presentation occurs at queue submission level, handled by frame graph execution.
    // This node's primary function is dependency declaration for proper synchronization.
    
    // Presentation dependency successfully established
}

// Node lifecycle implementation
bool SwapchainPresentNode::initializeNode(const FrameGraph& frameGraph) {
    // One-time initialization - validate dependencies
    if (!swapchain) {
        std::cerr << "SwapchainPresentNode: VulkanSwapchain is null" << std::endl;
        return false;
    }
    return true;
}

void SwapchainPresentNode::prepareFrame(const FrameContext& frameContext) {
    // Per-frame preparation - validate image index (timing data not needed for present node)
    uint32_t imageCount = swapchain ? swapchain->getImages().size() : 0;
    if (imageIndex >= imageCount) {
        std::cerr << "SwapchainPresentNode: Invalid image index " << imageIndex 
                  << " (max: " << imageCount << ")" << std::endl;
    }
}

void SwapchainPresentNode::releaseFrame(uint32_t frameIndex) {
    // Per-frame cleanup - nothing to clean up for present node
}
