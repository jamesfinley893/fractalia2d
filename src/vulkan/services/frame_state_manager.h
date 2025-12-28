#pragma once

#include <vector>
#include <vulkan/vulkan.h>
#include "../core/vulkan_constants.h"

class VulkanSync;

class FrameStateManager {
public:
    FrameStateManager();
    ~FrameStateManager() = default;

    void initialize();
    void cleanup();

    // Frame state tracking
    // Mark both compute and graphics usage for the given slot index
    void updateFrameState(uint32_t frameIndex, bool computeUsed, bool graphicsUsed);
    // Fine-grained updates to avoid slot mix-ups when stages use different slots
    void setComputeUsed(uint32_t frameIndex, bool computeUsed);
    void setGraphicsUsed(uint32_t frameIndex, bool graphicsUsed);
    
    // Get fences that need to be waited on for the given slot index
    std::vector<VkFence> getFencesToWait(uint32_t frameIndex, VulkanSync* sync) const;
    
    // Check if any fences need waiting for the given slot index
    bool hasActiveFences(uint32_t frameIndex) const;

private:
    // Track usage per frame
    struct FrameState {
        bool computeUsed = true;  // Initialize to true for safety on first use of the slot
        bool graphicsUsed = true;
    };
    
    std::vector<FrameState> frameStates;
};
