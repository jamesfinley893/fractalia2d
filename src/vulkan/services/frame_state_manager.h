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
    void updateFrameState(uint32_t frameIndex, bool computeUsed, bool graphicsUsed);
    
    // Get fences that need to be waited on for the given frame
    std::vector<VkFence> getFencesToWait(uint32_t frameIndex, VulkanSync* sync) const;
    
    // Check if any fences need waiting
    bool hasActiveFences(uint32_t frameIndex) const;

private:
    // Track usage per frame
    struct FrameState {
        bool computeUsed = true;  // Initialize to true for safety on first frame
        bool graphicsUsed = true;
    };
    
    std::vector<FrameState> frameStates;
};