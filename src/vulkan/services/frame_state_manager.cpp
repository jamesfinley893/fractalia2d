#include "frame_state_manager.h"
#include "../core/vulkan_sync.h"

FrameStateManager::FrameStateManager() {
    frameStates.resize(MAX_FRAMES_IN_FLIGHT);
}

void FrameStateManager::initialize() {
    // Reset all frame states
    for (auto& state : frameStates) {
        state.computeUsed = true;  // Initialize to true for safety on first frame
        state.graphicsUsed = true;
    }
}

void FrameStateManager::cleanup() {
    frameStates.clear();
}

void FrameStateManager::updateFrameState(uint32_t frameIndex, bool computeUsed, bool graphicsUsed) {
    if (frameIndex >= frameStates.size()) return;
    
    frameStates[frameIndex].computeUsed = computeUsed;
    frameStates[frameIndex].graphicsUsed = graphicsUsed;
}

std::vector<VkFence> FrameStateManager::getFencesToWait(uint32_t frameIndex, VulkanSync* sync) const {
    if (frameIndex >= frameStates.size() || !sync) {
        return {};
    }

    // Get the previous frame's state (circular buffer)
    uint32_t previousFrameIndex = (frameIndex == 0) ? MAX_FRAMES_IN_FLIGHT - 1 : frameIndex - 1;
    const auto& previousState = frameStates[previousFrameIndex];

    std::vector<VkFence> fencesToWait;
    fencesToWait.reserve(2);

    if (previousState.computeUsed) {
        fencesToWait.push_back(sync->getComputeFence(frameIndex));
    }
    if (previousState.graphicsUsed) {
        fencesToWait.push_back(sync->getInFlightFence(frameIndex));
    }

    return fencesToWait;
}

bool FrameStateManager::hasActiveFences(uint32_t frameIndex) const {
    if (frameIndex >= frameStates.size()) return false;
    
    uint32_t previousFrameIndex = (frameIndex == 0) ? MAX_FRAMES_IN_FLIGHT - 1 : frameIndex - 1;
    const auto& previousState = frameStates[previousFrameIndex];
    
    return previousState.computeUsed || previousState.graphicsUsed;
}