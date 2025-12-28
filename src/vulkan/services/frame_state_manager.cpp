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

void FrameStateManager::setComputeUsed(uint32_t frameIndex, bool computeUsed) {
    if (frameIndex >= frameStates.size()) return;
    frameStates[frameIndex].computeUsed = computeUsed;
}

void FrameStateManager::setGraphicsUsed(uint32_t frameIndex, bool graphicsUsed) {
    if (frameIndex >= frameStates.size()) return;
    frameStates[frameIndex].graphicsUsed = graphicsUsed;
}

std::vector<VkFence> FrameStateManager::getFencesToWait(uint32_t frameIndex, VulkanSync* sync) const {
    if (frameIndex >= frameStates.size() || !sync) {
        return {};
    }

    // We are about to reuse the resources in this slot index. If the slot
    // has outstanding work recorded the last time it was used, wait on its fences.
    const auto& slotState = frameStates[frameIndex];

    std::vector<VkFence> fencesToWait;
    fencesToWait.reserve(2);

    if (slotState.computeUsed) {
        fencesToWait.push_back(sync->getComputeFence(frameIndex));
    }
    if (slotState.graphicsUsed) {
        fencesToWait.push_back(sync->getInFlightFence(frameIndex));
    }

    return fencesToWait;
}

bool FrameStateManager::hasActiveFences(uint32_t frameIndex) const {
    if (frameIndex >= frameStates.size()) return false;

    const auto& slotState = frameStates[frameIndex];
    return slotState.computeUsed || slotState.graphicsUsed;
}
