#include "command_submission_service.h"
#include "../core/vulkan_context.h"
#include "../core/vulkan_sync.h"
#include "../core/vulkan_swapchain.h"
#include "../core/vulkan_function_loader.h"
#include "../core/vulkan_utils.h"
#include "../core/queue_manager.h"
#include <iostream>

CommandSubmissionService::CommandSubmissionService() {
}

CommandSubmissionService::~CommandSubmissionService() {
    cleanup();
}

bool CommandSubmissionService::initialize(VulkanContext* context, VulkanSync* sync, VulkanSwapchain* swapchain, QueueManager* queueManager) {
    this->context = context;
    this->sync = sync;
    this->swapchain = swapchain;
    this->queueManager = queueManager;
    
    if (!queueManager) {
        std::cerr << "CommandSubmissionService: QueueManager is required!" << std::endl;
        return false;
    }
    
    std::cout << "CommandSubmissionService: Initialized with QueueManager" << std::endl;
    return true;
}

void CommandSubmissionService::cleanup() {
}

SubmissionResult CommandSubmissionService::submitFrame(
    uint32_t currentFrame,
    uint32_t imageIndex,
    const FrameGraph::ExecutionResult& executionResult,
    bool framebufferResized
) {
    SubmissionResult result;

    // ASYNC COMPUTE: Submit compute and graphics work in parallel
    // Compute and graphics record against the same frame index for consistent fencing
    
    // 1. Submit compute work asynchronously (no waiting for graphics)
    if (executionResult.computeCommandBufferUsed) {
        result = submitComputeWorkAsync(currentFrame);
        if (!result.success) {
            return result;
        }
    }

    // 2. Submit graphics work in parallel (uses previous frame's compute results)
    if (executionResult.graphicsCommandBufferUsed) {
        result = submitGraphicsWork(currentFrame, executionResult.computeCommandBufferUsed);
        if (!result.success) {
            return result;
        }

        // 3. Present frame
        result = presentFrame(currentFrame, imageIndex, framebufferResized);
    }

    return result;
}

SubmissionResult CommandSubmissionService::submitComputeWorkAsync(uint32_t computeFrame) {
    SubmissionResult result;

    // Use current frame index for command buffer selection (not computeFrame)
    uint32_t frameIndex = computeFrame % MAX_FRAMES_IN_FLIGHT;
    VkCommandBuffer computeCommandBuffer = queueManager->getComputeCommandBuffer(frameIndex);

    // Reset compute fence for this frame
    VkFence computeFence = sync->getComputeFence(frameIndex);
    // Cache loader and device references for performance
    const auto& vk = context->getLoader();
    const VkDevice device = context->getDevice();
    
    VkResult resetResult = vk.vkResetFences(device, 1, &computeFence);
    if (resetResult != VK_SUCCESS) {
        std::cerr << "CommandSubmissionService: Failed to reset compute fence: " << resetResult << std::endl;
        result.lastResult = resetResult;
        return result;
    }

    // Submit compute work using VulkanUtils
    // Signal compute-finished semaphore so graphics can wait when needed
    std::vector<VkCommandBuffer> computeCmdBuffers = {computeCommandBuffer};
    VkSemaphore computeFinishedSemaphore = sync->getComputeFinishedSemaphore(frameIndex);
    
    VkResult computeSubmitResult = VulkanUtils::submitCommands(
        queueManager->getComputeQueue(),
        vk,
        computeCmdBuffers,
        {}, // no wait semaphores
        {}, // no wait stages
        {computeFinishedSemaphore}, // signal compute completion
        computeFence
    );

    if (!VulkanUtils::checkVkResult(computeSubmitResult, "submit compute commands")) {
        result.lastResult = computeSubmitResult;
        return result;
    }
    
    // Record telemetry for successful compute submission
    queueManager->getTelemetry().recordSubmission(CommandPoolType::Compute);
    
    result.success = true;
    return result;
}

SubmissionResult CommandSubmissionService::submitGraphicsWork(uint32_t currentFrame, bool waitForCompute) {
    SubmissionResult result;

    // Cache loader and device references for performance
    const auto& vk = context->getLoader();
    const VkDevice device = context->getDevice();

    VkCommandBuffer graphicsCommandBuffer = queueManager->getGraphicsCommandBuffer(currentFrame);

    // Reset graphics fence
    VkFence graphicsFence = sync->getInFlightFence(currentFrame);
    VkResult resetResult = vk.vkResetFences(device, 1, &graphicsFence);
    if (resetResult != VK_SUCCESS) {
        std::cerr << "CommandSubmissionService: Failed to reset graphics fence: " << resetResult << std::endl;
        result.lastResult = resetResult;
        return result;
    }

    // Setup graphics submission - async compute model (no compute sync needed)
    VkSubmitInfo graphicsSubmitInfo{};
    graphicsSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    std::vector<VkSemaphore> waitSemaphores;
    std::vector<VkPipelineStageFlags> waitStages;
    waitSemaphores.push_back(sync->getImageAvailableSemaphore(currentFrame));
    waitStages.push_back(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
    if (waitForCompute) {
        waitSemaphores.push_back(sync->getComputeFinishedSemaphore(currentFrame));
        waitStages.push_back(VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_VERTEX_INPUT_BIT);
    }
    graphicsSubmitInfo.waitSemaphoreCount = static_cast<uint32_t>(waitSemaphores.size());
    graphicsSubmitInfo.pWaitSemaphores = waitSemaphores.data();
    graphicsSubmitInfo.pWaitDstStageMask = waitStages.data();
    graphicsSubmitInfo.commandBufferCount = 1;
    graphicsSubmitInfo.pCommandBuffers = &graphicsCommandBuffer;

    VkSemaphore signalSemaphores[] = {sync->getRenderFinishedSemaphores()[currentFrame]};
    graphicsSubmitInfo.signalSemaphoreCount = 1;
    graphicsSubmitInfo.pSignalSemaphores = signalSemaphores;

    VkResult graphicsSubmitResult = vk.vkQueueSubmit(
        queueManager->getGraphicsQueue(), 
        1, 
        &graphicsSubmitInfo, 
        graphicsFence
    );

    if (graphicsSubmitResult != VK_SUCCESS) {
        std::cerr << "CommandSubmissionService: Failed to submit graphics commands: " << graphicsSubmitResult << std::endl;
        result.lastResult = graphicsSubmitResult;
        return result;
    }
    
    // Record telemetry for successful graphics submission
    queueManager->getTelemetry().recordSubmission(CommandPoolType::Graphics);
    
    result.success = true;
    return result;
}

SubmissionResult CommandSubmissionService::presentFrame(uint32_t currentFrame, uint32_t imageIndex, bool framebufferResized) {
    SubmissionResult result;

    // Cache loader reference for performance
    const auto& vk = context->getLoader();

    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    
    VkSemaphore signalSemaphores[] = {sync->getRenderFinishedSemaphores()[currentFrame]};
    presentInfo.pWaitSemaphores = signalSemaphores;

    VkSwapchainKHR swapChains[] = {swapchain->getSwapchain()};
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapChains;
    presentInfo.pImageIndices = &imageIndex;

    VkResult presentResult = vk.vkQueuePresentKHR(queueManager->getPresentQueue(), &presentInfo);
    
    if (presentResult == VK_ERROR_OUT_OF_DATE_KHR || presentResult == VK_SUBOPTIMAL_KHR || framebufferResized) {
        result.swapchainRecreationNeeded = true;
        result.success = true; // Still successful, just needs recreation
    } else if (presentResult != VK_SUCCESS) {
        std::cerr << "CommandSubmissionService: Failed to present swap chain image: " << presentResult << std::endl;
        result.lastResult = presentResult;
        return result;
    } else {
        result.success = true;
    }

    result.lastResult = presentResult;
    return result;
}
