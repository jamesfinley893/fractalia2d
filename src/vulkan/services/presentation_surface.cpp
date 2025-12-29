#include "presentation_surface.h"
#include "../core/vulkan_context.h"
#include "../core/vulkan_swapchain.h"
#include "../pipelines/graphics_pipeline_manager.h"
#include "../core/vulkan_function_loader.h"
#include "../core/vulkan_constants.h"
#include "../core/vulkan_sync.h"
#include "gpu_synchronization_service.h"
#include <iostream>

PresentationSurface::PresentationSurface() {
}

PresentationSurface::~PresentationSurface() {
    cleanup();
}

bool PresentationSurface::initialize(
    VulkanContext* context,
    VulkanSwapchain* swapchain,
    GraphicsPipelineManager* graphicsManager,
    GPUSynchronizationService* syncManager,
    VulkanSync* sync
) {
    this->context = context;
    this->swapchain = swapchain;
    this->graphicsManager = graphicsManager;
    this->syncManager = syncManager;
    this->sync = sync;
    if (!sync) {
        std::cerr << "PresentationSurface: VulkanSync is required for image acquisition" << std::endl;
        return false;
    }
    return true;
}

void PresentationSurface::cleanup() {
}

SurfaceAcquisitionResult PresentationSurface::acquireNextImage(uint32_t currentFrame) {
    SurfaceAcquisitionResult result;

    // Prevent concurrent acquisition attempts
    if (acquisitionInProgress) {
        std::cerr << "PresentationSurface: Warning - Concurrent image acquisition attempt blocked" << std::endl;
        return result; // success = false
    }

    if (framebufferResized) {
        result.recreationNeeded = true;
        return result;
    }

    // Set acquisition guard
    acquisitionInProgress = true;

    // Use reasonable timeout to prevent infinite waits
    const uint64_t timeoutNs = FENCE_TIMEOUT_2_SECONDS;
    
    VkSemaphore imageAvailableSemaphore = sync ? sync->getImageAvailableSemaphore(currentFrame) : VK_NULL_HANDLE;
    VkResult acquireResult = context->getLoader().vkAcquireNextImageKHR(
        context->getDevice(),
        swapchain->getSwapchain(),
        timeoutNs,
        imageAvailableSemaphore,
        VK_NULL_HANDLE,
        &result.imageIndex
    );

    result.result = acquireResult;

    // Clear acquisition guard before any return
    acquisitionInProgress = false;

    if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR) {
        result.recreationNeeded = true;
        return result;
    } else if (acquireResult == VK_TIMEOUT) {
        std::cerr << "PresentationSurface: Timeout waiting for swapchain image acquisition" << std::endl;
        return result; // success = false, proper timeout handling
    } else if (acquireResult != VK_SUCCESS && acquireResult != VK_SUBOPTIMAL_KHR) {
        std::cerr << "PresentationSurface: Failed to acquire swap chain image: " << acquireResult << std::endl;
        return result;
    }

    result.success = true;
    return result;
}

bool PresentationSurface::recreateSwapchain() {
    // Prevent concurrent recreation attempts
    if (recreationInProgress) {
        std::cout << "PresentationSurface: Recreation already in progress, skipping..." << std::endl;
        return true;
    }
    recreationInProgress = true;

    
    // CRITICAL FIX FOR SECOND RESIZE CRASH: Skip redundant fence wait
    // VulkanRenderer already called syncService->waitForAllFrames() and reset fences
    // Calling waitForAllFrames() again on already-reset fences causes undefined behavior/crash
    // if (!syncManager->waitForAllFrames()) {
    //     recreationInProgress = false;
    //     return false;
    // }

    // Clear graphics pipeline cache to prevent corruption
    graphicsManager->clearCache();
    
    // Recreate render pass for new swapchain format
    currentRenderPass = graphicsManager->createRenderPass(
        swapchain->getImageFormat(), VK_FORMAT_UNDEFINED, VK_SAMPLE_COUNT_2_BIT, true);
    if (currentRenderPass == VK_NULL_HANDLE) {
        std::cerr << "PresentationSurface: Failed to recreate render pass" << std::endl;
        recreationInProgress = false;
        return false;
    }
    
    // CRITICAL FIX FOR SECOND RESIZE CRASH: Recreate pipeline cache after render pass recreation
    // This addresses VkPipelineCache internal corruption that survives first resize but crashes on second
    if (!graphicsManager->recreatePipelineCache()) {
        std::cerr << "PresentationSurface: CRITICAL ERROR - Failed to recreate graphics pipeline cache during swapchain recreation" << std::endl;
        std::cerr << "  This may cause pipeline creation failures or crashes in subsequent frames" << std::endl;
        // Continue anyway - this is better than failing entirely
    }

    // Recreate swapchain
    if (!swapchain->recreate(currentRenderPass)) {
        recreationInProgress = false;
        return false;
    }

    // Recreate framebuffers
    if (!swapchain->createFramebuffers(currentRenderPass)) {
        recreationInProgress = false;
        return false;
    }

    framebufferResized = false;
    recreationInProgress = false;
    return true;
}
