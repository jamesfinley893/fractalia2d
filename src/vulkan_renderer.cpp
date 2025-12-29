#include "vulkan_renderer.h"
#include "vulkan/core/vulkan_function_loader.h"
#include "vulkan/core/vulkan_context.h"
#include "vulkan/core/vulkan_swapchain.h"
#include "vulkan/core/vulkan_sync.h"
#include "vulkan/core/queue_manager.h"
#include "vulkan/resources/core/resource_coordinator.h"
#include "vulkan/resources/managers/graphics_resource_manager.h"
#include "vulkan/rendering/frame_graph.h"
#include "vulkan/nodes/entity_compute_node.h"
#include "vulkan/nodes/entity_graphics_node.h"
#include "vulkan/nodes/swapchain_present_node.h"
#include "vulkan/services/render_frame_director.h"
#include "vulkan/services/command_submission_service.h"
#include "vulkan/rendering/frame_graph_resource_registry.h"
#include "vulkan/services/gpu_synchronization_service.h"
#include "vulkan/services/presentation_surface.h"
#include "vulkan/services/frame_state_manager.h"
#include "vulkan/services/error_recovery_service.h"
#include "vulkan/pipelines/pipeline_system_manager.h"
#include "ecs/gpu/gpu_entity_manager.h"
#include "ecs/components/component.h"
#include "ecs/components/camera_component.h"
#include <iostream>
#include <array>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

float VulkanRenderer::clampedDeltaTime = 0.0f;

VulkanRenderer::VulkanRenderer() {
}

VulkanRenderer::~VulkanRenderer() {
    cleanup();
}

bool VulkanRenderer::initialize(SDL_Window* window) {
    if (!window) {
        std::cerr << "VulkanRenderer: NULL window provided" << std::endl;
        return false;
    }
    this->window = window;
    
    // Phase 1: Core Vulkan initialization
    context = std::make_unique<VulkanContext>();
    if (!context || !context->initialize(window)) {
        std::cerr << "Failed to initialize Vulkan context" << std::endl;
        cleanup();
        return false;
    }
    
    swapchain = std::make_unique<VulkanSwapchain>();
    if (!swapchain || !swapchain->initialize(*context, window)) {
        std::cerr << "Failed to initialize Vulkan swapchain" << std::endl;
        cleanup();
        return false;
    }
    
    // Phase 2: Pipeline and synchronization objects (depend on context)
    pipelineSystem = std::make_unique<PipelineSystemManager>();
    if (!pipelineSystem || !pipelineSystem->initialize(*context)) {
        std::cerr << "Failed to initialize AAA Pipeline System" << std::endl;
        cleanup();
        return false;
    }
    
    sync = std::make_unique<VulkanSync>();
    if (!sync || !sync->initialize(*context)) {
        std::cerr << "Failed to initialize Vulkan sync" << std::endl;
        cleanup();
        return false;
    }
    
    queueManager = std::make_unique<QueueManager>();
    if (!queueManager || !queueManager->initialize(*context)) {
        std::cerr << "Failed to initialize Queue Manager" << std::endl;
        cleanup();
        return false;
    }
    
    // Phase 3: Render pass and framebuffers (depend on pipeline system and swapchain)
    if (!pipelineSystem->getGraphicsManager()) {
        std::cerr << "Graphics manager not available from pipeline system" << std::endl;
        cleanup();
        return false;
    }
    
    VkRenderPass renderPass = pipelineSystem->getGraphicsManager()->createRenderPass(
        swapchain->getImageFormat(), VK_FORMAT_UNDEFINED, VK_SAMPLE_COUNT_2_BIT, true);
    if (renderPass == VK_NULL_HANDLE) {
        std::cerr << "Failed to create render pass" << std::endl;
        cleanup();
        return false;
    }
    
    if (!swapchain->createFramebuffers(renderPass)) {
        std::cerr << "Failed to create framebuffers" << std::endl;
        cleanup();
        return false;
    }
    
    // Phase 4: Resource management (depends on context, queue manager)
    resourceCoordinator = std::make_unique<ResourceCoordinator>();
    if (!resourceCoordinator || !resourceCoordinator->initialize(*context, queueManager.get())) {
        std::cerr << "Failed to initialize Resource coordinator" << std::endl;
        cleanup();
        return false;
    }
    
    if (!resourceCoordinator->getGraphicsManager()->createAllGraphicsResources()) {
        std::cerr << "Failed to create graphics resources (uniform and triangle buffers)" << std::endl;
        cleanup();
        return false;
    }
    
    // Phase 5: Descriptor layouts and pools (depend on pipeline system)
    if (!pipelineSystem->getLayoutManager()) {
        std::cerr << "Layout manager not available from pipeline system" << std::endl;
        cleanup();
        return false;
    }
    
    auto layoutSpec = DescriptorLayoutPresets::createEntityGraphicsLayout();
    VkDescriptorSetLayout descriptorLayout = pipelineSystem->getLayoutManager()->getLayout(layoutSpec);
    if (descriptorLayout == VK_NULL_HANDLE) {
        std::cerr << "Failed to create descriptor layout" << std::endl;
        cleanup();
        return false;
    }
    
    if (!resourceCoordinator->getGraphicsManager()->createGraphicsDescriptorPool(descriptorLayout)) {
        std::cerr << "Failed to create descriptor pool" << std::endl;
        cleanup();
        return false;
    }
    
    if (!resourceCoordinator->getGraphicsManager()->createGraphicsDescriptorSets(descriptorLayout)) {
        std::cerr << "Failed to create descriptor sets" << std::endl;
        cleanup();
        return false;
    }
    
    // Phase 6: Entity management (depends on context, sync, resource context)
    gpuEntityManager = std::make_unique<GPUEntityManager>();
    if (!gpuEntityManager || !gpuEntityManager->initialize(*context, sync.get(), resourceCoordinator.get())) {
        std::cerr << "Failed to initialize GPU entity manager" << std::endl;
        cleanup();
        return false;
    }
    
    // Validate entity manager buffers before using them
    if (!gpuEntityManager->getMovementParamsBuffer() || !gpuEntityManager->getPositionBuffer()) {
        std::cerr << "GPU entity manager missing required buffers" << std::endl;
        cleanup();
        return false;
    }
    
    if (!resourceCoordinator->getGraphicsManager()->updateDescriptorSetsWithEntityAndPositionBuffers(
            gpuEntityManager->getMovementParamsBuffer(),
            gpuEntityManager->getPositionBuffer())) {
        std::cerr << "Failed to update descriptor sets with entity and position buffers" << std::endl;
        cleanup();
        return false;
    }
    std::cout << "Graphics descriptor sets updated with entity and position buffers" << std::endl;
    
    auto computeLayoutSpec = DescriptorLayoutPresets::createEntityComputeLayout();
    VkDescriptorSetLayout computeDescriptorLayout = pipelineSystem->getLayoutManager()->getLayout(computeLayoutSpec);
    if (computeDescriptorLayout == VK_NULL_HANDLE) {
        std::cerr << "Failed to create compute descriptor layout" << std::endl;
        cleanup();
        return false;
    }
    
    if (!gpuEntityManager->getDescriptorManager().createComputeDescriptorSets(computeDescriptorLayout)) {
        std::cerr << "Failed to create compute descriptor sets" << std::endl;
        cleanup();
        return false;
    }
    
    // Create graphics descriptor sets for entity rendering with unified layout
    auto graphicsLayoutSpec = DescriptorLayoutPresets::createEntityGraphicsLayout();
    VkDescriptorSetLayout graphicsDescriptorLayout = pipelineSystem->getLayoutManager()->getLayout(graphicsLayoutSpec);
    if (!gpuEntityManager->getDescriptorManager().createGraphicsDescriptorSets(graphicsDescriptorLayout)) {
        std::cerr << "Failed to create entity graphics descriptor sets" << std::endl;
        cleanup();
        return false;
    }
    
    // Phase 7: Modular architecture (depends on all previous components)
    if (!initializeModularArchitecture()) {
        std::cerr << "Failed to initialize modular architecture" << std::endl;
        cleanup();
        return false;
    }
    
    // Final validation - ensure all critical components are available
    if (!frameDirector || !submissionService || !frameStateManager || !errorRecoveryService) {
        std::cerr << "Initialization state validation failed - missing critical services" << std::endl;
        cleanup();
        return false;
    }
    
    pipelineSystem->warmupCommonPipelines();
    
    std::cout << "VulkanRenderer: AAA Pipeline System initialization complete" << std::endl;
    
    initialized = true;
    return true;
}

void VulkanRenderer::cleanup() {
    // Wait for device to be idle before cleanup - but only if context is valid
    if (context && context->getDevice() != VK_NULL_HANDLE) {
        try {
            const auto& vk = context->getLoader();
            const VkDevice device = context->getDevice();
            vk.vkDeviceWaitIdle(device);
        } catch (const std::exception& e) {
            std::cerr << "Exception during device wait idle: " << e.what() << std::endl;
        }
    }
    
    // Cleanup modular architecture first (higher-level components)
    cleanupModularArchitecture();
    
    // Cleanup RAII resources before destroying their dependencies
    if (sync) {
        try {
            sync->cleanupBeforeContextDestruction();
        } catch (const std::exception& e) {
            std::cerr << "Exception during sync cleanup: " << e.what() << std::endl;
        }
    }
    
    if (pipelineSystem) {
        try {
            pipelineSystem->cleanupBeforeContextDestruction();
        } catch (const std::exception& e) {
            std::cerr << "Exception during pipeline system cleanup: " << e.what() << std::endl;
        }
    }
    
    // Cleanup ResourceCoordinator RAII resources before destroying dependencies
    if (resourceCoordinator) {
        try {
            resourceCoordinator->cleanupBeforeContextDestruction();
        } catch (const std::exception& e) {
            std::cerr << "Exception during resource coordinator cleanup: " << e.what() << std::endl;
        }
    }
    
    // Reset components in reverse dependency order
    
    if (gpuEntityManager) {
        gpuEntityManager.reset();
    }
    
    if (resourceCoordinator) {
        resourceCoordinator.reset();
    }
    
    if (queueManager) {
        queueManager.reset();
    }
    
    if (sync) {
        sync.reset();
    }
    
    if (pipelineSystem) {
        pipelineSystem.reset();
    }
    
    if (swapchain) {
        swapchain.reset();
    }
    
    if (context) {
        context.reset();
    }
    
    initialized = false;
    window = nullptr;
}

void VulkanRenderer::drawFrame() {
    drawFrameModular();
}


bool VulkanRenderer::initializeModularArchitecture() {
    frameGraph = std::make_unique<FrameGraph>();
    if (!frameGraph->initialize(*context, sync.get(), queueManager.get())) {
        std::cerr << "Failed to initialize frame graph" << std::endl;
        return false;
    }
    
    resourceRegistry = std::make_unique<FrameGraphResourceRegistry>();
    if (!resourceRegistry->initialize(frameGraph.get(), gpuEntityManager.get())) {
        std::cerr << "Failed to initialize resource importer" << std::endl;
        return false;
    }
    
    if (!resourceRegistry->importEntityResources()) {
        std::cerr << "Failed to import entity resources" << std::endl;
        return false;
    }
    
    syncService = std::make_unique<GPUSynchronizationService>();
    if (!syncService->initialize(*context)) {
        std::cerr << "Failed to initialize synchronization manager" << std::endl;
        return false;
    }
    
    presentationSurface = std::make_unique<PresentationSurface>();
    if (!presentationSurface->initialize(context.get(), swapchain.get(), pipelineSystem->getGraphicsManager(), syncService.get())) {
        std::cerr << "Failed to initialize swapchain coordinator" << std::endl;
        return false;
    }
    
    frameDirector = std::make_unique<RenderFrameDirector>();
    if (!frameDirector->initialize(
        context.get(),
        swapchain.get(), 
        pipelineSystem.get(),
        sync.get(),
        resourceCoordinator.get(),
        gpuEntityManager.get(),
        frameGraph.get(),
        presentationSurface.get()
    )) {
        std::cerr << "Failed to initialize frame orchestrator" << std::endl;
        return false;
    }
    
    frameDirector->updateResourceIds(
        resourceRegistry->getEntityBufferId(),
        resourceRegistry->getPositionBufferId(),
        resourceRegistry->getCurrentPositionBufferId(),
        resourceRegistry->getTargetPositionBufferId()
    );
    
    submissionService = std::make_unique<CommandSubmissionService>();
    if (!submissionService->initialize(context.get(), sync.get(), swapchain.get(), queueManager.get())) {
        std::cerr << "Failed to initialize queue submission manager" << std::endl;
        return false;
    }
    
    frameStateManager = std::make_unique<FrameStateManager>();
    frameStateManager->initialize();
    
    errorRecoveryService = std::make_unique<ErrorRecoveryService>();
    errorRecoveryService->initialize(presentationSurface.get());
    
    std::cout << "Modular architecture initialized successfully" << std::endl;
    return true;
}

void VulkanRenderer::cleanupModularArchitecture() {
    errorRecoveryService.reset();
    frameStateManager.reset();
    submissionService.reset();
    frameDirector.reset();
    presentationSurface.reset();
    syncService.reset();
    resourceRegistry.reset();
    frameGraph.reset();
}

void VulkanRenderer::drawFrameModular() {
    // Wait for previous frame GPU work to complete using FrameStateManager
    if (frameStateManager && frameStateManager->hasActiveFences(currentFrame)) {
        auto fencesToWait = frameStateManager->getFencesToWait(currentFrame, sync.get());
        
        if (!fencesToWait.empty()) {
            const auto& vk = context->getLoader();
            const VkDevice device = context->getDevice();
            
            VkResult waitResult = vk.vkWaitForFences(device, static_cast<uint32_t>(fencesToWait.size()), 
                                                    fencesToWait.data(), VK_TRUE, UINT64_MAX);
            if (waitResult != VK_SUCCESS) {
                std::cerr << "VulkanRenderer: Failed to wait for GPU fences: " << waitResult << std::endl;
                return;
            }
        }
    }
    
    // Upload pending GPU entities
    if (gpuEntityManager && gpuEntityManager->hasPendingUploads()) {
        gpuEntityManager->uploadPendingEntities();
    }
    
    // Orchestrate the frame
    auto frameResult = frameDirector->directFrame(
        currentFrame,
        totalTime,
        deltaTime, 
        frameCounter,
        world
    );
    
    if (!frameResult.success) {
        RenderFrameResult retryResult = {};
        if (errorRecoveryService && errorRecoveryService->handleFrameFailure(
            frameResult, frameDirector.get(), currentFrame, totalTime, deltaTime, frameCounter, world, retryResult)) {
            frameResult = retryResult;
        } else {
            return;
        }
    } else {
        logFrameSuccessIfNeeded("Frame direction completed successfully");
    }
    
    // Note: Frame graph nodes already configured in directFrame() - no need to configure again
    
    // Submit frame work
    auto submissionResult = submissionService->submitFrame(
        currentFrame,
        frameResult.imageIndex,
        frameResult.executionResult,
        framebufferResized
    );
    
    if (!submissionResult.success) {
        std::cerr << "VulkanRenderer: Frame " << frameCounter << " FAILED in submissionService->submitFrame()" << std::endl;
        std::cerr << "  VkResult: " << submissionResult.lastResult << std::endl;
        return;
    } else {
        logFrameSuccessIfNeeded("Frame submission completed successfully");
    }
    
    if (submissionResult.swapchainRecreationNeeded || framebufferResized) {
        std::cout << "VulkanRenderer: SWAPCHAIN RECREATION INITIATED - Frame " << frameCounter << std::endl;
        
        if (presentationSurface && presentationSurface->recreateSwapchain()) {
            std::cout << "VulkanRenderer: SWAPCHAIN RECREATION COMPLETED - Next frames should render normally" << std::endl;
            framebufferResized = false;  // Reset the flag
        } else {
            std::cerr << "VulkanRenderer: CRITICAL ERROR - Swapchain recreation FAILED" << std::endl;
        }
    }
    
    // Periodic memory pressure monitoring (every 60 frames to avoid performance impact)
    if (frameCounter % 60 == 0 && resourceCoordinator) {
        bool memoryPressure = resourceCoordinator->isUnderMemoryPressure();
        if (memoryPressure) {
            VkDeviceSize totalAllocated = resourceCoordinator->getTotalAllocatedMemory();
            VkDeviceSize available = resourceCoordinator->getAvailableMemory();
            uint32_t allocCount = resourceCoordinator->getAllocationCount();
            std::cout << "VulkanRenderer: Frame " << frameCounter << " - Memory pressure status: HIGH" 
                      << ", Total allocated: " << (totalAllocated / (1024 * 1024)) << "MB"
                      << ", Available: " << (available / (1024 * 1024)) << "MB"
                      << ", Active allocations: " << allocCount << std::endl;
        }
    }
    
    // Update frame state tracking for next frame optimization
    // Only update if frame was successful to avoid tracking invalid state
    if (frameResult.success && submissionResult.success && frameStateManager) {
        frameStateManager->updateFrameState(
            currentFrame,
            frameResult.executionResult.computeCommandBufferUsed,
            frameResult.executionResult.graphicsCommandBufferUsed
        );
    }
    
    totalTime += deltaTime;
    frameCounter++;
    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}


void VulkanRenderer::updateAspectRatio(int windowWidth, int windowHeight) {
    // Camera aspect ratio updates are now handled by CameraService
    // This method is kept for renderer-specific aspect ratio handling if needed
}

void VulkanRenderer::setFramebufferResized(bool resized) {
    framebufferResized = resized;
    if (presentationSurface) {
        presentationSurface->setFramebufferResized(resized);
    }
}

void VulkanRenderer::logFrameSuccessIfNeeded(const char* operation) {
    // Monitor first few frames after resize with consolidated logging
    static uint32_t lastRecreationFrame = 0;
    static uint32_t framesAfterRecreation = 0;
    
    if (frameCounter - lastRecreationFrame <= 10 && lastRecreationFrame > 0) {
        framesAfterRecreation = frameCounter - lastRecreationFrame;
        
        std::cout << "VulkanRenderer: Frame " << frameCounter << " SUCCESS (+" << framesAfterRecreation 
                 << " frames post-resize) - " << operation << std::endl;
    }
}