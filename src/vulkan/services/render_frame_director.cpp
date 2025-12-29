#include "render_frame_director.h"
#include "presentation_surface.h"
#include "../core/vulkan_context.h"
#include "../core/vulkan_swapchain.h"
#include "../pipelines/pipeline_system_manager.h"
#include "../core/vulkan_sync.h"
#include "../resources/core/resource_coordinator.h"
#include "../resources/managers/graphics_resource_manager.h"
#include "../nodes/entity_compute_node.h"
#include "../nodes/physics_compute_node.h"
#include "../nodes/entity_graphics_node.h"
#include "../nodes/swapchain_present_node.h"
#include "../../ecs/gpu/gpu_entity_manager.h"
#include <iostream>

RenderFrameDirector::RenderFrameDirector() {
}

RenderFrameDirector::~RenderFrameDirector() {
    cleanup();
}

bool RenderFrameDirector::initialize(
    VulkanContext* context,
    VulkanSwapchain* swapchain,
    PipelineSystemManager* pipelineSystem,
    VulkanSync* sync,
    ResourceCoordinator* resourceCoordinator,
    GPUEntityManager* gpuEntityManager,
    FrameGraph* frameGraph,
    PresentationSurface* presentationSurface
) {
    this->context = context;
    this->swapchain = swapchain;
    this->pipelineSystem = pipelineSystem;
    this->sync = sync;
    this->resourceCoordinator = resourceCoordinator;
    this->gpuEntityManager = gpuEntityManager;
    this->frameGraph = frameGraph;
    this->presentationSurface = presentationSurface;

    return true;
}

void RenderFrameDirector::cleanup() {
    // Nothing to cleanup, dependencies are managed externally
}

RenderFrameResult RenderFrameDirector::directFrame(
    uint32_t currentFrame,
    float totalTime,
    float deltaTime,
    uint32_t frameCounter,
    flecs::world* world
) {
    RenderFrameResult result;

    // 1. Acquire swapchain image using PresentationSurface
    SurfaceAcquisitionResult acquisitionResult = presentationSurface->acquireNextImage(currentFrame);
    if (!acquisitionResult.success) {
        if (acquisitionResult.recreationNeeded) {
            std::cout << "RenderFrameDirector: Swapchain recreation needed, skipping frame" << std::endl;
        }
        return result; // Failed to acquire image
    }
    result.imageIndex = acquisitionResult.imageIndex;

    // 2. Setup frame graph
    setupFrameGraph(result.imageIndex);

    // 3. Compile frame graph (don't execute yet)
    if (!compileFrameGraph(currentFrame, totalTime, deltaTime, frameCounter)) {
        return result;
    }

    // 4. Configure frame graph nodes with world reference after swapchain acquisition
    configureFrameGraphNodes(result.imageIndex, world);

    // 5. Execute frame graph with timing data and global frame counter
    uint32_t globalFrame = globalFrameCounter_.fetch_add(1, std::memory_order_relaxed);
    result.executionResult = frameGraph->execute(currentFrame, totalTime, deltaTime, globalFrame);
    result.success = true;

    return result;
}

void RenderFrameDirector::updateResourceIds(
    FrameGraphTypes::ResourceId entityBufferId,
    FrameGraphTypes::ResourceId positionBufferId,
    FrameGraphTypes::ResourceId currentPositionBufferId,
    FrameGraphTypes::ResourceId targetPositionBufferId
) {
    this->entityBufferId = entityBufferId;
    this->positionBufferId = positionBufferId;
    this->currentPositionBufferId = currentPositionBufferId;
    this->targetPositionBufferId = targetPositionBufferId;
}


void RenderFrameDirector::setupFrameGraph(uint32_t imageIndex) {
    // Only reset frame graph if not already compiled to avoid recompilation every frame
    bool needsInitialization = !frameGraphInitialized;
    if (needsInitialization) {
        frameGraph->reset();
        
        // Initialize swapchain image resource ID cache
        swapchainImageIds.resize(swapchain->getImages().size(), 0);
        
        std::cout << "RenderFrameDirector: Initializing frame graph for first time" << std::endl;
    }
    
    // Import current swapchain image only if not already cached
    if (swapchainImageIds[imageIndex] == 0) {
        VkImage swapchainImage = swapchain->getImages()[imageIndex];
        VkImageView swapchainImageView = swapchain->getImageViews()[imageIndex];
        std::string swapchainName = "SwapchainImage_" + std::to_string(imageIndex);
        swapchainImageIds[imageIndex] = frameGraph->importExternalImage(
            swapchainName,
            swapchainImage,
            swapchainImageView,
            swapchain->getImageFormat(),
            swapchain->getExtent()
        );
    }
    
    swapchainImageId = swapchainImageIds[imageIndex];
    
    // Add nodes to frame graph only once during initialization
    if (needsInitialization) {
        // Movement compute node (sets velocity every 900 frames)
        computeNodeId = frameGraph->addNode<EntityComputeNode>(
            entityBufferId,
            positionBufferId,
            currentPositionBufferId,
            targetPositionBufferId,
            pipelineSystem->getComputeManager(),
            gpuEntityManager
        );
        
        // Physics compute node (updates positions based on velocity every frame)
        physicsNodeId = frameGraph->addNode<PhysicsComputeNode>(
            entityBufferId,
            positionBufferId,
            currentPositionBufferId,
            targetPositionBufferId,
            pipelineSystem->getComputeManager(),
            gpuEntityManager
        );
        
        // ELEGANT SOLUTION: Pass a dynamic swapchain image reference
        // Nodes will resolve the actual resource ID at execution time
        graphicsNodeId = frameGraph->addNode<EntityGraphicsNode>(
            entityBufferId,
            positionBufferId,
            0, // Placeholder - will be resolved dynamically
            pipelineSystem->getGraphicsManager(),
            swapchain,
            resourceCoordinator,
            gpuEntityManager
        );
        
        presentNodeId = frameGraph->addNode<SwapchainPresentNode>(
            0, // Placeholder - will be resolved dynamically  
            swapchain
        );
        
        // Mark as initialized after nodes are added
        frameGraphInitialized = true;
        std::cout << "RenderFrameDirector: Created nodes - Compute:" << computeNodeId 
                  << " Physics:" << physicsNodeId << " Graphics:" << graphicsNodeId 
                  << " Present:" << presentNodeId << std::endl;
    }
    
    // Configure nodes with frame-specific data will be done externally
}

void RenderFrameDirector::configureFrameGraphNodes(uint32_t imageIndex, flecs::world* world) {
    // ELEGANT ORCHESTRATION: Configure nodes with current frame's swapchain image
    if (auto* graphicsNode = frameGraph->getNode<EntityGraphicsNode>(graphicsNodeId)) {
        graphicsNode->setImageIndex(imageIndex);
        graphicsNode->setCurrentSwapchainImageId(swapchainImageId); // Dynamic resolution
        graphicsNode->setWorld(world);
    }
    
    if (auto* presentNode = frameGraph->getNode<SwapchainPresentNode>(presentNodeId)) {
        presentNode->setImageIndex(imageIndex);
        presentNode->setCurrentSwapchainImageId(swapchainImageId); // Dynamic resolution
    }
}

void RenderFrameDirector::configureNodes(
    FrameGraphTypes::NodeId graphicsNodeId, 
    FrameGraphTypes::NodeId presentNodeId, 
    uint32_t imageIndex, 
    flecs::world* world
) {
    // Set the correct image index for the current frame and world reference
    if (EntityGraphicsNode* graphicsNode = frameGraph->getNode<EntityGraphicsNode>(graphicsNodeId)) {
        graphicsNode->setImageIndex(imageIndex);
        graphicsNode->setWorld(world);
    }
    if (SwapchainPresentNode* presentNode = frameGraph->getNode<SwapchainPresentNode>(presentNodeId)) {
        presentNode->setImageIndex(imageIndex);
    }
}

void RenderFrameDirector::resetSwapchainCache() {
    // ELEGANT SOLUTION: Swapchain recreation is now seamless
    // 1. Remove old swapchain images from frame graph
    frameGraph->removeSwapchainResources();
    
    // 2. Reset cache for new swapchain size  
    swapchainImageIds.assign(swapchain->getImages().size(), 0);
    
    // Command pool management is now handled by QueueManager - no manual recreation needed
    std::cout << "RenderFrameDirector: Swapchain cache reset complete (QueueManager handles command pools)" << std::endl;
    
    // 4. CRITICAL FIX: Update BOTH graphics AND compute descriptor sets after swapchain recreation
    // This fixes the second window resize crash by ensuring all descriptor sets have valid buffer bindings
    if (gpuEntityManager && resourceCoordinator) {
        VkBuffer movementParamsBuffer = gpuEntityManager->getMovementParamsBuffer();
        VkBuffer positionBuffer = gpuEntityManager->getPositionBuffer();
        
        if (movementParamsBuffer != VK_NULL_HANDLE && positionBuffer != VK_NULL_HANDLE) {
            bool graphicsSuccess = resourceCoordinator->getGraphicsManager()->updateDescriptorSetsWithEntityAndPositionBuffers(movementParamsBuffer, positionBuffer);
            bool computeSuccess = gpuEntityManager->getDescriptorManager().recreateDescriptorSets();
            
            if (graphicsSuccess && computeSuccess) {
                std::cout << "RenderFrameDirector: Successfully updated graphics AND compute descriptor sets after swapchain recreation" << std::endl;
            } else {
                std::cerr << "RenderFrameDirector: ERROR - Failed to update descriptor sets after swapchain recreation!" << std::endl;
                std::cerr << "  Graphics descriptor sets: " << (graphicsSuccess ? "SUCCESS" : "FAILED") << std::endl;
                std::cerr << "  Compute descriptor sets: " << (computeSuccess ? "SUCCESS" : "FAILED") << std::endl;
            }
        } else {
            std::cerr << "RenderFrameDirector: WARNING - Invalid entity or position buffer during swapchain recreation" << std::endl;
            std::cerr << "  Movement params buffer: " << (movementParamsBuffer != VK_NULL_HANDLE ? "VALID" : "NULL") << std::endl;
            std::cerr << "  Position buffer: " << (positionBuffer != VK_NULL_HANDLE ? "VALID" : "NULL") << std::endl;
        }
    } else {
        std::cerr << "RenderFrameDirector: WARNING - Missing gpuEntityManager or resourceCoordinator during swapchain recreation" << std::endl;
    }
    
    // 5. That's it! Next frame will naturally import new images
    // No forced rebuilds, no stale references, no complexity
}

bool RenderFrameDirector::compileFrameGraph(uint32_t currentFrame, float totalTime, float deltaTime, uint32_t frameCounter) {
    // Compile frame graph only if not already compiled
    if (!frameGraph->isCompiled() && !frameGraph->compile()) {
        std::cerr << "RenderFrameDirector: Failed to compile frame graph" << std::endl;
        return false;
    }
    
    // Command buffer reset is now handled by QueueManager during frame execution
    // No manual reset needed here
    
    // Timing data is now passed directly to frame graph execution via prepareFrame
    
    return true;
}