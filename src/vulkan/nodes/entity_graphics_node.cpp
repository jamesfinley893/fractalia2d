#include "entity_graphics_node.h"
#include "../pipelines/graphics_pipeline_manager.h"
#include "../core/vulkan_swapchain.h"
#include "../resources/core/resource_coordinator.h"
#include "../resources/managers/graphics_resource_manager.h"
#include "../../ecs/gpu/gpu_entity_manager.h"
#include "../../ecs/gpu/soft_body_constants.h"
#include "../core/vulkan_context.h"
#include "../core/vulkan_function_loader.h"
#include "../../ecs/components/camera_component.h"
#include "../pipelines/descriptor_layout_manager.h"
#include <iostream>
#include "../../ecs/core/service_locator.h"
#include "../../ecs/services/camera_service.h"
#include <array>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <flecs.h>
#include <stdexcept>
#include <memory>

EntityGraphicsNode::EntityGraphicsNode(const Data& data)
    : data(data) {
    
    // Validate dependencies during construction for fail-fast behavior
    if (!this->data.graphicsManager) {
        throw std::invalid_argument("EntityGraphicsNode: graphicsManager cannot be null");
    }
    if (!this->data.swapchain) {
        throw std::invalid_argument("EntityGraphicsNode: swapchain cannot be null");
    }
    if (!this->data.resourceCoordinator) {
        throw std::invalid_argument("EntityGraphicsNode: resourceCoordinator cannot be null");
    }
    if (!this->data.gpuEntityManager) {
        throw std::invalid_argument("EntityGraphicsNode: gpuEntityManager cannot be null");
    }
}

std::vector<ResourceDependency> EntityGraphicsNode::getInputs() const {
    return {
        {data.positionBufferId, ResourceAccess::Read, PipelineStage::VertexShader},
        {data.movementParamsBufferId, ResourceAccess::Read, PipelineStage::VertexShader},
        {data.controlParamsBufferId, ResourceAccess::Read, PipelineStage::VertexShader},
    };
}

std::vector<ResourceDependency> EntityGraphicsNode::getOutputs() const {
    // ELEGANT SOLUTION: Use dynamic swapchain image ID resolved each frame
    return {
        {currentSwapchainImageId, ResourceAccess::Write, PipelineStage::ColorAttachment},
    };
}

bool EntityGraphicsNode::ensurePipeline() {
    if (!data.graphicsManager || !data.swapchain) {
        return false;
    }

    VkFormat currentFormat = data.swapchain->getImageFormat();
    if (currentFormat != cachedSwapchainFormat) {
        pipelineDirty = true;
    }

    auto layoutSpec = DescriptorLayoutPresets::createEntityGraphicsLayout();
    VkDescriptorSetLayout descriptorLayout = data.graphicsManager->getLayoutManager()->getLayout(layoutSpec);
    if (descriptorLayout == VK_NULL_HANDLE) {
        return false;
    }
    if (descriptorLayout != cachedDescriptorLayout) {
        pipelineDirty = true;
    }

    if (!pipelineDirty) {
        return cachedPipeline != VK_NULL_HANDLE && cachedPipelineLayout != VK_NULL_HANDLE && cachedRenderPass != VK_NULL_HANDLE;
    }

    VkRenderPass renderPass = data.graphicsManager->createRenderPass(
        currentFormat,
        VK_FORMAT_UNDEFINED,
        VK_SAMPLE_COUNT_2_BIT,
        true
    );
    if (renderPass == VK_NULL_HANDLE) {
        return false;
    }

    GraphicsPipelineState pipelineState = GraphicsPipelinePresets::createEntityRenderingState(
        renderPass, descriptorLayout);

    VkPipeline pipeline = data.graphicsManager->getPipeline(pipelineState);
    VkPipelineLayout pipelineLayout = data.graphicsManager->getPipelineLayout(pipelineState);
    if (pipeline == VK_NULL_HANDLE || pipelineLayout == VK_NULL_HANDLE) {
        return false;
    }

    cachedSwapchainFormat = currentFormat;
    cachedDescriptorLayout = descriptorLayout;
    cachedRenderPass = renderPass;
    cachedPipeline = pipeline;
    cachedPipelineLayout = pipelineLayout;
    pipelineDirty = false;
    return true;
}

void EntityGraphicsNode::execute(VkCommandBuffer commandBuffer, const FrameGraph& frameGraph) {
    
    // Validate dependencies are still valid
    if (!data.graphicsManager || !data.swapchain || !data.resourceCoordinator || !data.gpuEntityManager) {
        std::cerr << "EntityGraphicsNode: Critical error - dependencies became null during execution" << std::endl;
        return;
    }
    
    const uint32_t entityCount = data.gpuEntityManager->getEntityCount();
    
    if (entityCount == 0) {
        FRAME_GRAPH_DEBUG_LOG_THROTTLED(noEntitiesCounter, 1800, "EntityGraphicsNode: No entities to render");
        return;
    }
    
    // Get Vulkan context from frame graph
    const VulkanContext* context = frameGraph.getContext();
    if (!context) {
        std::cerr << "EntityGraphicsNode: Missing Vulkan context" << std::endl;
        return;
    }
    
    // Update uniform buffer with camera matrices (now handled by EntityDescriptorManager)
    updateUniformBuffer();
    
    if (!ensurePipeline()) {
        std::cerr << "EntityGraphicsNode: Failed to ensure graphics pipeline" << std::endl;
        return;
    }
    
    // Validate swapchain state before accessing framebuffers
    const auto& framebuffers = data.swapchain->getFramebuffers();
    if (imageIndex >= framebuffers.size()) {
        std::cerr << "EntityGraphicsNode: Invalid imageIndex " << imageIndex 
                  << " >= framebuffer count " << framebuffers.size() << std::endl;
        return;
    }
    
    // Begin render pass
    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = cachedRenderPass;
    renderPassInfo.framebuffer = framebuffers[imageIndex];
    renderPassInfo.renderArea.offset = {0, 0};
    renderPassInfo.renderArea.extent = data.swapchain->getExtent();

    // Clear values: MSAA color, resolve color (no depth)
    std::array<VkClearValue, 2> clearValues{};
    clearValues[0].color = {{0.1f, 0.1f, 0.2f, 1.0f}};  // MSAA color attachment
    clearValues[1].color = {{0.1f, 0.1f, 0.2f, 1.0f}};  // Resolve attachment
    renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
    renderPassInfo.pClearValues = clearValues.data();

    // Cache loader reference for performance
    const auto& vk = context->getLoader();

    vk.vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

    // Set dynamic viewport and scissor
    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(data.swapchain->getExtent().width);
    viewport.height = static_cast<float>(data.swapchain->getExtent().height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vk.vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = data.swapchain->getExtent();
    vk.vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

    // Entity count already retrieved above
    
    // Bind graphics pipeline
    vk.vkCmdBindPipeline(
        commandBuffer, 
        VK_PIPELINE_BIND_POINT_GRAPHICS, 
        cachedPipeline
    );
    
    // Bind single descriptor set with unified layout (uniform + storage buffers)
    VkDescriptorSet entityDescriptorSet = data.gpuEntityManager->getDescriptorManager().getGraphicsDescriptorSet(currentFrameIndex);
    
    if (entityDescriptorSet != VK_NULL_HANDLE) {
        vk.vkCmdBindDescriptorSets(
            commandBuffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            cachedPipelineLayout,
            0, 1, &entityDescriptorSet,
            0, nullptr
        );
    } else {
        std::cerr << "EntityGraphicsNode: ERROR - Missing graphics descriptor set for frame " << currentFrameIndex << "!" << std::endl;
        return;
    }

    // Push constants for vertex shader
    struct VertexPushConstants {
        float time;                 // Current simulation time
        float dt;                   // Time per frame  
        uint32_t count;             // Total number of entities
    } vertexPushConstants = { 
        frameTime,
        frameDeltaTime,
        entityCount
    };
    
    vk.vkCmdPushConstants(
        commandBuffer, 
        cachedPipelineLayout,
        VK_SHADER_STAGE_VERTEX_BIT, 
        0, sizeof(VertexPushConstants), 
        &vertexPushConstants
    );

    // Draw entities
    if (entityCount > 0) {
        constexpr uint32_t trianglesPerBody = SoftBodyConstants::kTrianglesPerBody;
        // Bind vertex buffer: only geometry vertices (SoA uses storage buffers for entity data)
        VkBuffer vertexBuffers[] = {
            data.resourceCoordinator->getGraphicsManager()->getVertexBuffer()      // Vertex positions for triangle geometry
        };
        VkDeviceSize offsets[] = {0};
        vk.vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
        
        // Bind index buffer for triangle geometry
        vk.vkCmdBindIndexBuffer(
            commandBuffer, data.resourceCoordinator->getGraphicsManager()->getIndexBuffer(), 0, VK_INDEX_TYPE_UINT16);
        
        // Draw indexed instances: all entities with triangle geometry
        vk.vkCmdDrawIndexed(
            commandBuffer, 
            data.resourceCoordinator->getGraphicsManager()->getIndexCount(),  // Number of indices per triangle
            entityCount * trianglesPerBody,   // Number of instances (triangles)
            0, 0, 0                          // Index/vertex/instance offsets
        );
        
        // Debug: confirm draw call (thread-safe)
        FRAME_GRAPH_DEBUG_LOG_THROTTLED(drawCounter, 1800, "EntityGraphicsNode: Drew " << entityCount << " entities with " << data.resourceCoordinator->getGraphicsManager()->getIndexCount() << " indices per triangle");
    }

    // End render pass
    vk.vkCmdEndRenderPass(commandBuffer);
}

void EntityGraphicsNode::updateUniformBuffer() {
    if (!data.resourceCoordinator) return;
    
    // Check if uniform buffer needs updating for this frame index
    bool needsUpdate = uniformBufferDirty || (lastUpdatedFrameIndex != currentFrameIndex);
    
    struct UniformBufferObject {
        glm::mat4 view;
        glm::mat4 proj;
    } newUBO{};

    // Get camera matrices from service
    auto& cameraService = ServiceLocator::instance().requireService<CameraService>();
    newUBO.view = cameraService.getViewMatrix();
    newUBO.proj = cameraService.getProjectionMatrix();
    
    // Debug camera matrix application (once every 30 seconds) - thread-safe
    if constexpr (FRAME_GRAPH_DEBUG_ENABLED) {
        uint32_t counter = FrameGraphDebug::incrementCounter(debugCounter);
        if (counter % 1800 == 0) {
            std::cout << "[FrameGraph Debug] EntityGraphicsNode: Using camera matrices from service (occurrence #" << counter << ")" << std::endl;
            std::cout << "  View matrix[3]: " << newUBO.view[3][0] << ", " << newUBO.view[3][1] << ", " << newUBO.view[3][2] << std::endl;
            std::cout << "  Proj matrix[0][0]: " << newUBO.proj[0][0] << ", [1][1]: " << newUBO.proj[1][1] << std::endl;
        }
    }
    
    // If no valid matrices, use fallback
    if (newUBO.view == glm::mat4(0.0f) || newUBO.proj == glm::mat4(0.0f)) {
        // Original fallback matrices when no world is set
        newUBO.view = glm::mat4(1.0f);
        newUBO.proj = glm::ortho(-4.0f, 4.0f, -3.0f, 3.0f, -5.0f, 5.0f);
        newUBO.proj[1][1] *= -1; // Flip Y for Vulkan
        
        FRAME_GRAPH_DEBUG_LOG_THROTTLED(debugCounter, 1800, "EntityGraphicsNode: Using fallback matrices - no world reference");
    }
    
    // Check if matrices actually changed (avoid memcmp by comparing key components)
    bool matricesChanged = (newUBO.view != cachedUBO.view) || (newUBO.proj != cachedUBO.proj);
    
    // Only update if dirty, frame changed, or matrices changed
    if (needsUpdate || matricesChanged) {
        auto uniformBuffers = data.resourceCoordinator->getGraphicsManager()->getUniformBuffersMapped();
        
        // Auto-recreate uniform buffers if they were destroyed (e.g., during resize)
        if (uniformBuffers.empty()) {
            std::cout << "EntityGraphicsNode: Uniform buffers missing, attempting to recreate..." << std::endl;
            if (data.resourceCoordinator->getGraphicsManager()->createAllGraphicsResources()) {
                std::cout << "EntityGraphicsNode: Successfully recreated graphics resources" << std::endl;
                uniformBuffers = data.resourceCoordinator->getGraphicsManager()->getUniformBuffersMapped();
            } else {
                std::cerr << "EntityGraphicsNode: CRITICAL ERROR: Failed to recreate graphics resources!" << std::endl;
                return;
            }
        }
        
        if (!uniformBuffers.empty() && currentFrameIndex < uniformBuffers.size()) {
            void* data = uniformBuffers[currentFrameIndex];
            if (data) {
                memcpy(data, &newUBO, sizeof(newUBO));
                
                // Update cache and tracking
                cachedUBO.view = newUBO.view;
                cachedUBO.proj = newUBO.proj;
                uniformBufferDirty = false;
                lastUpdatedFrameIndex = currentFrameIndex;
                
                // Debug optimized updates (once every 30 seconds) - thread-safe
                FRAME_GRAPH_DEBUG_LOG_THROTTLED(updateCounter, 1800, "EntityGraphicsNode: Updated uniform buffer (optimized)");
            }
        } else {
            std::cerr << "EntityGraphicsNode: ERROR: invalid currentFrameIndex (" << currentFrameIndex 
                     << ") or uniformBuffers size (" << uniformBuffers.size() << ")!" << std::endl;
        }
    }
}

// Node lifecycle implementation
bool EntityGraphicsNode::initializeNode(const FrameGraph& frameGraph) {
    // One-time initialization - validate dependencies
    if (!data.graphicsManager) {
        std::cerr << "EntityGraphicsNode: GraphicsPipelineManager is null" << std::endl;
        return false;
    }
    if (!data.swapchain) {
        std::cerr << "EntityGraphicsNode: VulkanSwapchain is null" << std::endl;
        return false;
    }
    if (!data.resourceCoordinator) {
        std::cerr << "EntityGraphicsNode: ResourceCoordinator is null" << std::endl;
        return false;
    }
    if (!data.gpuEntityManager) {
        std::cerr << "EntityGraphicsNode: GPUEntityManager is null" << std::endl;
        return false;
    }
    return ensurePipeline();
}

void EntityGraphicsNode::prepareFrame(const FrameContext& frameContext) {
    // Store timing data
    frameTime = frameContext.time;
    frameDeltaTime = frameContext.deltaTime;
    currentFrameIndex = frameContext.frameIndex;
    
    // Check if uniform buffer needs updating
    if (uniformBufferDirty || lastUpdatedFrameIndex != frameContext.frameIndex) {
        updateUniformBuffer();
    }
}

void EntityGraphicsNode::releaseFrame(uint32_t frameIndex) {
    // Per-frame cleanup - nothing to clean up for graphics node
}
