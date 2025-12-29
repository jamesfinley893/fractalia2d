#include "graphics_pipeline_manager.h"
#include "shader_manager.h"
#include "descriptor_layout_manager.h"
#include "../core/vulkan_constants.h"
#include <iostream>
#include <glm/glm.hpp>

GraphicsPipelineManager::GraphicsPipelineManager(VulkanContext* ctx) 
    : VulkanManagerBase(ctx)
    , maxCacheSize_(DEFAULT_GRAPHICS_CACHE_SIZE)
    , cache_(maxCacheSize_)
    , renderPassManager_(ctx)
    , factory_(ctx)
    , layoutBuilder_(ctx) {
}

GraphicsPipelineManager::~GraphicsPipelineManager() {
    cleanupBeforeContextDestruction();
}

bool GraphicsPipelineManager::initialize(ShaderManager* shaderManager,
                                       DescriptorLayoutManager* layoutManager) {
    shaderManager_ = shaderManager;
    layoutManager_ = layoutManager;
    
    VkPipelineCacheCreateInfo cacheInfo{};
    cacheInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    cacheInfo.initialDataSize = 0;
    cacheInfo.pInitialData = nullptr;
    
    pipelineCache_ = vulkan_raii::create_pipeline_cache(context, &cacheInfo);
    if (!pipelineCache_) {
        std::cerr << "Failed to create graphics pipeline cache" << std::endl;
        return false;
    }
    
    if (!factory_.initialize(shaderManager, &pipelineCache_)) {
        std::cerr << "Failed to initialize graphics pipeline factory" << std::endl;
        return false;
    }
    
    std::cout << "GraphicsPipelineManager initialized successfully" << std::endl;
    return true;
}

void GraphicsPipelineManager::cleanup() {
    cleanupBeforeContextDestruction();
}

void GraphicsPipelineManager::cleanupBeforeContextDestruction() {
    if (!context) return;
    
    clearCache();
    renderPassManager_.clearCache();
    pipelineCache_.reset();
    
    context = nullptr;
}

VkPipeline GraphicsPipelineManager::getPipeline(const GraphicsPipelineState& state) {
    VkPipeline cachedPipeline = cache_.getPipeline(state);
    if (cachedPipeline != VK_NULL_HANDLE) {
        return cachedPipeline;
    }
    
    auto newPipeline = factory_.createPipeline(state);
    if (!newPipeline) {
        std::cerr << "Failed to create graphics pipeline" << std::endl;
        return VK_NULL_HANDLE;
    }
    
    VkPipeline pipeline = newPipeline->pipeline.get();
    cache_.storePipeline(state, std::move(newPipeline));
    
    return pipeline;
}

VkPipelineLayout GraphicsPipelineManager::getPipelineLayout(const GraphicsPipelineState& state) {
    VkPipelineLayout cachedLayout = cache_.getPipelineLayout(state);
    if (cachedLayout != VK_NULL_HANDLE) {
        return cachedLayout;
    }
    
    VkPipeline pipeline = getPipeline(state);
    if (pipeline == VK_NULL_HANDLE) {
        return VK_NULL_HANDLE;
    }
    
    return cache_.getPipelineLayout(state);
}

std::vector<VkPipeline> GraphicsPipelineManager::createPipelinesBatch(const std::vector<GraphicsPipelineState>& states) {
    std::vector<VkPipeline> pipelines;
    pipelines.reserve(states.size());
    
    for (const auto& state : states) {
        VkPipeline pipeline = getPipeline(state);
        pipelines.push_back(pipeline);
    }
    
    return pipelines;
}

VkRenderPass GraphicsPipelineManager::createRenderPass(VkFormat colorFormat, 
                                                      VkFormat depthFormat,
                                                      VkSampleCountFlagBits samples,
                                                      bool enableMSAA) {
    return renderPassManager_.createRenderPass(colorFormat, depthFormat, samples, enableMSAA);
}

GraphicsPipelineState GraphicsPipelineManager::createDefaultState() {
    GraphicsPipelineState state{};
    
    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | 
                                         VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;
    state.colorBlendAttachments.push_back(colorBlendAttachment);
    
    return state;
}

GraphicsPipelineState GraphicsPipelineManager::createMSAAState() {
    GraphicsPipelineState state{};
    state.rasterizationSamples = VK_SAMPLE_COUNT_2_BIT;
    state.sampleShadingEnable = VK_FALSE;
    state.minSampleShading = 1.0f;
    
    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | 
                                         VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;
    state.colorBlendAttachments.push_back(colorBlendAttachment);
    
    return state;
}

GraphicsPipelineState GraphicsPipelineManager::createWireframeState() {
    GraphicsPipelineState state = createDefaultState();
    state.polygonMode = VK_POLYGON_MODE_LINE;
    state.lineWidth = 1.0f;
    return state;
}

GraphicsPipelineState GraphicsPipelineManager::createInstancedState() {
    return createDefaultState();
}

void GraphicsPipelineManager::warmupCache(const std::vector<GraphicsPipelineState>& commonStates) {
    for (const auto& state : commonStates) {
        getPipeline(state);
    }
}

void GraphicsPipelineManager::optimizeCache(uint64_t currentFrame) {
    cache_.optimizeCache(currentFrame);
}

void GraphicsPipelineManager::clearCache() {
    cache_.clear();
    renderPassManager_.clearCache();
}

bool GraphicsPipelineManager::recreatePipelineCache() {
    if (!context) {
        std::cerr << "GraphicsPipelineManager: Cannot recreate pipeline cache - no context" << std::endl;
        return false;
    }
    
    if (isRecreating_) {
        std::cerr << "GraphicsPipelineManager: Recreation already in progress, ignoring request" << std::endl;
        return true; // Consider it successful to avoid error cascades
    }
    
    isRecreating_ = true;
    std::cout << "GraphicsPipelineManager: Recreating pipeline cache to prevent corruption" << std::endl;
    
    // Wait for device idle to ensure no pipelines are in use
    const auto& vk = context->getLoader();
    const VkDevice device = context->getDevice();
    vk.vkDeviceWaitIdle(device);
    
    // Clear caches in dependency order
    clearCache();
    
    if (layoutManager_) {
        std::cout << "GraphicsPipelineManager: Also clearing descriptor layout cache" << std::endl;
        layoutManager_->clearCache();
    }
    
    // Reset pipeline cache
    if (pipelineCache_) {
        std::cout << "GraphicsPipelineManager: Destroying pipeline cache" << std::endl;
        pipelineCache_.reset();
    }
    
    // Create new pipeline cache
    VkPipelineCacheCreateInfo cacheInfo{};
    cacheInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    cacheInfo.initialDataSize = 0;
    cacheInfo.pInitialData = nullptr;
    
    pipelineCache_ = vulkan_raii::create_pipeline_cache(context, &cacheInfo);
    if (!pipelineCache_) {
        std::cerr << "GraphicsPipelineManager: Failed to recreate pipeline cache" << std::endl;
        isRecreating_ = false;
        return false;
    }
    
    isRecreating_ = false;
    std::cout << "GraphicsPipelineManager: Pipeline cache successfully recreated" << std::endl;
    return true;
}

bool GraphicsPipelineManager::reloadPipeline(const GraphicsPipelineState& state) {
    if (!hotReloadEnabled_) {
        return false;
    }
    
    if (cache_.contains(state)) {
        auto newPipeline = factory_.createPipeline(state);
        if (newPipeline) {
            cache_.storePipeline(state, std::move(newPipeline));
            return true;
        }
    }
    
    return false;
}

namespace GraphicsPipelinePresets {
    GraphicsPipelineState createEntityRenderingState(VkRenderPass renderPass, 
                                                    VkDescriptorSetLayout descriptorLayout) {
        GraphicsPipelineState state{};
        state.renderPass = renderPass;
        state.descriptorSetLayouts.push_back(descriptorLayout);
        
        state.shaderStages = {
            "shaders/vertex.vert.spv",
            "shaders/fragment.frag.spv"
        };
        
        VkVertexInputBindingDescription vertexBinding{};
        vertexBinding.binding = 0;
        vertexBinding.stride = sizeof(glm::vec3) + sizeof(glm::vec3);
        vertexBinding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        state.vertexBindings.push_back(vertexBinding);
        
        // SoA approach: No instance binding for entity data (using storage buffers instead)
        // Only keep vertex position attribute (geometry data)
        
        VkVertexInputAttributeDescription posAttr{};
        posAttr.binding = 0;
        posAttr.location = 0;
        posAttr.format = VK_FORMAT_R32G32B32_SFLOAT;
        posAttr.offset = 0;
        state.vertexAttributes.push_back(posAttr);
        
        VkVertexInputAttributeDescription colorAttr{};
        colorAttr.binding = 0;
        colorAttr.location = 1;
        colorAttr.format = VK_FORMAT_R32G32B32_SFLOAT;
        colorAttr.offset = sizeof(glm::vec3);
        state.vertexAttributes.push_back(colorAttr);
        
        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | 
                                             VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_FALSE;
        state.colorBlendAttachments.push_back(colorBlendAttachment);
        
        return state;
    }
}