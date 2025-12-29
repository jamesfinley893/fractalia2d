#pragma once

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <chrono>
#include "../core/vulkan_context.h"
#include "../core/vulkan_constants.h"
#include "../core/vulkan_manager_base.h"
#include "../core/vulkan_raii.h"
#include "graphics_pipeline_state_hash.h"
#include "graphics_pipeline_cache.h"
#include "graphics_render_pass_manager.h"
#include "graphics_pipeline_factory.h"
#include "graphics_pipeline_layout_builder.h"

class ShaderManager;
class DescriptorLayoutManager;

class GraphicsPipelineManager : public VulkanManagerBase {
public:
    explicit GraphicsPipelineManager(VulkanContext* ctx);
    ~GraphicsPipelineManager();

    bool initialize(ShaderManager* shaderManager,
                   DescriptorLayoutManager* layoutManager);
    void cleanup();
    void cleanupBeforeContextDestruction();

    VkPipeline getPipeline(const GraphicsPipelineState& state);
    VkPipelineLayout getPipelineLayout(const GraphicsPipelineState& state);
    
    std::vector<VkPipeline> createPipelinesBatch(const std::vector<GraphicsPipelineState>& states);
    
    VkRenderPass createRenderPass(VkFormat colorFormat, 
                                 VkFormat depthFormat = VK_FORMAT_UNDEFINED,
                                 VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT,
                                 bool enableMSAA = false);
    
    GraphicsPipelineState createDefaultState();
    GraphicsPipelineState createMSAAState();
    GraphicsPipelineState createWireframeState();
    GraphicsPipelineState createInstancedState();
    
    void warmupCache(const std::vector<GraphicsPipelineState>& commonStates);
    void optimizeCache(uint64_t currentFrame);
    void clearCache();
    
    bool recreatePipelineCache();
    
    DescriptorLayoutManager* getLayoutManager() { return layoutManager_; }
    const DescriptorLayoutManager* getLayoutManager() const { return layoutManager_; }
    
    PipelineStats getStats() const { return cache_.getStats(); }
    void resetFrameStats() { cache_.resetFrameStats(); }
    void debugPrintCache() const { cache_.debugPrintCache(); }
    
    bool reloadPipeline(const GraphicsPipelineState& state);
    void enableHotReload(bool enable) { hotReloadEnabled_ = enable; }

private:
    vulkan_raii::PipelineCache pipelineCache_;
    
    ShaderManager* shaderManager_ = nullptr;
    DescriptorLayoutManager* layoutManager_ = nullptr;
    
    bool hotReloadEnabled_ = false;
    bool isRecreating_ = false;  // Synchronization for cache recreation
    uint32_t maxCacheSize_ = DEFAULT_GRAPHICS_CACHE_SIZE;
    uint64_t cacheCleanupInterval_ = CACHE_CLEANUP_INTERVAL;
    
    GraphicsPipelineCache cache_;
    GraphicsRenderPassManager renderPassManager_;
    GraphicsPipelineFactory factory_;
    GraphicsPipelineLayoutBuilder layoutBuilder_;
};

namespace GraphicsPipelinePresets {
    GraphicsPipelineState createEntityRenderingState(VkRenderPass renderPass, 
                                                    VkDescriptorSetLayout descriptorLayout);
    
    GraphicsPipelineState createWireframeOverlayState(VkRenderPass renderPass);
    GraphicsPipelineState createUIRenderingState(VkRenderPass renderPass);
    GraphicsPipelineState createShadowMappingState(VkRenderPass renderPass);
}