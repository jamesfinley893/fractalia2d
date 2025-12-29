#pragma once

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <chrono>
#include <functional>
#include "../core/vulkan_constants.h"
#include <future>
#include <glm/glm.hpp>
#include "../core/vulkan_context.h"
#include "../core/vulkan_manager_base.h"
#include "../core/vulkan_raii.h"
#include "compute_pipeline_types.h"
#include "compute_pipeline_cache.h"
#include "compute_pipeline_factory.h"
#include "compute_dispatcher.h"
#include "compute_device_info.h"

class ShaderManager;
class DescriptorLayoutManager;

class ComputePipelineManager : public VulkanManagerBase {
public:
    explicit ComputePipelineManager(VulkanContext* ctx);
    ~ComputePipelineManager();

    bool initialize(ShaderManager* shaderManager,
                   DescriptorLayoutManager* layoutManager);
    void cleanup();
    void cleanupBeforeContextDestruction();

    VkPipeline getPipeline(const ComputePipelineState& state);
    VkPipelineLayout getPipelineLayout(const ComputePipelineState& state);
    
    std::vector<VkPipeline> createPipelinesBatch(const std::vector<ComputePipelineState>& states);
    
    void dispatch(VkCommandBuffer commandBuffer, const ComputeDispatch& dispatch);
    void dispatchIndirect(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset);
    
    void dispatchBuffer(VkCommandBuffer commandBuffer, const ComputePipelineState& state,
                       uint32_t elementCount, const std::vector<VkDescriptorSet>& descriptorSets,
                       const void* pushConstants = nullptr, uint32_t pushConstantSize = 0);
    
    void dispatchImage(VkCommandBuffer commandBuffer, const ComputePipelineState& state,
                      uint32_t width, uint32_t height, const std::vector<VkDescriptorSet>& descriptorSets,
                      const void* pushConstants = nullptr, uint32_t pushConstantSize = 0);
    
    // Default PSO presets for common compute patterns
    ComputePipelineState createBufferProcessingState(const std::string& shaderPath, 
                                                    VkDescriptorSetLayout descriptorLayout);
    ComputePipelineState createImageProcessingState(const std::string& shaderPath,
                                                   VkDescriptorSetLayout descriptorLayout);
    ComputePipelineState createParticleSystemState(const std::string& shaderPath,
                                                  VkDescriptorSetLayout descriptorLayout);
    
    // Cache management
    void warmupCache(const std::vector<ComputePipelineState>& commonStates);
    void optimizeCache(uint64_t currentFrame);
    void clearCache();
    
    // Pipeline cache recreation for swapchain resize operations
    bool recreatePipelineCache();
    
    // Async compilation for hot reloading
    bool compileAsync(const ComputePipelineState& state);
    bool isAsyncCompilationComplete(const ComputePipelineState& state);
    
    // Performance profiling
    struct ComputeProfileData {
        std::chrono::nanoseconds lastDispatchTime{0};
        uint64_t totalDispatches = 0;
        uint64_t totalWorkgroups = 0;
        float averageWorkgroupUtilization = 0.0f;
    };
    
    void beginProfiling(const ComputePipelineState& state);
    void endProfiling(const ComputePipelineState& state);
    ComputeProfileData getProfilingData(const ComputePipelineState& state) const;
    
    // Statistics and debugging
    struct ComputeStats {
        uint32_t totalPipelines = 0;
        uint32_t cacheHits = 0;
        uint32_t cacheMisses = 0;
        uint32_t dispatchesThisFrame = 0;
        uint64_t totalDispatches = 0;
        std::chrono::nanoseconds totalCompilationTime{0};
        float hitRatio = 0.0f;
    };
    
    ComputeStats getStats() const;
    void resetFrameStats();
    void debugPrintCache() const;
    
    // Workgroup optimization
    glm::uvec3 calculateOptimalWorkgroupSize(uint32_t dataSize, 
                                            const glm::uvec3& maxWorkgroupSize = {1024, 1024, 64}) const;
    uint32_t calculateOptimalWorkgroupCount(uint32_t dataSize, uint32_t workgroupSize) const;
    
    // Component access
    ComputePipelineCache* getCache() { return &cache_; }
    ComputePipelineFactory* getFactory() { return &factory_; }
    ComputeDispatcher* getDispatcher() { return &dispatcher_; }
    ComputeDeviceInfo* getDeviceInfo() { return &deviceInfo_; }
    
    // Memory barrier optimization
    void insertOptimalBarriers(VkCommandBuffer commandBuffer, 
                              const std::vector<VkBufferMemoryBarrier>& bufferBarriers,
                              const std::vector<VkImageMemoryBarrier>& imageBarriers,
                              VkPipelineStageFlags srcStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              VkPipelineStageFlags dstStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    
    // Access to layout manager for descriptor layout creation
    DescriptorLayoutManager* getLayoutManager() { return layoutManager_; }
    const DescriptorLayoutManager* getLayoutManager() const { return layoutManager_; }

private:
    // Core Vulkan objects
    vulkan_raii::PipelineCache pipelineCache_;
    
    // Dependencies
    ShaderManager* shaderManager_ = nullptr;
    DescriptorLayoutManager* layoutManager_ = nullptr;
    
    // State management
    bool isRecreating_ = false;  // Synchronization for cache recreation
    
    // Focused components
    ComputePipelineCache cache_;
    ComputePipelineFactory factory_;
    ComputeDispatcher dispatcher_;
    ComputeDeviceInfo deviceInfo_;
    
    // Async compilation tracking
    std::unordered_map<ComputePipelineState, std::future<std::unique_ptr<CachedComputePipeline>>, ComputePipelineStateHash> asyncCompilations;
    
    // Performance tracking
    std::unordered_map<ComputePipelineState, ComputeProfileData, ComputePipelineStateHash> profileData;
    
    // Configuration
    uint32_t maxCacheSize_ = DEFAULT_COMPUTE_CACHE_SIZE;
    uint64_t cacheCleanupInterval_ = CACHE_CLEANUP_INTERVAL;
    bool enableProfiling = false;
    
    // Internal pipeline creation callback for cache
    std::unique_ptr<CachedComputePipeline> createPipelineInternal(const ComputePipelineState& state);
    
    bool shouldEvictPipeline(const CachedComputePipeline& pipeline, uint64_t currentFrame) const;
};

// Utility functions for common compute patterns
namespace ComputePipelinePresets {
    // Entity movement computation (for your use case)
    ComputePipelineState createEntityMovementState(VkDescriptorSetLayout descriptorLayout);
    
    // Physics computation (velocity-based position updates)
    ComputePipelineState createPhysicsState(VkDescriptorSetLayout descriptorLayout);
    
    // Particle system update
    ComputePipelineState createParticleUpdateState(VkDescriptorSetLayout descriptorLayout);
    
    // Frustum culling
    ComputePipelineState createFrustumCullingState(VkDescriptorSetLayout descriptorLayout);
    
    // GPU sorting algorithms
    ComputePipelineState createRadixSortState(VkDescriptorSetLayout descriptorLayout);
    
    // Prefix sum (scan)
    ComputePipelineState createPrefixSumState(VkDescriptorSetLayout descriptorLayout);
}