#include "compute_pipeline_manager.h"
#include "shader_manager.h"
#include "descriptor_layout_manager.h"
#include "../core/vulkan_function_loader.h"
#include "../core/vulkan_utils.h"
#include "../core/vulkan_constants.h"
#include <iostream>
#include <chrono>
#include <algorithm>
#include <cassert>
#include <future>
#include <cmath>
#include <glm/glm.hpp>


// ComputePipelineManager implementation
ComputePipelineManager::ComputePipelineManager(VulkanContext* ctx) 
    : VulkanManagerBase(ctx), cache_(DEFAULT_COMPUTE_CACHE_SIZE), factory_(ctx), dispatcher_(ctx), deviceInfo_(ctx) {
}

ComputePipelineManager::~ComputePipelineManager() {
    cleanupBeforeContextDestruction();
}

bool ComputePipelineManager::initialize(ShaderManager* shaderManager,
                                      DescriptorLayoutManager* layoutManager) {
    this->shaderManager_ = shaderManager;
    this->layoutManager_ = layoutManager;
    
    // Create pipeline cache for optimal performance
    VkPipelineCacheCreateInfo cacheInfo{};
    cacheInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    cacheInfo.initialDataSize = 0;
    cacheInfo.pInitialData = nullptr;
    
    pipelineCache_ = vulkan_raii::create_pipeline_cache(context, &cacheInfo);
    if (!pipelineCache_) {
        std::cerr << "Failed to create compute pipeline cache" << std::endl;
        return false;
    }
    
    // Initialize factory with pipeline cache
    if (!factory_.initialize(shaderManager_, pipelineCache_)) {
        std::cerr << "Failed to initialize compute pipeline factory" << std::endl;
        return false;
    }
    
    // Set up cache callback to create pipelines
    cache_.setCreatePipelineCallback([this](const ComputePipelineState& state) {
        return createPipelineInternal(state);
    });
    
    // Device properties are now handled by ComputeDeviceInfo component
    
    std::cout << "ComputePipelineManager initialized successfully" << std::endl;
    return true;
}

void ComputePipelineManager::cleanup() {
    cleanupBeforeContextDestruction();
}

void ComputePipelineManager::cleanupBeforeContextDestruction() {
    if (!context) return;
    
    // Wait for any async compilations to complete
    for (auto& [state, future] : asyncCompilations) {
        if (future.valid()) {
            future.wait();
        }
    }
    asyncCompilations.clear();
    
    // Clear pipeline cache (RAII handles cleanup automatically)
    clearCache();
    
    // Reset pipeline cache (RAII handles cleanup automatically)
    pipelineCache_.reset();
    
    context = nullptr;
}

VkPipeline ComputePipelineManager::getPipeline(const ComputePipelineState& state) {
    // Check async compilation first
    auto asyncIt = asyncCompilations.find(state);
    if (asyncIt != asyncCompilations.end() && asyncIt->second.valid()) {
        if (asyncIt->second.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
            auto cachedPipeline = asyncIt->second.get();
            asyncCompilations.erase(asyncIt);
            
            if (cachedPipeline) {
                VkPipeline pipeline = cachedPipeline->pipeline.get();
                cache_.insert(state, std::move(cachedPipeline));
                return pipeline;
            }
        }
    }
    
    return cache_.getPipeline(state);
}

VkPipelineLayout ComputePipelineManager::getPipelineLayout(const ComputePipelineState& state) {
    return cache_.getPipelineLayout(state);
}

void ComputePipelineManager::dispatch(VkCommandBuffer commandBuffer, const ComputeDispatch& dispatch) {
    dispatcher_.dispatch(commandBuffer, dispatch);
}

void ComputePipelineManager::dispatchBuffer(VkCommandBuffer commandBuffer, 
                                          const ComputePipelineState& state,
                                          uint32_t elementCount, 
                                          const std::vector<VkDescriptorSet>& descriptorSets,
                                          const void* pushConstants, 
                                          uint32_t pushConstantSize) {
    VkPipeline pipeline = getPipeline(state);
    VkPipelineLayout layout = getPipelineLayout(state);
    
    if (pipeline == VK_NULL_HANDLE || layout == VK_NULL_HANDLE) {
        std::cerr << "Failed to get compute pipeline for buffer dispatch" << std::endl;
        return;
    }
    
    // Create optimized dispatch
    ComputeDispatch dispatch{};
    dispatch.pipeline = pipeline;
    dispatch.layout = layout;
    dispatch.descriptorSets = descriptorSets;
    dispatch.pushConstantData = pushConstants;
    dispatch.pushConstantSize = pushConstantSize;
    
    // Calculate optimal workgroup configuration
    glm::uvec3 workgroupSize(state.workgroupSizeX, state.workgroupSizeY, state.workgroupSizeZ);
    dispatch.calculateOptimalDispatch(elementCount, workgroupSize);
    
    this->dispatch(commandBuffer, dispatch);
}

void ComputePipelineManager::dispatchImage(VkCommandBuffer commandBuffer,
                                         const ComputePipelineState& state,
                                         uint32_t width, uint32_t height,
                                         const std::vector<VkDescriptorSet>& descriptorSets,
                                         const void* pushConstants,
                                         uint32_t pushConstantSize) {
    VkPipeline pipeline = getPipeline(state);
    VkPipelineLayout layout = getPipelineLayout(state);
    
    if (pipeline == VK_NULL_HANDLE || layout == VK_NULL_HANDLE) {
        std::cerr << "Failed to get compute pipeline for image dispatch" << std::endl;
        return;
    }
    
    // Create optimized dispatch for 2D image processing
    ComputeDispatch dispatch{};
    dispatch.pipeline = pipeline;
    dispatch.layout = layout;
    dispatch.descriptorSets = descriptorSets;
    dispatch.pushConstantData = pushConstants;
    dispatch.pushConstantSize = pushConstantSize;
    
    // Calculate 2D dispatch
    dispatch.groupCountX = (width + state.workgroupSizeX - 1) / state.workgroupSizeX;
    dispatch.groupCountY = (height + state.workgroupSizeY - 1) / state.workgroupSizeY;
    dispatch.groupCountZ = 1;
    
    this->dispatch(commandBuffer, dispatch);
}

std::unique_ptr<CachedComputePipeline> ComputePipelineManager::createPipelineInternal(const ComputePipelineState& state) {
    auto cachedPipeline = factory_.createPipeline(state);
    if (cachedPipeline) {
        // Set up dispatch optimization info using device info
        cachedPipeline->dispatchInfo.optimalWorkgroupSize = deviceInfo_.getOptimalWorkgroupSize();
        cachedPipeline->dispatchInfo.maxInvocationsPerWorkgroup = deviceInfo_.getMaxComputeWorkgroupInvocations();
        cachedPipeline->dispatchInfo.supportsSubgroupOperations = deviceInfo_.supportsSubgroupOperations();
    }
    return cachedPipeline;
}


void ComputePipelineManager::clearCache() {
    if (!context) return;
    cache_.clear();
}

bool ComputePipelineManager::recreatePipelineCache() {
    if (!context) {
        std::cerr << "ComputePipelineManager: Cannot recreate pipeline cache - no context" << std::endl;
        return false;
    }
    
    if (isRecreating_) {
        std::cerr << "ComputePipelineManager: Recreation already in progress, ignoring request" << std::endl;
        return true; // Consider it successful to avoid error cascades
    }
    
    isRecreating_ = true;
    std::cout << "ComputePipelineManager: Recreating pipeline cache for swapchain resize" << std::endl;
    
    // Wait for device idle to ensure no pipelines are in use
    const auto& vk = context->getLoader();
    const VkDevice device = context->getDevice();
    vk.vkDeviceWaitIdle(device);
    
    // Clear async compilations to prevent race conditions
    asyncCompilations.clear();
    
    // Clear caches in dependency order
    clearCache();
    
    if (layoutManager_) {
        std::cout << "ComputePipelineManager: Also clearing descriptor layout cache to prevent stale handles" << std::endl;
        layoutManager_->clearCache();
    }
    
    // Reset pipeline cache
    if (pipelineCache_) {
        std::cout << "ComputePipelineManager: Destroying corrupted pipeline cache" << std::endl;
        pipelineCache_.reset();
    }
    
    // Create new pipeline cache
    VkPipelineCacheCreateInfo cacheInfo{};
    cacheInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    cacheInfo.initialDataSize = 0;
    cacheInfo.pInitialData = nullptr;
    
    pipelineCache_ = vulkan_raii::create_pipeline_cache(context, &cacheInfo);
    if (!pipelineCache_) {
        std::cerr << "ComputePipelineManager: Failed to recreate pipeline cache" << std::endl;
        return false;
    }
    
    // Reinitialize factory with new cache
    factory_.initialize(shaderManager_, pipelineCache_);
    
    isRecreating_ = false;
    std::cout << "ComputePipelineManager: Pipeline cache successfully recreated" << std::endl;
    return true;
}


glm::uvec3 ComputePipelineManager::calculateOptimalWorkgroupSize(uint32_t dataSize,
                                                                const glm::uvec3& maxWorkgroupSize) const {
    return deviceInfo_.calculateOptimalWorkgroupSize(dataSize, maxWorkgroupSize);
}

uint32_t ComputePipelineManager::calculateOptimalWorkgroupCount(uint32_t dataSize, uint32_t workgroupSize) const {
    return deviceInfo_.calculateOptimalWorkgroupCount(dataSize, workgroupSize);
}

void ComputePipelineManager::debugPrintCache() const {
    std::cout << "ComputePipelineManager Cache Stats:" << std::endl;
    auto stats = getStats();
    std::cout << "  Total pipelines: " << stats.totalPipelines << std::endl;
    std::cout << "  Cache hits: " << stats.cacheHits << std::endl;
    std::cout << "  Cache misses: " << stats.cacheMisses << std::endl;
    std::cout << "  Hit ratio: " << stats.hitRatio << std::endl;
}

void ComputePipelineManager::insertOptimalBarriers(VkCommandBuffer commandBuffer,
                                                   const std::vector<VkBufferMemoryBarrier>& bufferBarriers,
                                                   const std::vector<VkImageMemoryBarrier>& imageBarriers,
                                                   VkPipelineStageFlags srcStage,
                                                   VkPipelineStageFlags dstStage) {
    dispatcher_.insertOptimalBarriers(commandBuffer, bufferBarriers, imageBarriers, srcStage, dstStage);
}



void ComputePipelineManager::resetFrameStats() {
    cache_.resetFrameStats();
    dispatcher_.resetFrameStats();
}



ComputePipelineState ComputePipelineManager::createBufferProcessingState(const std::string& shaderPath,
                                                                       VkDescriptorSetLayout descriptorLayout) {
    ComputePipelineState state{};
    state.shaderPath = shaderPath;
    state.descriptorSetLayouts.push_back(descriptorLayout);
    state.workgroupSizeX = THREADS_PER_WORKGROUP;
    state.workgroupSizeY = 1;
    state.workgroupSizeZ = 1;
    state.isFrequentlyUsed = true;
    
    return state;
}

// ComputePipelinePresets namespace implementation
namespace ComputePipelinePresets {
    ComputePipelineState createEntityMovementState(VkDescriptorSetLayout descriptorLayout) {
        ComputePipelineState state{};
        state.shaderPath = "shaders/movement_random.comp.spv";
        state.descriptorSetLayouts.push_back(descriptorLayout);
        state.workgroupSizeX = THREADS_PER_WORKGROUP;  // MUST match shader local_size_x
        state.workgroupSizeY = 1;
        state.workgroupSizeZ = 1;
        state.isFrequentlyUsed = true;
        
        // Add push constants for time/frame data (must match ComputePushConstants struct)
        VkPushConstantRange pushConstant{};
        pushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushConstant.offset = 0;
        pushConstant.size = sizeof(float) * 2 + sizeof(uint32_t) * 6;  // time, deltaTime, entityCount, frame, entityOffset, padding[3]
        state.pushConstantRanges.push_back(pushConstant);
        
        return state;
    }
    
    ComputePipelineState createPhysicsState(VkDescriptorSetLayout descriptorLayout) {
        ComputePipelineState state{};
        state.shaderPath = "shaders/physics.comp.spv";
        state.descriptorSetLayouts.push_back(descriptorLayout);
        state.workgroupSizeX = THREADS_PER_WORKGROUP;  // MUST match shader local_size_x
        state.workgroupSizeY = 1;
        state.workgroupSizeZ = 1;
        state.isFrequentlyUsed = true;
        
        // Add push constants for time/frame data (must match PhysicsPushConstants struct)
        VkPushConstantRange pushConstant{};
        pushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushConstant.offset = 0;
        pushConstant.size = sizeof(float) * 2 + sizeof(uint32_t) * 6;  // time, deltaTime, entityCount, frame, entityOffset, padding[3]
        state.pushConstantRanges.push_back(pushConstant);
        
        return state;
    }
}

void ComputePipelineManager::optimizeCache(uint64_t currentFrame) {
    cache_.optimizeCache(currentFrame);
}

void ComputePipelineManager::warmupCache(const std::vector<ComputePipelineState>& commonStates) {
    for (const auto& state : commonStates) {
        getPipeline(state);
    }
}

ComputePipelineManager::ComputeStats ComputePipelineManager::getStats() const {
    auto cacheStats = cache_.getStats();
    auto dispatchStats = dispatcher_.getStats();
    
    ComputeStats stats{};
    stats.totalPipelines = cacheStats.totalPipelines;
    stats.cacheHits = cacheStats.cacheHits;
    stats.cacheMisses = cacheStats.cacheMisses;
    stats.dispatchesThisFrame = dispatchStats.dispatchesThisFrame;
    stats.totalDispatches = dispatchStats.totalDispatches;
    stats.totalCompilationTime = cacheStats.totalCompilationTime;
    stats.hitRatio = cacheStats.hitRatio;
    
    return stats;
}