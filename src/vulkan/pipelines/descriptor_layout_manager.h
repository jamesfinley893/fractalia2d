#pragma once

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <functional>
#include "../core/vulkan_context.h"
#include "../core/vulkan_raii.h"
#include "../core/vulkan_constants.h"

// Descriptor binding specification for flexible layout creation
struct DescriptorBinding {
    uint32_t binding = 0;
    VkDescriptorType type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uint32_t descriptorCount = 1;
    VkShaderStageFlags stageFlags = VK_SHADER_STAGE_ALL;
    
    // Bindless support
    bool isBindless = false;
    uint32_t maxBindlessDescriptors = 0;
    
    // Immutable samplers (for texture arrays)
    std::vector<VkSampler> immutableSamplers;
    
    // Debug name for easier debugging
    std::string debugName;
    
    // Comparison for caching
    bool operator==(const DescriptorBinding& other) const;
    size_t getHash() const;
};

// Descriptor set layout specification
struct DescriptorLayoutSpec {
    std::vector<DescriptorBinding> bindings;
    VkDescriptorSetLayoutCreateFlags flags = 0;
    
    // Bindless configuration
    bool enableBindless = false;
    bool enableUpdateAfterBind = false;
    bool enablePartiallyBound = false;
    
    // Debug information
    std::string layoutName;
    
    // Comparison for caching
    bool operator==(const DescriptorLayoutSpec& other) const;
    size_t getHash() const;
};

// Hash specialization for layout caching
struct DescriptorLayoutSpecHash {
    std::size_t operator()(const DescriptorLayoutSpec& spec) const {
        return spec.getHash();
    }
};

// Cached descriptor set layout with metadata
struct CachedDescriptorLayout {
    vulkan_raii::DescriptorSetLayout layout;
    DescriptorLayoutSpec spec;
    
    // Usage tracking
    uint64_t lastUsedFrame = 0;
    uint32_t useCount = 0;
    
    // Pool sizing hints based on usage
    struct PoolSizeHints {
        uint32_t uniformBuffers = 0;
        uint32_t storageBuffers = 0;
        uint32_t sampledImages = 0;
        uint32_t storageImages = 0;
        uint32_t samplers = 0;
        uint32_t combinedImageSamplers = 0;
        uint32_t inputAttachments = 0;
    } poolSizeHints;
    
    // Bindless information
    bool isBindless = false;
    uint32_t maxBindlessDescriptors = 0;
};

// Descriptor pool configuration with automatic sizing
struct DescriptorPoolConfig {
    uint32_t maxSets = DEFAULT_MAX_DESCRIPTOR_SETS;
    
    // Pool size multipliers (will be calculated based on layout usage)
    float uniformBufferMultiplier = 2.0f;
    float storageBufferMultiplier = 2.0f;
    float sampledImageMultiplier = 4.0f;
    float storageImageMultiplier = 2.0f;
    float samplerMultiplier = 2.0f;
    float combinedImageSamplerMultiplier = 4.0f;
    
    // Bindless support
    bool enableBindless = false;
    uint32_t maxBindlessDescriptors = 65536; // Keep as is - specific to hardware limit
    
    // Pool management
    bool allowFreeDescriptorSets = true;
    bool enableUpdateAfterBind = false;
    
    std::string debugName;
};

// AAA Descriptor Layout Manager with advanced features
class DescriptorLayoutManager {
public:
    DescriptorLayoutManager();
    ~DescriptorLayoutManager();

    // Initialization
    bool initialize(const VulkanContext& context);
    void cleanup();
    void cleanupBeforeContextDestruction();

    // Layout creation and caching
    VkDescriptorSetLayout getLayout(const DescriptorLayoutSpec& spec);
    VkDescriptorSetLayout createLayout(const DescriptorLayoutSpec& spec);
    
    // Batch layout creation for reduced driver overhead
    std::vector<VkDescriptorSetLayout> createLayoutsBatch(const std::vector<DescriptorLayoutSpec>& specs);
    
    // Descriptor pool management
    VkDescriptorPool createOptimalPool(const DescriptorPoolConfig& config);
    VkDescriptorPool createPoolForLayouts(const std::vector<VkDescriptorSetLayout>& layouts,
                                         uint32_t maxSets = DEFAULT_MAX_DESCRIPTOR_SETS);
    
    // Common layout presets
    VkDescriptorSetLayout getUniformBufferLayout(VkShaderStageFlags stages = VK_SHADER_STAGE_ALL);
    VkDescriptorSetLayout getStorageBufferLayout(VkShaderStageFlags stages = VK_SHADER_STAGE_COMPUTE_BIT);
    VkDescriptorSetLayout getCombinedImageSamplerLayout(uint32_t textureCount = 1,
                                                       VkShaderStageFlags stages = VK_SHADER_STAGE_FRAGMENT_BIT);
    VkDescriptorSetLayout getStorageImageLayout(VkShaderStageFlags stages = VK_SHADER_STAGE_COMPUTE_BIT);
    
    // Bindless layouts
    VkDescriptorSetLayout getBindlessTextureLayout(uint32_t maxTextures = MAX_BINDLESS_TEXTURES);
    VkDescriptorSetLayout getBindlessBufferLayout(uint32_t maxBuffers = MAX_BINDLESS_BUFFERS);
    
    // Layout analysis and optimization
    struct LayoutAnalysis {
        uint32_t totalBindings = 0;
        uint32_t bindlessBindings = 0;
        bool requiresUpdateAfterBind = false;
        bool requiresPartiallyBound = false;
        std::vector<VkDescriptorType> descriptorTypes;
        VkShaderStageFlags stageFlags = 0;
    };
    
    LayoutAnalysis analyzeLayout(VkDescriptorSetLayout layout) const;
    LayoutAnalysis analyzeLayoutSpec(const DescriptorLayoutSpec& spec) const;
    
    // Cache management
    void warmupCache(const std::vector<DescriptorLayoutSpec>& commonLayouts);
    void optimizeCache(uint64_t currentFrame);
    void clearCache();
    
    // Statistics and debugging
    struct LayoutStats {
        uint32_t totalLayouts = 0;
        uint32_t cacheHits = 0;
        uint32_t cacheMisses = 0;
        uint32_t bindlessLayouts = 0;
        uint32_t activePools = 0;
        float hitRatio = 0.0f;
    };
    
    LayoutStats getStats() const { return stats; }
    void resetFrameStats();
    void debugPrintCache() const;
    void debugPrintLayout(VkDescriptorSetLayout layout) const;
    
    // Device capability queries
    bool supportsBindless() const;
    bool supportsUpdateAfterBind() const;
    uint32_t getMaxBindlessDescriptors() const;
    uint32_t getMaxDescriptorSetSamplers() const;
    
    // Validation helpers
    bool validateLayoutSpec(const DescriptorLayoutSpec& spec) const;
    bool isLayoutCompatible(VkDescriptorSetLayout layout1, VkDescriptorSetLayout layout2) const;

private:
    // Core Vulkan objects
    const VulkanContext* context_ = nullptr;
    
    // Layout cache
    std::unordered_map<DescriptorLayoutSpec, std::unique_ptr<CachedDescriptorLayout>, DescriptorLayoutSpecHash> layoutCache_;
    
    // Pool management
    std::vector<vulkan_raii::DescriptorPool> managedPools_;
    
    // Device features and limits
    VkPhysicalDeviceDescriptorIndexingFeatures indexingFeatures_{};
    VkPhysicalDeviceDescriptorIndexingProperties indexingProperties_{};
    VkPhysicalDeviceProperties deviceProperties_{};
    
    // Common layout cache (for frequently used patterns)
    struct CommonLayouts {
        vulkan_raii::DescriptorSetLayout uniformBuffer;
        vulkan_raii::DescriptorSetLayout storageBuffer;
        vulkan_raii::DescriptorSetLayout combinedImageSampler;
        vulkan_raii::DescriptorSetLayout storageImage;
        vulkan_raii::DescriptorSetLayout bindlessTextures;
        vulkan_raii::DescriptorSetLayout bindlessBuffers;
    } commonLayouts;
    
    // Statistics
    mutable LayoutStats stats;
    
    // Configuration
    uint32_t maxCacheSize_ = DEFAULT_LAYOUT_CACHE_SIZE;
    uint64_t cacheCleanupInterval_ = CACHE_CLEANUP_INTERVAL;
    
    // Internal layout creation
    std::unique_ptr<CachedDescriptorLayout> createLayoutInternal(const DescriptorLayoutSpec& spec);
    VkDescriptorSetLayout createVulkanLayout(const DescriptorLayoutSpec& spec);
    
    // Cache management helpers
    void evictLeastRecentlyUsed();
    bool shouldEvictLayout(const CachedDescriptorLayout& layout, uint64_t currentFrame) const;
    
    // Pool size calculation
    std::vector<VkDescriptorPoolSize> calculatePoolSizes(const std::vector<VkDescriptorSetLayout>& layouts,
                                                        uint32_t maxSets) const;
    void updatePoolSizeHints(CachedDescriptorLayout& cachedLayout);
    
    // Device capability initialization
    void queryDeviceFeatures();
    void initializeCommonLayouts();
    
    // Validation helpers
    bool validateBinding(const DescriptorBinding& binding) const;
    bool checkBindlessSupport(const DescriptorLayoutSpec& spec) const;
    
    // Debug helpers
    std::string descriptorTypeToString(VkDescriptorType type) const;
    std::string shaderStagesToString(VkShaderStageFlags stages) const;
};

// Utility functions for common descriptor layout patterns
namespace DescriptorLayoutPresets {
    // Entity rendering layouts (for your use case)
    DescriptorLayoutSpec createEntityGraphicsLayout();
    DescriptorLayoutSpec createEntityComputeLayout();
    
    // Common rendering layouts
    DescriptorLayoutSpec createMaterialLayout(uint32_t textureCount = 4);
    DescriptorLayoutSpec createLightingLayout();
    DescriptorLayoutSpec createShadowMappingLayout();
    
    // UI and post-processing
    DescriptorLayoutSpec createUILayout();
    DescriptorLayoutSpec createPostProcessLayout();
    
    // Bindless variants
    DescriptorLayoutSpec createBindlessTextureLayout(uint32_t maxTextures = MAX_BINDLESS_TEXTURES);
    DescriptorLayoutSpec createBindlessBufferLayout(uint32_t maxBuffers = MAX_BINDLESS_BUFFERS);
    
    // Debug and profiling
    DescriptorLayoutSpec createDebugLayout();
    DescriptorLayoutSpec createProfilerLayout();
}

// Helper builder for complex layouts
class DescriptorLayoutBuilder {
public:
    DescriptorLayoutBuilder& addUniformBuffer(uint32_t binding, VkShaderStageFlags stages = VK_SHADER_STAGE_ALL);
    DescriptorLayoutBuilder& addStorageBuffer(uint32_t binding, VkShaderStageFlags stages = VK_SHADER_STAGE_COMPUTE_BIT);
    DescriptorLayoutBuilder& addCombinedImageSampler(uint32_t binding, uint32_t count = 1, 
                                                    VkShaderStageFlags stages = VK_SHADER_STAGE_FRAGMENT_BIT);
    DescriptorLayoutBuilder& addStorageImage(uint32_t binding, VkShaderStageFlags stages = VK_SHADER_STAGE_COMPUTE_BIT);
    DescriptorLayoutBuilder& addSampler(uint32_t binding, VkShaderStageFlags stages = VK_SHADER_STAGE_FRAGMENT_BIT);
    
    // Bindless variants
    DescriptorLayoutBuilder& addBindlessTextures(uint32_t binding, uint32_t maxCount = MAX_BINDLESS_TEXTURES);
    DescriptorLayoutBuilder& addBindlessBuffers(uint32_t binding, uint32_t maxCount = MAX_BINDLESS_BUFFERS);
    
    // Configuration
    DescriptorLayoutBuilder& setName(const std::string& name);
    DescriptorLayoutBuilder& enableUpdateAfterBind(bool enable = true);
    DescriptorLayoutBuilder& enablePartiallyBound(bool enable = true);
    
    // Build the layout specification
    DescriptorLayoutSpec build();
    
    // Reset for reuse
    void reset();

private:
    DescriptorLayoutSpec spec_;
};