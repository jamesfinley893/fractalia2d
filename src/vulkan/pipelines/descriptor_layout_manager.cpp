#include "descriptor_layout_manager.h"
#include "../core/vulkan_function_loader.h"
#include "hash_utils.h"
#include <iostream>
#include <algorithm>
#include <cassert>
#include <sstream>

// DescriptorBinding implementation
bool DescriptorBinding::operator==(const DescriptorBinding& other) const {
    return binding == other.binding &&
           type == other.type &&
           descriptorCount == other.descriptorCount &&
           stageFlags == other.stageFlags &&
           isBindless == other.isBindless &&
           maxBindlessDescriptors == other.maxBindlessDescriptors &&
           immutableSamplers == other.immutableSamplers;
}

size_t DescriptorBinding::getHash() const {
    VulkanHash::HashCombiner hasher;
    
    hasher.combine(binding)
          .combine(type)
          .combine(descriptorCount)
          .combine(stageFlags)
          .combine(isBindless)
          .combine(maxBindlessDescriptors)
          .combineContainer(immutableSamplers);
    
    return hasher.get();
}

// DescriptorLayoutSpec implementation
bool DescriptorLayoutSpec::operator==(const DescriptorLayoutSpec& other) const {
    return bindings == other.bindings &&
           flags == other.flags &&
           enableBindless == other.enableBindless &&
           enableUpdateAfterBind == other.enableUpdateAfterBind &&
           enablePartiallyBound == other.enablePartiallyBound;
}

size_t DescriptorLayoutSpec::getHash() const {
    VulkanHash::HashCombiner hasher;
    
    for (const auto& binding : bindings) {
        hasher.combine(binding.getHash());
    }
    
    hasher.combine(flags)
          .combine(enableBindless)
          .combine(enableUpdateAfterBind)
          .combine(enablePartiallyBound);
    
    return hasher.get();
}

// DescriptorLayoutManager implementation
DescriptorLayoutManager::DescriptorLayoutManager() {
}

DescriptorLayoutManager::~DescriptorLayoutManager() {
    cleanupBeforeContextDestruction();
}

bool DescriptorLayoutManager::initialize(const VulkanContext& context) {
    this->context_ = &context;
    
    // Query device features and properties for bindless support
    queryDeviceFeatures();
    
    // Initialize common layouts for performance
    initializeCommonLayouts();
    
    std::cout << "DescriptorLayoutManager initialized successfully" << std::endl;
    
    if (supportsBindless()) {
        std::cout << "  - Bindless descriptors supported (max: " << getMaxBindlessDescriptors() << ")" << std::endl;
    }
    if (supportsUpdateAfterBind()) {
        std::cout << "  - Update after bind supported" << std::endl;
    }
    
    return true;
}

void DescriptorLayoutManager::cleanup() {
    cleanupBeforeContextDestruction();
}

void DescriptorLayoutManager::cleanupBeforeContextDestruction() {
    if (!context_) return;
    
    // Clear layout cache (RAII handles cleanup automatically)
    clearCache();
    
    // Reset common layouts (RAII handles cleanup automatically)
    commonLayouts.uniformBuffer.reset();
    commonLayouts.storageBuffer.reset();
    commonLayouts.combinedImageSampler.reset();
    commonLayouts.storageImage.reset();
    commonLayouts.bindlessTextures.reset();
    commonLayouts.bindlessBuffers.reset();
    
    // Clear managed pools (RAII handles cleanup automatically)
    managedPools_.clear();
    
    context_ = nullptr;
}

VkDescriptorSetLayout DescriptorLayoutManager::getLayout(const DescriptorLayoutSpec& spec) {
    // Check cache first
    auto it = layoutCache_.find(spec);
    if (it != layoutCache_.end()) {
        // Cache hit
        stats.cacheHits++;
        it->second->lastUsedFrame = stats.cacheHits + stats.cacheMisses;  // Rough frame counter
        it->second->useCount++;
        return it->second->layout.get();
    }
    
    // Cache miss - create new layout
    stats.cacheMisses++;
    
    auto cachedLayout = createLayoutInternal(spec);
    if (!cachedLayout) {
        std::cerr << "Failed to create descriptor layout: " << spec.layoutName << std::endl;
        return VK_NULL_HANDLE;
    }
    
    VkDescriptorSetLayout layout = cachedLayout->layout.get();
    
    // Store in cache
    layoutCache_[spec] = std::move(cachedLayout);
    stats.totalLayouts++;
    
    if (spec.enableBindless) {
        stats.bindlessLayouts++;
    }
    
    // Check if cache needs cleanup
    if (layoutCache_.size() > maxCacheSize_) {
        evictLeastRecentlyUsed();
    }
    
    return layout;
}

VkDescriptorSetLayout DescriptorLayoutManager::createLayout(const DescriptorLayoutSpec& spec) {
    if (!validateLayoutSpec(spec)) {
        std::cerr << "Invalid descriptor layout specification: " << spec.layoutName << std::endl;
        return VK_NULL_HANDLE;
    }
    
    return createVulkanLayout(spec);
}

std::unique_ptr<CachedDescriptorLayout> DescriptorLayoutManager::createLayoutInternal(const DescriptorLayoutSpec& spec) {
    if (!validateLayoutSpec(spec)) {
        return nullptr;
    }
    
    auto cachedLayout = std::make_unique<CachedDescriptorLayout>();
    cachedLayout->spec = spec;
    cachedLayout->isBindless = spec.enableBindless;
    
    // Create the Vulkan layout
    VkDescriptorSetLayout rawLayout = createVulkanLayout(spec);
    if (rawLayout == VK_NULL_HANDLE) {
        return nullptr;
    }
    cachedLayout->layout = vulkan_raii::make_descriptor_set_layout(rawLayout, context_);
    
    // Update pool size hints
    updatePoolSizeHints(*cachedLayout);
    
    return cachedLayout;
}

VkDescriptorSetLayout DescriptorLayoutManager::createVulkanLayout(const DescriptorLayoutSpec& spec) {
    std::vector<VkDescriptorSetLayoutBinding> vulkanBindings;
    std::vector<VkDescriptorBindingFlags> bindingFlags;
    
    vulkanBindings.reserve(spec.bindings.size());
    bindingFlags.reserve(spec.bindings.size());
    
    bool hasBindless = false;
    
    for (const auto& binding : spec.bindings) {
        VkDescriptorSetLayoutBinding vulkanBinding{};
        vulkanBinding.binding = binding.binding;
        vulkanBinding.descriptorType = binding.type;
        vulkanBinding.descriptorCount = binding.isBindless ? binding.maxBindlessDescriptors : binding.descriptorCount;
        vulkanBinding.stageFlags = binding.stageFlags;
        vulkanBinding.pImmutableSamplers = binding.immutableSamplers.empty() ? nullptr : binding.immutableSamplers.data();
        
        vulkanBindings.push_back(vulkanBinding);
        
        // Set up binding flags for bindless
        VkDescriptorBindingFlags flags = 0;
        if (binding.isBindless) {
            flags |= VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT;
            hasBindless = true;
            
            if (spec.enableUpdateAfterBind) {
                flags |= VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
            }
            if (spec.enablePartiallyBound) {
                flags |= VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;
            }
        }
        bindingFlags.push_back(flags);
    }
    
    // Main layout create info
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.flags = spec.flags;
    layoutInfo.bindingCount = static_cast<uint32_t>(vulkanBindings.size());
    layoutInfo.pBindings = vulkanBindings.data();
    
    // Extended info for bindless support
    VkDescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsInfo{};
    if (hasBindless) {
        layoutInfo.flags |= VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
        
        bindingFlagsInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
        bindingFlagsInfo.bindingCount = static_cast<uint32_t>(bindingFlags.size());
        bindingFlagsInfo.pBindingFlags = bindingFlags.data();
        
        layoutInfo.pNext = &bindingFlagsInfo;
    }
    
    VkDescriptorSetLayout layout;
    VkResult result = context_->getLoader().vkCreateDescriptorSetLayout(
        context_->getDevice(), &layoutInfo, nullptr, &layout);
    
    if (result != VK_SUCCESS) {
        std::cerr << "Failed to create descriptor set layout: " << result << std::endl;
        return VK_NULL_HANDLE;
    }
    
    return layout;
}

VkDescriptorPool DescriptorLayoutManager::createOptimalPool(const DescriptorPoolConfig& config) {
    std::vector<VkDescriptorPoolSize> poolSizes;
    
    // Calculate pool sizes based on layout usage patterns
    if (config.uniformBufferMultiplier > 0) {
        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSize.descriptorCount = static_cast<uint32_t>(config.maxSets * config.uniformBufferMultiplier);
        poolSizes.push_back(poolSize);
    }
    
    if (config.storageBufferMultiplier > 0) {
        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSize.descriptorCount = static_cast<uint32_t>(config.maxSets * config.storageBufferMultiplier);
        poolSizes.push_back(poolSize);
    }
    
    if (config.sampledImageMultiplier > 0) {
        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        poolSize.descriptorCount = static_cast<uint32_t>(config.maxSets * config.sampledImageMultiplier);
        poolSizes.push_back(poolSize);
    }
    
    if (config.combinedImageSamplerMultiplier > 0) {
        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSize.descriptorCount = static_cast<uint32_t>(config.maxSets * config.combinedImageSamplerMultiplier);
        poolSizes.push_back(poolSize);
    }
    
    if (config.storageImageMultiplier > 0) {
        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        poolSize.descriptorCount = static_cast<uint32_t>(config.maxSets * config.storageImageMultiplier);
        poolSizes.push_back(poolSize);
    }
    
    if (config.samplerMultiplier > 0) {
        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_SAMPLER;
        poolSize.descriptorCount = static_cast<uint32_t>(config.maxSets * config.samplerMultiplier);
        poolSizes.push_back(poolSize);
    }
    
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.flags = config.allowFreeDescriptorSets ? VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT : 0;
    
    if (config.enableUpdateAfterBind) {
        poolInfo.flags |= VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
    }
    
    poolInfo.maxSets = config.maxSets;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    
    VkDescriptorPool rawPool;
    VkResult result = context_->getLoader().vkCreateDescriptorPool(
        context_->getDevice(), &poolInfo, nullptr, &rawPool);
    
    if (result != VK_SUCCESS) {
        std::cerr << "Failed to create descriptor pool: " << result << std::endl;
        return VK_NULL_HANDLE;
    }
    
    vulkan_raii::DescriptorPool pool = vulkan_raii::make_descriptor_pool(rawPool, context_);
    VkDescriptorPool handle = pool.get();
    managedPools_.push_back(std::move(pool));
    stats.activePools++;
    
    return handle;
}

VkDescriptorSetLayout DescriptorLayoutManager::getUniformBufferLayout(VkShaderStageFlags stages) {
    if (commonLayouts.uniformBuffer) {
        return commonLayouts.uniformBuffer.get();
    }
    
    DescriptorLayoutSpec spec;
    spec.layoutName = "UniformBuffer";
    
    DescriptorBinding binding{};
    binding.binding = 0;
    binding.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    binding.descriptorCount = 1;
    binding.stageFlags = stages;
    binding.debugName = "uniformBuffer";
    
    spec.bindings.push_back(binding);
    
    return getLayout(spec);
}

VkDescriptorSetLayout DescriptorLayoutManager::getStorageBufferLayout(VkShaderStageFlags stages) {
    if (commonLayouts.storageBuffer) {
        return commonLayouts.storageBuffer.get();
    }
    
    DescriptorLayoutSpec spec;
    spec.layoutName = "StorageBuffer";
    
    DescriptorBinding binding{};
    binding.binding = 0;
    binding.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binding.descriptorCount = 1;
    binding.stageFlags = stages;
    binding.debugName = "storageBuffer";
    
    spec.bindings.push_back(binding);
    
    return getLayout(spec);
}

void DescriptorLayoutManager::clearCache() {
    if (!context_) return;
    
    // Clear all cached layouts (RAII handles cleanup automatically)
    layoutCache_.clear();
    stats.totalLayouts = 0;
    stats.bindlessLayouts = 0;
}

void DescriptorLayoutManager::evictLeastRecentlyUsed() {
    if (layoutCache_.empty()) return;
    
    // Find least recently used layout
    auto lruIt = layoutCache_.begin();
    for (auto it = layoutCache_.begin(); it != layoutCache_.end(); ++it) {
        if (it->second->lastUsedFrame < lruIt->second->lastUsedFrame) {
            lruIt = it;
        }
    }
    
    // Erase the layout (RAII handles cleanup automatically)
    layoutCache_.erase(lruIt);
    stats.totalLayouts--;
}

void DescriptorLayoutManager::updatePoolSizeHints(CachedDescriptorLayout& cachedLayout) {
    for (const auto& binding : cachedLayout.spec.bindings) {
        uint32_t count = binding.isBindless ? binding.maxBindlessDescriptors : binding.descriptorCount;
        
        switch (binding.type) {
            case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
            case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
                cachedLayout.poolSizeHints.uniformBuffers += count;
                break;
            case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
            case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
                cachedLayout.poolSizeHints.storageBuffers += count;
                break;
            case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
                cachedLayout.poolSizeHints.sampledImages += count;
                break;
            case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
                cachedLayout.poolSizeHints.storageImages += count;
                break;
            case VK_DESCRIPTOR_TYPE_SAMPLER:
                cachedLayout.poolSizeHints.samplers += count;
                break;
            case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
                cachedLayout.poolSizeHints.combinedImageSamplers += count;
                break;
            case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
                cachedLayout.poolSizeHints.inputAttachments += count;
                break;
            default:
                break;
        }
    }
}

void DescriptorLayoutManager::queryDeviceFeatures() {
    // Query descriptor indexing features
    indexingFeatures_.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES;
    
    VkPhysicalDeviceFeatures2 features2{};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &indexingFeatures_;
    
    // Note: In a full implementation, this would call vkGetPhysicalDeviceFeatures2
    (void)features2; // Suppress unused variable warning
    
    // Note: This would need proper implementation with vkGetPhysicalDeviceFeatures2
    // For now, we'll assume basic support
    indexingFeatures_.descriptorBindingVariableDescriptorCount = VK_TRUE;
    indexingFeatures_.descriptorBindingPartiallyBound = VK_TRUE;
    indexingFeatures_.descriptorBindingUpdateUnusedWhilePending = VK_TRUE;
    indexingFeatures_.descriptorBindingStorageBufferUpdateAfterBind = VK_TRUE;
    
    // Query properties
    context_->getLoader().vkGetPhysicalDeviceProperties(context_->getPhysicalDevice(), &deviceProperties_);
}

void DescriptorLayoutManager::initializeCommonLayouts() {
    // Initialize the most commonly used layouts for performance
    // These will be cached and reused frequently
    
    // Note: Implementation would create actual common layouts here
    // For now, they'll be created on-demand through getLayout()
}

bool DescriptorLayoutManager::validateLayoutSpec(const DescriptorLayoutSpec& spec) const {
    if (spec.bindings.empty()) {
        std::cerr << "Layout validation failed: no bindings specified" << std::endl;
        return false;
    }
    
    for (const auto& binding : spec.bindings) {
        if (!validateBinding(binding)) {
            return false;
        }
    }
    
    if (spec.enableBindless && !checkBindlessSupport(spec)) {
        return false;
    }
    
    return true;
}

bool DescriptorLayoutManager::validateBinding(const DescriptorBinding& binding) const {
    if (binding.descriptorCount == 0 && !binding.isBindless) {
        std::cerr << "Binding validation failed: descriptor count is zero" << std::endl;
        return false;
    }
    
    if (binding.isBindless && binding.maxBindlessDescriptors == 0) {
        std::cerr << "Binding validation failed: bindless descriptor count is zero" << std::endl;
        return false;
    }
    
    if (binding.stageFlags == 0) {
        std::cerr << "Binding validation failed: no shader stages specified" << std::endl;
        return false;
    }
    
    return true;
}

bool DescriptorLayoutManager::checkBindlessSupport(const DescriptorLayoutSpec& spec) const {
    if (!supportsBindless()) {
        std::cerr << "Bindless descriptors not supported on this device" << std::endl;
        return false;
    }
    
    if (spec.enableUpdateAfterBind && !supportsUpdateAfterBind()) {
        std::cerr << "Update after bind not supported on this device" << std::endl;
        return false;
    }
    
    return true;
}

bool DescriptorLayoutManager::supportsBindless() const {
    return indexingFeatures_.descriptorBindingVariableDescriptorCount == VK_TRUE;
}

bool DescriptorLayoutManager::supportsUpdateAfterBind() const {
    return indexingFeatures_.descriptorBindingStorageBufferUpdateAfterBind == VK_TRUE;
}

uint32_t DescriptorLayoutManager::getMaxBindlessDescriptors() const {
    // Return a conservative default
    return 16384;
}

void DescriptorLayoutManager::resetFrameStats() {
    stats.hitRatio = static_cast<float>(stats.cacheHits) / static_cast<float>(stats.cacheHits + stats.cacheMisses);
}

std::string DescriptorLayoutManager::descriptorTypeToString(VkDescriptorType type) const {
    switch (type) {
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER: return "UNIFORM_BUFFER";
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: return "STORAGE_BUFFER";
        case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER: return "COMBINED_IMAGE_SAMPLER";
        case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE: return "STORAGE_IMAGE";
        case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE: return "SAMPLED_IMAGE";
        case VK_DESCRIPTOR_TYPE_SAMPLER: return "SAMPLER";
        default: return "UNKNOWN";
    }
}

// DescriptorLayoutPresets namespace implementation
namespace DescriptorLayoutPresets {
    DescriptorLayoutSpec createEntityGraphicsLayout() {
        DescriptorLayoutSpec spec;
        spec.layoutName = "EntityGraphics";
        
        // UBO for camera/view matrices
        DescriptorBinding uboBinding{};
        uboBinding.binding = 0;
        uboBinding.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        uboBinding.descriptorCount = 1;
        uboBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        uboBinding.debugName = "cameraUBO";
        
        // Storage buffer for entity data
        DescriptorBinding entityBinding{};
        entityBinding.binding = 1;
        entityBinding.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        entityBinding.descriptorCount = 1;
        entityBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        entityBinding.debugName = "entityBuffer";
        
        // Storage buffer for positions
        DescriptorBinding positionBinding{};
        positionBinding.binding = 2;
        positionBinding.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        positionBinding.descriptorCount = 1;
        positionBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        positionBinding.debugName = "positionBuffer";
        
        spec.bindings = {uboBinding, entityBinding, positionBinding};
        return spec;
    }
    
    DescriptorLayoutSpec createEntityComputeLayout() {
        DescriptorLayoutSpec spec;
        spec.layoutName = "EntityComputeSoA";
        
        // SoA Structure of Arrays layout matching compute shaders
        
        // Binding 0: VelocityBuffer (velocity.xy, damping, reserved)
        DescriptorBinding velocityBinding{};
        velocityBinding.binding = 0;
        velocityBinding.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        velocityBinding.descriptorCount = 1;
        velocityBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        velocityBinding.debugName = "velocityBuffer";
        
        // Binding 1: MovementParamsBuffer (amplitude, frequency, phase, timeOffset)
        DescriptorBinding movementParamsBinding{};
        movementParamsBinding.binding = 1;
        movementParamsBinding.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        movementParamsBinding.descriptorCount = 1;
        movementParamsBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        movementParamsBinding.debugName = "movementParamsBuffer";
        
        // Binding 2: RuntimeStateBuffer (totalTime, initialized, stateTimer, entityState)
        DescriptorBinding runtimeStateBinding{};
        runtimeStateBinding.binding = 2;
        runtimeStateBinding.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        runtimeStateBinding.descriptorCount = 1;
        runtimeStateBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        runtimeStateBinding.debugName = "runtimeStateBuffer";
        
        // Binding 3: PositionBuffer (output positions for graphics)
        DescriptorBinding positionOutputBinding{};
        positionOutputBinding.binding = 3;
        positionOutputBinding.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        positionOutputBinding.descriptorCount = 1;
        positionOutputBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        positionOutputBinding.debugName = "positionOutputBuffer";
        
        // Binding 4: CurrentPositionBuffer (physics integration state)
        DescriptorBinding currentPosBinding{};
        currentPosBinding.binding = 4;
        currentPosBinding.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        currentPosBinding.descriptorCount = 1;
        currentPosBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        currentPosBinding.debugName = "currentPositionBuffer";
        
        // Binding 7: SpatialMapBuffer (spatial hash grid for collision detection)
        DescriptorBinding spatialMapBinding{};
        spatialMapBinding.binding = 7;
        spatialMapBinding.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        spatialMapBinding.descriptorCount = 1;
        spatialMapBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        spatialMapBinding.debugName = "spatialMapBuffer";
        
        spec.bindings = {velocityBinding, movementParamsBinding, runtimeStateBinding, positionOutputBinding, currentPosBinding, spatialMapBinding};
        return spec;
    }
}

// DescriptorLayoutBuilder implementation
DescriptorLayoutBuilder& DescriptorLayoutBuilder::addUniformBuffer(uint32_t binding, VkShaderStageFlags stages) {
    DescriptorBinding desc{};
    desc.binding = binding;
    desc.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    desc.descriptorCount = 1;
    desc.stageFlags = stages;
    desc.debugName = "uniformBuffer_" + std::to_string(binding);
    
    spec_.bindings.push_back(desc);
    return *this;
}

DescriptorLayoutBuilder& DescriptorLayoutBuilder::addStorageBuffer(uint32_t binding, VkShaderStageFlags stages) {
    DescriptorBinding desc{};
    desc.binding = binding;
    desc.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    desc.descriptorCount = 1;
    desc.stageFlags = stages;
    desc.debugName = "storageBuffer_" + std::to_string(binding);
    
    spec_.bindings.push_back(desc);
    return *this;
}

DescriptorLayoutBuilder& DescriptorLayoutBuilder::setName(const std::string& name) {
    spec_.layoutName = name;
    return *this;
}

DescriptorLayoutSpec DescriptorLayoutBuilder::build() {
    return spec_;
}

void DescriptorLayoutBuilder::reset() {
    spec_ = DescriptorLayoutSpec{};
}

void DescriptorLayoutManager::optimizeCache(uint64_t currentFrame) {
    // Simple LRU eviction for descriptor layout cache
    for (auto it = layoutCache_.begin(); it != layoutCache_.end();) {
        if (currentFrame - it->second->lastUsedFrame > CACHE_CLEANUP_INTERVAL) {
            // Erase layout (RAII handles cleanup automatically)
            it = layoutCache_.erase(it);
            stats.totalLayouts--;
        } else {
            ++it;
        }
    }
}

void DescriptorLayoutManager::warmupCache(const std::vector<DescriptorLayoutSpec>& commonLayouts) {
    // Pre-create commonly used descriptor layouts
    for (const auto& spec : commonLayouts) {
        getLayout(spec); // This will create and cache the layout
    }
}