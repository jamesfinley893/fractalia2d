#include "shader_manager.h"
#include "../core/vulkan_function_loader.h"
#include "../core/vulkan_raii.h"
#include "hash_utils.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <filesystem>

// ShaderModuleSpec implementation
bool ShaderModuleSpec::operator==(const ShaderModuleSpec& other) const {
    return filePath == other.filePath &&
           sourceType == other.sourceType &&
           stageInfo.stage == other.stageInfo.stage &&
           stageInfo.entryPoint == other.stageInfo.entryPoint &&
           includePaths == other.includePaths &&
           defines == other.defines &&
           stageInfo.specializationData == other.stageInfo.specializationData;
}

size_t ShaderModuleSpec::getHash() const {
    VulkanHash::HashCombiner hasher;
    
    hasher.combine(filePath)
          .combine(static_cast<uint32_t>(sourceType))
          .combine(stageInfo.stage)
          .combine(stageInfo.entryPoint)
          .combineContainer(includePaths)
          .combineContainer(stageInfo.specializationData);
    
    for (const auto& [key, value] : defines) {
        hasher.combine(key + value);
    }
    
    return hasher.get();
}

// ShaderManager implementation
ShaderManager::ShaderManager() {
}

ShaderManager::~ShaderManager() {
    cleanup();
}

bool ShaderManager::initialize(const VulkanContext& context) {
    this->context_ = &context;
    
    // Check for external shader compilers
    if (ShaderCompiler::isGlslcAvailable()) {
        std::cout << "ShaderManager: glslc compiler found" << std::endl;
    } else {
        std::cout << "ShaderManager: glslc compiler not found - GLSL compilation disabled" << std::endl;
    }
    
    if (ShaderCompiler::isSpirvOptAvailable()) {
        std::cout << "ShaderManager: spirv-opt optimizer found" << std::endl;
    }
    
    std::cout << "ShaderManager initialized successfully" << std::endl;
    return true;
}

void ShaderManager::cleanup() {
    cleanupBeforeContextDestruction();
}

void ShaderManager::cleanupBeforeContextDestruction() {
    if (!context_) return;
    
    // Clear shader cache before context destruction
    clearCache();
    
    context_ = nullptr;
}

VkShaderModule ShaderManager::loadShader(const ShaderModuleSpec& spec) {
    // Check cache first
    auto it = shaderCache_.find(spec);
    if (it != shaderCache_.end()) {
        // Check for hot reload if enabled
        if (hotReloadEnabled && spec.enableHotReload) {
            if (isFileNewer(spec.filePath, it->second->sourceModified)) {
                std::cout << "Hot reloading shader: " << spec.filePath << std::endl;
                if (reloadShader(spec)) {
                    stats.hotReloadsThisFrame++;
                    it = shaderCache_.find(spec);  // Get updated cache entry
                }
            }
        }
        
        // Cache hit
        stats.cacheHits++;
        it->second->lastUsedFrame = stats.cacheHits + stats.cacheMisses;  // Rough frame counter
        it->second->useCount++;
        return it->second->module.get();
    }
    
    // Cache miss - create new shader
    stats.cacheMisses++;
    stats.compilationsThisFrame++;
    
    auto cachedShader = createShaderInternal(spec);
    if (!cachedShader) {
        std::cerr << "Failed to create shader: " << spec.filePath << std::endl;
        return VK_NULL_HANDLE;
    }
    
    VkShaderModule module = cachedShader->module.get();
    
    // Store in cache
    shaderCache_[spec] = std::move(cachedShader);
    stats.totalShaders++;
    
    // Check if cache needs cleanup
    if (shaderCache_.size() > maxCacheSize_) {
        evictLeastRecentlyUsed();
    }
    
    return module;
}

VkShaderModule ShaderManager::loadShaderFromFile(const std::string& filePath,
                                                VkShaderStageFlagBits stage,
                                                const std::string& entryPoint) {
    ShaderModuleSpec spec;
    spec.filePath = filePath;
    spec.sourceType = ShaderSourceType::SPIRV_BINARY;  // Assume SPIR-V by default
    spec.stageInfo.stage = stage;
    spec.stageInfo.entryPoint = entryPoint;
    spec.stageInfo.debugName = std::filesystem::path(filePath).filename().string();
    
    // Auto-detect shader type from extension
    std::string extension = std::filesystem::path(filePath).extension().string();
    if (extension == ".glsl" || extension == ".vert" || extension == ".frag" || 
        extension == ".comp" || extension == ".geom" || extension == ".tesc" || extension == ".tese") {
        spec.sourceType = ShaderSourceType::GLSL_SOURCE;
    }
    
    return loadShader(spec);
}

VkShaderModule ShaderManager::loadSPIRVFromFile(const std::string& filePath) {
    VkShaderStageFlagBits stage = getShaderStageFromFilename(filePath);
    return loadShaderFromFile(filePath, stage);
}

std::unique_ptr<CachedShaderModule> ShaderManager::createShaderInternal(const ShaderModuleSpec& spec) {
    auto startTime = std::chrono::high_resolution_clock::now();
    
    if (!fileExists(spec.filePath)) {
        std::cerr << "Shader file not found: " << spec.filePath << std::endl;
        return nullptr;
    }
    
    auto cachedShader = std::make_unique<CachedShaderModule>();
    cachedShader->spec = spec;
    cachedShader->sourceModified = getFileModifiedTime(spec.filePath);
    cachedShader->isHotReloadable = spec.enableHotReload;
    cachedShader->module.setContext(context_);
    
    // Load or compile shader based on source type
    std::vector<uint32_t> spirvCode;
    
    switch (spec.sourceType) {
        case ShaderSourceType::SPIRV_BINARY:
            spirvCode = loadSPIRVBinaryFromFile(spec.filePath);
            break;
            
        case ShaderSourceType::GLSL_SOURCE: {
            auto compilationResult = compileGLSLFromFile(spec.filePath, spec.defines);
            if (!compilationResult.success) {
                logShaderError(spec.filePath, compilationResult.errorMessage);
                return nullptr;
            }
            spirvCode = std::move(compilationResult.spirvCode);
            break;
        }
        
        case ShaderSourceType::HLSL_SOURCE:
            std::cerr << "HLSL compilation not yet implemented" << std::endl;
            return nullptr;
    }
    
    if (spirvCode.empty()) {
        std::cerr << "Failed to load shader code: " << spec.filePath << std::endl;
        return nullptr;
    }
    
    // Validate SPIR-V
    if (!validateSPIRV(spirvCode)) {
        std::cerr << "Invalid SPIR-V code: " << spec.filePath << std::endl;
        return nullptr;
    }
    
    // Create Vulkan shader module
    auto shaderModule = createVulkanShaderModule(spirvCode);
    if (!shaderModule) {
        std::cerr << "Failed to create Vulkan shader module: " << spec.filePath << std::endl;
        return nullptr;
    }
    cachedShader->module = std::move(shaderModule);
    
    // Store SPIR-V code for reflection
    cachedShader->spirvCode = std::move(spirvCode);
    
    // Perform shader reflection
    performBasicReflection(*cachedShader);
    
    auto endTime = std::chrono::high_resolution_clock::now();
    cachedShader->compilationTime = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime);
    stats.totalCompilationTime += cachedShader->compilationTime;
    
    logShaderCompilation(spec, cachedShader->compilationTime, true);
    
    return cachedShader;
}

vulkan_raii::ShaderModule ShaderManager::createVulkanShaderModule(const std::vector<uint32_t>& spirvCode) {
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = spirvCode.size() * sizeof(uint32_t);
    createInfo.pCode = spirvCode.data();
    
    VkShaderModule shaderModule;
    VkResult result = context_->getLoader().vkCreateShaderModule(
        context_->getDevice(), &createInfo, nullptr, &shaderModule);
    
    if (result != VK_SUCCESS) {
        std::cerr << "Failed to create shader module: " << result << std::endl;
        return vulkan_raii::ShaderModule(VK_NULL_HANDLE, context_);
    }
    
    return vulkan_raii::make_shader_module(shaderModule, context_);
}

std::vector<uint32_t> ShaderManager::loadSPIRVBinaryFromFile(const std::string& filePath) const {
    std::ifstream file(filePath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Failed to open SPIR-V file: " << filePath << std::endl;
        return {};
    }
    
    size_t fileSize = static_cast<size_t>(file.tellg());
    if (fileSize % sizeof(uint32_t) != 0) {
        std::cerr << "Invalid SPIR-V file size: " << filePath << std::endl;
        return {};
    }
    
    std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(buffer.data()), fileSize);
    file.close();
    
    return buffer;
}

bool ShaderManager::validateSPIRV(const std::vector<uint32_t>& spirvCode) const {
    if (spirvCode.empty() || spirvCode.size() < 5) {
        return false;
    }
    
    // Check SPIR-V magic number
    const uint32_t SPIRV_MAGIC = 0x07230203;
    if (spirvCode[0] != SPIRV_MAGIC) {
        return false;
    }
    
    // Basic validation - more comprehensive validation would require SPIRV-Tools
    return true;
}

ShaderCompilationResult ShaderManager::compileGLSLFromFile(const std::string& filePath,
                                                          const std::unordered_map<std::string, std::string>& defines) {
    std::string source = loadShaderSource(filePath);
    if (source.empty()) {
        ShaderCompilationResult result;
        result.success = false;
        result.errorMessage = "Failed to load shader source";
        return result;
    }
    
    VkShaderStageFlagBits stage = getShaderStageFromFilename(filePath);
    return compileGLSL(source, stage, filePath, defines);
}

ShaderCompilationResult ShaderManager::compileGLSL(const std::string& source,
                                                   VkShaderStageFlagBits stage,
                                                   const std::string& fileName,
                                                   const std::unordered_map<std::string, std::string>& defines) {
    ShaderCompilationResult result;
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Compile using external glslc compiler
    std::vector<uint32_t> spirvCode = compileSPIRVWithGlslc(source, stage, fileName, defines);
    
    auto endTime = std::chrono::high_resolution_clock::now();
    result.compilationTime = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime);
    
    if (spirvCode.empty()) {
        result.success = false;
        result.errorMessage = "GLSL compilation failed";
        return result;
    }
    
    result.success = true;
    result.spirvCode = std::move(spirvCode);
    return result;
}

std::vector<uint32_t> ShaderManager::compileSPIRVWithGlslc(const std::string& source,
                                                          VkShaderStageFlagBits stage,
                                                          const std::string& fileName,
                                                          const std::unordered_map<std::string, std::string>& defines) const {
    // For now, return empty vector as external compilation requires process execution
    // In a full implementation, this would:
    // 1. Write source to temporary file
    // 2. Execute glslc with appropriate arguments
    // 3. Read compiled SPIR-V from output file
    // 4. Clean up temporary files
    
    std::cerr << "GLSL compilation not implemented in this demo version" << std::endl;
    return {};
}

VkPipelineShaderStageCreateInfo ShaderManager::createShaderStage(VkShaderModule module,
                                                                 VkShaderStageFlagBits stage,
                                                                 const std::string& entryPoint,
                                                                 const VkSpecializationInfo* specializationInfo) {
    VkPipelineShaderStageCreateInfo shaderStageInfo{};
    shaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStageInfo.stage = stage;
    shaderStageInfo.module = module;
    shaderStageInfo.pName = entryPoint.c_str();
    shaderStageInfo.pSpecializationInfo = specializationInfo;
    
    return shaderStageInfo;
}

std::vector<VkPipelineShaderStageCreateInfo> ShaderManager::createGraphicsShaderStages(
    VkShaderModule vertexShader,
    VkShaderModule fragmentShader,
    VkShaderModule geometryShader,
    VkShaderModule tessControlShader,
    VkShaderModule tessEvalShader) {
    
    std::vector<VkPipelineShaderStageCreateInfo> stages;
    
    if (vertexShader != VK_NULL_HANDLE) {
        stages.push_back(createShaderStage(vertexShader, VK_SHADER_STAGE_VERTEX_BIT));
    }
    
    if (tessControlShader != VK_NULL_HANDLE) {
        stages.push_back(createShaderStage(tessControlShader, VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT));
    }
    
    if (tessEvalShader != VK_NULL_HANDLE) {
        stages.push_back(createShaderStage(tessEvalShader, VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT));
    }
    
    if (geometryShader != VK_NULL_HANDLE) {
        stages.push_back(createShaderStage(geometryShader, VK_SHADER_STAGE_GEOMETRY_BIT));
    }
    
    if (fragmentShader != VK_NULL_HANDLE) {
        stages.push_back(createShaderStage(fragmentShader, VK_SHADER_STAGE_FRAGMENT_BIT));
    }
    
    return stages;
}

VkPipelineShaderStageCreateInfo ShaderManager::createComputeShaderStage(VkShaderModule computeShader,
                                                                        const std::string& entryPoint) {
    return createShaderStage(computeShader, VK_SHADER_STAGE_COMPUTE_BIT, entryPoint);
}

void ShaderManager::clearCache() {
    if (!context_) return;
    
    // RAII wrappers automatically destroy shader modules
    shaderCache_.clear();
    stats.totalShaders = 0;
}

void ShaderManager::evictLeastRecentlyUsed() {
    if (shaderCache_.empty()) return;
    
    // Find least recently used shader
    auto lruIt = shaderCache_.begin();
    for (auto it = shaderCache_.begin(); it != shaderCache_.end(); ++it) {
        if (it->second->lastUsedFrame < lruIt->second->lastUsedFrame) {
            lruIt = it;
        }
    }
    
    // RAII wrapper automatically destroys shader module
    shaderCache_.erase(lruIt);
    stats.totalShaders--;
}

bool ShaderManager::reloadShader(const ShaderModuleSpec& spec) {
    auto it = shaderCache_.find(spec);
    if (it == shaderCache_.end()) {
        return false;
    }
    
    // RAII wrapper automatically destroys old shader module
    // Remove from cache
    shaderCache_.erase(it);
    stats.totalShaders--;
    
    // Reload shader
    VkShaderModule newModule = loadShader(spec);
    
    // Trigger reload callbacks
    auto callbackIt = reloadCallbacks_.find(spec.filePath);
    if (callbackIt != reloadCallbacks_.end()) {
        for (const auto& callback : callbackIt->second) {
            callback(newModule);
        }
    }
    
    return newModule != VK_NULL_HANDLE;
}

std::string ShaderManager::loadShaderSource(const std::string& filePath) const {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "Failed to open shader source file: " << filePath << std::endl;
        return {};
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

bool ShaderManager::fileExists(const std::string& path) const {
    return std::filesystem::exists(path);
}

std::filesystem::file_time_type ShaderManager::getFileModifiedTime(const std::string& path) const {
    try {
        return std::filesystem::last_write_time(path);
    } catch (const std::filesystem::filesystem_error&) {
        return std::filesystem::file_time_type{};
    }
}

bool ShaderManager::isFileNewer(const std::string& path, std::filesystem::file_time_type lastModified) const {
    auto currentModified = getFileModifiedTime(path);
    return currentModified > lastModified;
}

VkShaderStageFlagBits ShaderManager::getShaderStageFromFilename(const std::string& filename) const {
    std::string extension = std::filesystem::path(filename).extension().string();
    
    if (extension == ".vert" || filename.find("vertex") != std::string::npos) {
        return VK_SHADER_STAGE_VERTEX_BIT;
    } else if (extension == ".frag" || filename.find("fragment") != std::string::npos) {
        return VK_SHADER_STAGE_FRAGMENT_BIT;
    } else if (extension == ".comp" || filename.find("compute") != std::string::npos) {
        return VK_SHADER_STAGE_COMPUTE_BIT;
    } else if (extension == ".geom" || filename.find("geometry") != std::string::npos) {
        return VK_SHADER_STAGE_GEOMETRY_BIT;
    } else if (extension == ".tesc" || filename.find("tesscontrol") != std::string::npos) {
        return VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
    } else if (extension == ".tese" || filename.find("tesseval") != std::string::npos) {
        return VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
    }
    
    // Default to vertex shader
    return VK_SHADER_STAGE_VERTEX_BIT;
}

void ShaderManager::performBasicReflection(CachedShaderModule& cachedModule) const {
    // Basic reflection implementation
    // In a full implementation, this would use SPIRV-Reflect or similar
    
    // For now, just set some defaults based on shader stage
    VkShaderStageFlagBits stage = cachedModule.spec.stageInfo.stage;
    
    if (stage == VK_SHADER_STAGE_COMPUTE_BIT) {
        // Default compute workgroup size
        cachedModule.reflection.localSizeX = 32;
        cachedModule.reflection.localSizeY = 1;
        cachedModule.reflection.localSizeZ = 1;
    }
}

void ShaderManager::logShaderCompilation(const ShaderModuleSpec& spec,
                                        std::chrono::nanoseconds compilationTime,
                                        bool success) const {
    if (success) {
        std::cout << "Compiled shader: " << spec.filePath 
                  << " (time: " << compilationTime.count() / 1000000.0f << "ms)" << std::endl;
    } else {
        std::cerr << "Failed to compile shader: " << spec.filePath << std::endl;
    }
}

void ShaderManager::logShaderError(const std::string& shaderPath, const std::string& error) const {
    std::cerr << "Shader compilation error in " << shaderPath << ":\n" << error << std::endl;
}

void ShaderManager::resetFrameStats() {
    stats.compilationsThisFrame = 0;
    stats.hotReloadsThisFrame = 0;
    stats.hitRatio = static_cast<float>(stats.cacheHits) / static_cast<float>(stats.cacheHits + stats.cacheMisses);
}

// ShaderPresets namespace implementation
namespace ShaderPresets {
    ShaderModuleSpec createVertexShaderSpec(const std::string& path) {
        ShaderModuleSpec spec;
        spec.filePath = path;
        spec.sourceType = ShaderSourceType::SPIRV_BINARY;
        spec.stageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
        spec.stageInfo.entryPoint = "main";
        spec.stageInfo.debugName = std::filesystem::path(path).filename().string();
        spec.enableHotReload = true;
        return spec;
    }
    
    ShaderModuleSpec createFragmentShaderSpec(const std::string& path) {
        ShaderModuleSpec spec;
        spec.filePath = path;
        spec.sourceType = ShaderSourceType::SPIRV_BINARY;
        spec.stageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        spec.stageInfo.entryPoint = "main";
        spec.stageInfo.debugName = std::filesystem::path(path).filename().string();
        spec.enableHotReload = true;
        return spec;
    }
    
    ShaderModuleSpec createComputeShaderSpec(const std::string& path) {
        ShaderModuleSpec spec;
        spec.filePath = path;
        spec.sourceType = ShaderSourceType::SPIRV_BINARY;
        spec.stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        spec.stageInfo.entryPoint = "main";
        spec.stageInfo.debugName = std::filesystem::path(path).filename().string();
        spec.enableHotReload = true;
        return spec;
    }
    
    ShaderModuleSpec createEntityVertexShaderSpec() {
        return createVertexShaderSpec("shaders/compiled/vertex.spv");
    }
    
    ShaderModuleSpec createEntityFragmentShaderSpec() {
        return createFragmentShaderSpec("shaders/compiled/fragment.spv");
    }
    
    ShaderModuleSpec createEntityComputeShaderSpec() {
        return createComputeShaderSpec("shaders/compiled/movement_random.comp.spv");
    }
}

// ShaderCompiler static class implementation
bool ShaderCompiler::isGlslcAvailable() {
    // Simple check - in a real implementation, this would try to execute glslc --version
    return false;  // Disabled for this demo
}

bool ShaderCompiler::isSpirvOptAvailable() {
    // Simple check - in a real implementation, this would try to execute spirv-opt --version
    return false;  // Disabled for this demo
}

void ShaderManager::optimizeCache(uint64_t currentFrame) {
    // Simple LRU eviction for shader cache
    for (auto it = shaderCache_.begin(); it != shaderCache_.end();) {
        if (currentFrame - it->second->lastUsedFrame > CACHE_CLEANUP_INTERVAL) {
            // RAII wrapper automatically destroys shader module
            it = shaderCache_.erase(it);
            stats.totalShaders--;
        } else {
            ++it;
        }
    }
}