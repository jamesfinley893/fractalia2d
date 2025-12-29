#include "frame_graph.h"
#include "../core/vulkan_context.h"
#include "../core/vulkan_sync.h"
#include "../core/queue_manager.h"
#include "../monitoring/gpu_memory_monitor.h"
#include "../monitoring/gpu_timeout_detector.h"
#include <iostream>
#include <cassert>

FrameGraph::FrameGraph() {
}

FrameGraph::~FrameGraph() {
    cleanup();
}

bool FrameGraph::initialize(const VulkanContext& context, VulkanSync* sync, QueueManager* queueManager) {
    this->context_ = &context;
    this->sync_ = sync;
    this->queueManager_ = queueManager;
    
    if (context.getDevice() == VK_NULL_HANDLE || !sync || !queueManager) {
        std::cerr << "FrameGraph: Invalid context, sync, or queue manager objects" << std::endl;
        return false;
    }
    
    // Initialize modular components
    if (!resourceManager_.initialize(context)) {
        std::cerr << "FrameGraph: Failed to initialize ResourceManager" << std::endl;
        return false;
    }
    
    barrierManager_.initialize(&context);
    
    // Set up resource accessors for barrier manager
    barrierManager_.setResourceAccessors(
        [this](FrameGraphTypes::ResourceId id) { return resourceManager_.getBufferResource(id); },
        [this](FrameGraphTypes::ResourceId id) { return resourceManager_.getImageResource(id); }
    );
    
    initialized_ = true;
    std::cout << "FrameGraph initialized successfully with modular components" << std::endl;
    return true;
}

void FrameGraph::cleanup() {
    if (!initialized_) return;
    
    cleanupBeforeContextDestruction();
    
    nodes_.clear();
    executionOrder_.clear();
    barrierManager_.reset();
    resourceManager_.cleanup();
    
    nextNodeId_ = 1;
    compiled_ = false;
    initialized_ = false;
}

void FrameGraph::cleanupBeforeContextDestruction() {
    resourceManager_.cleanupBeforeContextDestruction();
}

void FrameGraph::setMemoryMonitor(GPUMemoryMonitor* monitor) {
    resourceManager_.setMemoryMonitor(monitor);
}

// Resource management delegation
FrameGraphTypes::ResourceId FrameGraph::createBuffer(const std::string& name, VkDeviceSize size, VkBufferUsageFlags usage) {
    return resourceManager_.createBuffer(name, size, usage);
}

FrameGraphTypes::ResourceId FrameGraph::createImage(const std::string& name, VkFormat format, VkExtent2D extent, VkImageUsageFlags usage) {
    return resourceManager_.createImage(name, format, extent, usage);
}

FrameGraphTypes::ResourceId FrameGraph::importExternalBuffer(const std::string& name, VkBuffer buffer, VkDeviceSize size, VkBufferUsageFlags usage) {
    return resourceManager_.importExternalBuffer(name, buffer, size, usage);
}

FrameGraphTypes::ResourceId FrameGraph::importExternalImage(const std::string& name, VkImage image, VkImageView view, VkFormat format, VkExtent2D extent) {
    return resourceManager_.importExternalImage(name, image, view, format, extent);
}

VkBuffer FrameGraph::getBuffer(FrameGraphTypes::ResourceId id) const {
    return resourceManager_.getBuffer(id);
}

VkImage FrameGraph::getImage(FrameGraphTypes::ResourceId id) const {
    return resourceManager_.getImage(id);
}

VkImageView FrameGraph::getImageView(FrameGraphTypes::ResourceId id) const {
    return resourceManager_.getImageView(id);
}

void FrameGraph::debugPrint() const {
    resourceManager_.debugPrint();
    std::cout << "Nodes (" << nodes_.size() << "):" << std::endl;
    for (const auto& [id, node] : nodes_) {
        std::cout << "  ID " << id << ": " << node->getName() << std::endl;
    }
    
    if (compiled_) {
        std::cout << "Execution Order: ";
        for (auto nodeId : executionOrder_) {
            auto it = nodes_.find(nodeId);
            if (it != nodes_.end()) {
                std::cout << it->second->getName() << " -> ";
            }
        }
        std::cout << "END" << std::endl;
    }
}

void FrameGraph::logAllocationTelemetry() const {
    resourceManager_.logAllocationTelemetry();
}

void FrameGraph::performResourceCleanup() {
    resourceManager_.performResourceCleanup();
}

bool FrameGraph::isMemoryPressureCritical() const {
    return resourceManager_.isMemoryPressureCritical();
}

void FrameGraph::evictNonCriticalResources() {
    resourceManager_.evictNonCriticalResources();
}

bool FrameGraph::compile() {
    if (!initialized_) {
        std::cerr << "FrameGraph: Cannot compile, not initialized" << std::endl;
        return false;
    }
    
    // Compiling frame graph
    static int compileCount = 0;
    if (compileCount++ < 5) {  // Only log first 5 compilations
        std::cout << "FrameGraph compilation #" << compileCount << std::endl;
    }
    
    // Backup current state for transactional compilation
    compiler_.backupState(executionOrder_, compiled_);
    
    // Clear current compilation state
    executionOrder_.clear();
    barrierManager_.reset();
    compiled_ = false;
    
    // Use compiler for dependency analysis and topological sorting
    FrameGraphCompilation::CircularDependencyReport cycleReport;
    if (!compiler_.compileWithCycleDetection(nodes_, executionOrder_, cycleReport)) {
        std::cerr << "FrameGraph: Compilation failed due to circular dependencies" << std::endl;
        
        // Print detailed cycle analysis
        if (!cycleReport.cycles.empty()) {
            std::cerr << "\nDetailed Cycle Analysis:" << std::endl;
            for (size_t i = 0; i < cycleReport.cycles.size(); i++) {
                const auto& cycle = cycleReport.cycles[i];
                std::cerr << "Cycle " << (i + 1) << ": ";
                
                for (size_t j = 0; j < cycle.nodeChain.size(); j++) {
                    auto nodeIt = nodes_.find(cycle.nodeChain[j]);
                    if (nodeIt != nodes_.end()) {
                        std::cerr << nodeIt->second->getName();
                        if (j < cycle.resourceChain.size()) {
                            std::cerr << " --[resource " << cycle.resourceChain[j] << "]--> ";
                        }
                    }
                }
                std::cerr << std::endl;
            }
            
            std::cerr << "\nResolution Suggestions:" << std::endl;
            for (const auto& suggestion : cycleReport.resolutionSuggestions) {
                std::cerr << "- " << suggestion << std::endl;
            }
        }
        
        // Attempt partial compilation as fallback
        FrameGraphCompilation::PartialCompilationResult partialResult = compiler_.attemptPartialCompilation(nodes_);
        if (partialResult.hasValidSubgraph) {
            std::cerr << "\nFalling back to partial compilation:" << std::endl;
            std::cerr << "- Executing " << partialResult.validNodes.size() << " valid nodes" << std::endl;
            std::cerr << "- Skipping " << partialResult.problematicNodes.size() << " problematic nodes" << std::endl;
            
            executionOrder_ = partialResult.validNodes;
            
            // Analyze and create barriers for valid subgraph
            barrierManager_.analyzeBarrierRequirements(executionOrder_, nodes_);
            barrierManager_.createOptimalBarrierBatches(executionOrder_, nodes_);
            
            // Initialize valid nodes only with new standardized lifecycle
            for (auto nodeId : executionOrder_) {
                auto it = nodes_.find(nodeId);
                if (it != nodes_.end()) {
                    if (!it->second->initializeNode(*this)) {
                        std::cerr << "FrameGraph: Node initialization failed for node " << nodeId << " in partial compilation" << std::endl;
                        compiler_.restoreState(executionOrder_, compiled_);
                        return false;
                    }
                }
            }
            
            compiled_ = true;
            std::cerr << "Partial compilation successful" << std::endl;
            return true;
        }
        
        compiler_.restoreState(executionOrder_, compiled_);
        return false;
    }
    
    // Analyze and create synchronization barriers
    barrierManager_.analyzeBarrierRequirements(executionOrder_, nodes_);
    barrierManager_.createOptimalBarrierBatches(executionOrder_, nodes_);
    
    // Initialize nodes with standardized lifecycle
    // Lifecycle: initializeNode() once during compilation, then per-frame: prepareFrame() → execute() → releaseFrame()
    for (auto nodeId : executionOrder_) {
        auto it = nodes_.find(nodeId);
        if (it != nodes_.end()) {
            if (!it->second->initializeNode(*this)) {
                std::cerr << "FrameGraph: Node initialization failed for node " << nodeId << std::endl;
                return false;
            }
        }
    }
    
    compiled_ = true;
    std::cout << "FrameGraph compilation successful (" << executionOrder_.size() << " nodes)" << std::endl;
    
    return true;
}


FrameGraph::ExecutionResult FrameGraph::execute(uint32_t frameIndex, float time, float deltaTime, uint32_t globalFrame) {
    ExecutionResult result;
    if (!compiled_) {
        std::cerr << "FrameGraph: Cannot execute, not compiled" << std::endl;
        return result;
    }
    
    // Store global frame for node access
    currentGlobalFrame_ = globalFrame;
    
    if (!sync_) {
        std::cerr << "FrameGraph: Cannot execute, no sync object" << std::endl;
        return result;
    }
    
    // Check for memory pressure and perform cleanup if needed
    if (isMemoryPressureCritical()) {
        performResourceCleanup();
        
        // If still critical, attempt resource eviction
        if (isMemoryPressureCritical()) {
            evictNonCriticalResources();
        }
    }
    
    // Analyze which command buffers we'll need
    auto [computeNeeded, graphicsNeeded] = analyzeQueueRequirements();
    result.computeCommandBufferUsed = computeNeeded;
    result.graphicsCommandBufferUsed = graphicsNeeded;
    
    // Reset command buffers for this frame before recording
    if (queueManager_) {
        queueManager_->resetCommandBuffersForFrame(frameIndex);
    }
    
    // Begin only the command buffers that will be used
    beginCommandBuffers(computeNeeded, graphicsNeeded, frameIndex);
    
    // Execute nodes with timeout monitoring if available
    bool computeExecuted = false;
    if (timeoutDetector_) {
        if (!executeWithTimeoutMonitoring(frameIndex, time, deltaTime, globalFrame, computeExecuted)) {
            // Timeout occurred, end command buffers and return early
            endCommandBuffers(computeNeeded, graphicsNeeded, frameIndex);
            handleExecutionTimeout();
            return result;
        }
    } else {
        executeNodesInOrder(frameIndex, time, deltaTime, globalFrame, computeExecuted);
    }
    
    // End only the command buffers that were begun
    endCommandBuffers(computeNeeded, graphicsNeeded, frameIndex);
    
    // Frame graph complete - command buffers are ready for submission by VulkanRenderer
    return result;
}

void FrameGraph::reset() {
    resourceManager_.reset();
    
    // Reset barrier manager
    barrierManager_.reset();
    
    // Keep execution order and compilation state if compiled
    if (!compiled_) {
        executionOrder_.clear();
    }
}

void FrameGraph::removeSwapchainResources() {
    resourceManager_.removeSwapchainResources();
}

// Private helper methods

std::pair<bool, bool> FrameGraph::analyzeQueueRequirements() const {
    bool computeNeeded = false;
    bool graphicsNeeded = false;
    
    for (auto nodeId : executionOrder_) {
        auto it = nodes_.find(nodeId);
        if (it != nodes_.end()) {
            if (it->second->needsComputeQueue()) computeNeeded = true;
            if (it->second->needsGraphicsQueue()) graphicsNeeded = true;
        }
    }
    
    return {computeNeeded, graphicsNeeded};
}

void FrameGraph::beginCommandBuffers(bool useCompute, bool useGraphics, uint32_t frameIndex) {
    const auto& vk = context_->getLoader();
    
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    
    if (useCompute) {
        VkCommandBuffer computeCmd = queueManager_->getComputeCommandBuffer(frameIndex);
        vk.vkBeginCommandBuffer(computeCmd, &beginInfo);
    }
    if (useGraphics) {
        VkCommandBuffer graphicsCmd = queueManager_->getGraphicsCommandBuffer(frameIndex);
        vk.vkBeginCommandBuffer(graphicsCmd, &beginInfo);
    }
}

void FrameGraph::endCommandBuffers(bool useCompute, bool useGraphics, uint32_t frameIndex) {
    const auto& vk = context_->getLoader();
    
    if (useCompute) {
        VkCommandBuffer computeCmd = queueManager_->getComputeCommandBuffer(frameIndex);
        vk.vkEndCommandBuffer(computeCmd);
    }
    if (useGraphics) {
        VkCommandBuffer graphicsCmd = queueManager_->getGraphicsCommandBuffer(frameIndex);
        vk.vkEndCommandBuffer(graphicsCmd);
    }
}

void FrameGraph::executeNodesInOrder(uint32_t frameIndex, float time, float deltaTime, uint32_t globalFrame, bool& computeExecuted) {
    VkCommandBuffer currentComputeCmd = queueManager_->getComputeCommandBuffer(frameIndex);
    VkCommandBuffer currentGraphicsCmd = queueManager_->getGraphicsCommandBuffer(frameIndex);
    
    for (auto nodeId : executionOrder_) {
        auto it = nodes_.find(nodeId);
        if (it == nodes_.end()) continue;
        
        auto& node = it->second;
        
        // Prepare frame with new standardized lifecycle
        FrameContext frameContext{frameIndex, time, deltaTime, globalFrame};
        node->prepareFrame(frameContext);
        
        VkCommandBuffer cmdBuffer = node->needsComputeQueue() ? currentComputeCmd : currentGraphicsCmd;
        
        // Insert barriers for this node on the same queue it will execute on
        barrierManager_.insertBarriersForNode(nodeId, cmdBuffer, computeExecuted, node->needsGraphicsQueue());
        
        if (node->needsComputeQueue()) {
            computeExecuted = true;
        }
        
        node->execute(cmdBuffer, *this);
        
        // Release frame with new standardized lifecycle
        node->releaseFrame(frameIndex);
    }
}

bool FrameGraph::executeWithTimeoutMonitoring(uint32_t frameIndex, float time, float deltaTime, uint32_t globalFrame, bool& computeExecuted) {
    VkCommandBuffer currentComputeCmd = queueManager_->getComputeCommandBuffer(frameIndex);
    VkCommandBuffer currentGraphicsCmd = queueManager_->getGraphicsCommandBuffer(frameIndex);
    
    for (auto nodeId : executionOrder_) {
        auto it = nodes_.find(nodeId);
        if (it == nodes_.end()) continue;
        
        auto& node = it->second;
        
        // Check GPU health before executing
        if (!timeoutDetector_->isGPUHealthy()) {
            std::cerr << "[FrameGraph] GPU unhealthy, aborting execution" << std::endl;
            return false;
        }
        
        // Prepare frame with new standardized lifecycle
        FrameContext frameContext{frameIndex, time, deltaTime, globalFrame};
        node->prepareFrame(frameContext);
        
        VkCommandBuffer cmdBuffer = node->needsComputeQueue() ? currentComputeCmd : currentGraphicsCmd;
        
        // Insert barriers for this node on the same queue it will execute on
        barrierManager_.insertBarriersForNode(nodeId, cmdBuffer, computeExecuted, node->needsGraphicsQueue());
        
        // Begin timeout monitoring for this node
        std::string nodeName = node->getName() + "_FrameGraph";
        if (node->needsComputeQueue()) {
            timeoutDetector_->beginComputeDispatch(nodeName.c_str(), 1); // Generic workgroup count
            computeExecuted = true;
        }
        
        // Execute the node
        node->execute(cmdBuffer, *this);
        
        // End timeout monitoring
        if (node->needsComputeQueue()) {
            timeoutDetector_->endComputeDispatch();
            
            // Check if we need to apply recovery recommendations
            auto recommendation = timeoutDetector_->getRecoveryRecommendation();
            if (recommendation.shouldReduceWorkload) {
                std::cout << "[FrameGraph] Applying timeout recovery recommendations" << std::endl;
                // Future: Could implement workload reduction at frame graph level
            }
        }
        
        // Final health check after node execution
        if (!timeoutDetector_->isGPUHealthy()) {
            std::cerr << "[FrameGraph] GPU became unhealthy after node execution" << std::endl;
            return false;
        }
        
        // Release frame with new standardized lifecycle
        node->releaseFrame(frameIndex);
    }
    
    return true;
}

void FrameGraph::handleExecutionTimeout() {
    std::cerr << "[FrameGraph] Execution timeout detected - frame graph execution aborted" << std::endl;
    
    if (timeoutDetector_) {
        auto stats = timeoutDetector_->getStats();
        std::cerr << "[FrameGraph] Timeout stats - Average: " << stats.averageDispatchTimeMs 
                  << "ms, Peak: " << stats.peakDispatchTimeMs << "ms, Warnings: " << stats.warningCount << std::endl;
    }
    
    // Could implement additional recovery strategies here:
    // - Mark nodes for reduced execution
    // - Schedule recompilation with simpler graph
    // - Request external systems to reduce entity count
}

