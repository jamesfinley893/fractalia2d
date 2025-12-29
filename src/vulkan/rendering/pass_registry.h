#pragma once

#include "frame_graph.h"
#include <functional>
#include <string>
#include <vector>

class VulkanContext;
class VulkanSwapchain;
class ResourceCoordinator;
class GPUEntityManager;
class PipelineSystemManager;

struct RenderPassContext {
    VulkanContext* context = nullptr;
    VulkanSwapchain* swapchain = nullptr;
    PipelineSystemManager* pipelineSystem = nullptr;
    ResourceCoordinator* resourceCoordinator = nullptr;
    GPUEntityManager* gpuEntityManager = nullptr;
    FrameGraph* frameGraph = nullptr;
};

struct RenderPassDefinition {
    std::string name;
    std::function<void(FrameGraph&, const RenderPassContext&)> registerPass;
};

class RenderPassRegistry {
public:
    void registerPass(const std::string& name,
                      std::function<void(FrameGraph&, const RenderPassContext&)> registerFn);

    void build(FrameGraph& frameGraph, const RenderPassContext& context) const;

private:
    std::vector<RenderPassDefinition> passes;
};
