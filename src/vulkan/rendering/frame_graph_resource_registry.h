#pragma once

#include "frame_graph.h"

// Forward declarations
class FrameGraph;
class GPUEntityManager;

class FrameGraphResourceRegistry {
public:
    FrameGraphResourceRegistry();
    ~FrameGraphResourceRegistry();

    bool initialize(FrameGraph* frameGraph, GPUEntityManager* gpuEntityManager);
    void cleanup();

    // Import all entity-related resources into frame graph
    bool importEntityResources();

    // Getters for resource IDs
    FrameGraphTypes::ResourceId getVelocityBufferId() const { return velocityBufferId; }
    FrameGraphTypes::ResourceId getMovementParamsBufferId() const { return movementParamsBufferId; }
    FrameGraphTypes::ResourceId getRuntimeStateBufferId() const { return runtimeStateBufferId; }
    FrameGraphTypes::ResourceId getColorBufferId() const { return colorBufferId; }
    FrameGraphTypes::ResourceId getModelMatrixBufferId() const { return modelMatrixBufferId; }
    FrameGraphTypes::ResourceId getSpatialMapBufferId() const { return spatialMapBufferId; }
    FrameGraphTypes::ResourceId getPositionBufferId() const { return positionBufferId; }
    FrameGraphTypes::ResourceId getCurrentPositionBufferId() const { return currentPositionBufferId; }
    FrameGraphTypes::ResourceId getTargetPositionBufferId() const { return targetPositionBufferId; }

private:
    // Dependencies
    FrameGraph* frameGraph = nullptr;
    GPUEntityManager* gpuEntityManager = nullptr;

    // Resource IDs
    FrameGraphTypes::ResourceId velocityBufferId = 0;
    FrameGraphTypes::ResourceId movementParamsBufferId = 0;
    FrameGraphTypes::ResourceId runtimeStateBufferId = 0;
    FrameGraphTypes::ResourceId colorBufferId = 0;
    FrameGraphTypes::ResourceId modelMatrixBufferId = 0;
    FrameGraphTypes::ResourceId spatialMapBufferId = 0;
    FrameGraphTypes::ResourceId positionBufferId = 0;
    FrameGraphTypes::ResourceId currentPositionBufferId = 0;
    FrameGraphTypes::ResourceId targetPositionBufferId = 0;
};
