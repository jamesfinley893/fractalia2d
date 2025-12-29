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
    FrameGraphTypes::ResourceId getControlParamsBufferId() const { return controlParamsBufferId; }
    FrameGraphTypes::ResourceId getSpatialNextBufferId() const { return spatialNextBufferId; }
    FrameGraphTypes::ResourceId getParticleVelocityBufferId() const { return particleVelocityBufferId; }
    FrameGraphTypes::ResourceId getParticleInvMassBufferId() const { return particleInvMassBufferId; }
    FrameGraphTypes::ResourceId getParticleBodyBufferId() const { return particleBodyBufferId; }
    FrameGraphTypes::ResourceId getBodyDataBufferId() const { return bodyDataBufferId; }
    FrameGraphTypes::ResourceId getBodyParamsBufferId() const { return bodyParamsBufferId; }
    FrameGraphTypes::ResourceId getDistanceConstraintBufferId() const { return distanceConstraintBufferId; }
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
    FrameGraphTypes::ResourceId controlParamsBufferId = 0;
    FrameGraphTypes::ResourceId spatialNextBufferId = 0;
    FrameGraphTypes::ResourceId particleVelocityBufferId = 0;
    FrameGraphTypes::ResourceId particleInvMassBufferId = 0;
    FrameGraphTypes::ResourceId particleBodyBufferId = 0;
    FrameGraphTypes::ResourceId bodyDataBufferId = 0;
    FrameGraphTypes::ResourceId bodyParamsBufferId = 0;
    FrameGraphTypes::ResourceId distanceConstraintBufferId = 0;
    FrameGraphTypes::ResourceId positionBufferId = 0;
    FrameGraphTypes::ResourceId currentPositionBufferId = 0;
    FrameGraphTypes::ResourceId targetPositionBufferId = 0;
};
