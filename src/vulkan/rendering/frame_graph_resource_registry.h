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
    FrameGraphTypes::ResourceId getNodeVelocityBufferId() const { return nodeVelocityBufferId; }
    FrameGraphTypes::ResourceId getNodeInvMassBufferId() const { return nodeInvMassBufferId; }
    FrameGraphTypes::ResourceId getBodyDataBufferId() const { return bodyDataBufferId; }
    FrameGraphTypes::ResourceId getBodyParamsBufferId() const { return bodyParamsBufferId; }
    FrameGraphTypes::ResourceId getTriangleRestBufferId() const { return triangleRestBufferId; }
    FrameGraphTypes::ResourceId getTriangleAreaBufferId() const { return triangleAreaBufferId; }
    FrameGraphTypes::ResourceId getNodeForceBufferId() const { return nodeForceBufferId; }
    FrameGraphTypes::ResourceId getNodeRestBufferId() const { return nodeRestBufferId; }
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
    FrameGraphTypes::ResourceId nodeVelocityBufferId = 0;
    FrameGraphTypes::ResourceId nodeInvMassBufferId = 0;
    FrameGraphTypes::ResourceId bodyDataBufferId = 0;
    FrameGraphTypes::ResourceId bodyParamsBufferId = 0;
    FrameGraphTypes::ResourceId triangleRestBufferId = 0;
    FrameGraphTypes::ResourceId triangleAreaBufferId = 0;
    FrameGraphTypes::ResourceId nodeForceBufferId = 0;
    FrameGraphTypes::ResourceId nodeRestBufferId = 0;
    FrameGraphTypes::ResourceId positionBufferId = 0;
    FrameGraphTypes::ResourceId currentPositionBufferId = 0;
    FrameGraphTypes::ResourceId targetPositionBufferId = 0;
};
