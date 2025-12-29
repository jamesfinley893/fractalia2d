#include "frame_graph_resource_registry.h"
#include "frame_graph.h"
#include "../../ecs/gpu/gpu_entity_manager.h"
#include <iostream>

FrameGraphResourceRegistry::FrameGraphResourceRegistry() {
}

FrameGraphResourceRegistry::~FrameGraphResourceRegistry() {
    cleanup();
}

bool FrameGraphResourceRegistry::initialize(FrameGraph* frameGraph, GPUEntityManager* gpuEntityManager) {
    this->frameGraph = frameGraph;
    this->gpuEntityManager = gpuEntityManager;
    return true;
}

void FrameGraphResourceRegistry::cleanup() {
    // Dependencies are managed externally
}

bool FrameGraphResourceRegistry::importEntityResources() {
    if (!frameGraph || !gpuEntityManager) {
        std::cerr << "FrameGraphResourceRegistry: Invalid dependencies" << std::endl;
        return false;
    }

    // Import SoA buffers
    velocityBufferId = frameGraph->importExternalBuffer(
        "VelocityBuffer",
        gpuEntityManager->getVelocityBuffer(),
        gpuEntityManager->getVelocityBufferSize(),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );

    movementParamsBufferId = frameGraph->importExternalBuffer(
        "MovementParamsBuffer",
        gpuEntityManager->getMovementParamsBuffer(),
        gpuEntityManager->getMovementParamsBufferSize(),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );

    runtimeStateBufferId = frameGraph->importExternalBuffer(
        "RuntimeStateBuffer",
        gpuEntityManager->getRuntimeStateBuffer(),
        gpuEntityManager->getRuntimeStateBufferSize(),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );

    colorBufferId = frameGraph->importExternalBuffer(
        "ColorBuffer",
        gpuEntityManager->getColorBuffer(),
        gpuEntityManager->getColorBufferSize(),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );

    modelMatrixBufferId = frameGraph->importExternalBuffer(
        "ModelMatrixBuffer",
        gpuEntityManager->getModelMatrixBuffer(),
        gpuEntityManager->getModelMatrixBufferSize(),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );

    spatialMapBufferId = frameGraph->importExternalBuffer(
        "SpatialMapBuffer",
        gpuEntityManager->getSpatialMapBuffer(),
        gpuEntityManager->getSpatialMapBufferSize(),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );

    controlParamsBufferId = frameGraph->importExternalBuffer(
        "ControlParamsBuffer",
        gpuEntityManager->getControlParamsBuffer(),
        gpuEntityManager->getControlParamsBufferSize(),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );

    // Import position buffer
    positionBufferId = frameGraph->importExternalBuffer(
        "PositionBuffer",
        gpuEntityManager->getPositionBuffer(),
        gpuEntityManager->getPositionBufferSize(),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
    );

    // Import current position buffer
    currentPositionBufferId = frameGraph->importExternalBuffer(
        "CurrentPositionBuffer",
        gpuEntityManager->getCurrentPositionBuffer(),
        gpuEntityManager->getPositionBufferSize(),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );

    // Import target position buffer
    targetPositionBufferId = frameGraph->importExternalBuffer(
        "TargetPositionBuffer",
        gpuEntityManager->getTargetPositionBuffer(),
        gpuEntityManager->getPositionBufferSize(),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );

    return true;
}
