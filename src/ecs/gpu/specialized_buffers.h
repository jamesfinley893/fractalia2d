#pragma once

#include "buffer_base.h"
#include <glm/glm.hpp>

/**
 * Specialized buffer classes following Single Responsibility Principle
 * Each class manages exactly one type of entity data
 */

// SINGLE responsibility: velocity data management
class VelocityBuffer : public BufferBase {
public:
    using BufferBase::initialize; // Bring base class initialize into scope
    
    bool initialize(const VulkanContext& context, ResourceCoordinator* resourceCoordinator, uint32_t maxEntities) {
        return BufferBase::initialize(context, resourceCoordinator, maxEntities, sizeof(glm::vec4), 0);
    }
    
protected:
    const char* getBufferTypeName() const override { return "Velocity"; }
};

// SINGLE responsibility: movement parameters management
class MovementParamsBuffer : public BufferBase {
public:
    using BufferBase::initialize; // Bring base class initialize into scope
    
    bool initialize(const VulkanContext& context, ResourceCoordinator* resourceCoordinator, uint32_t maxEntities) {
        return BufferBase::initialize(context, resourceCoordinator, maxEntities, sizeof(glm::vec4), 0);
    }
    
protected:
    const char* getBufferTypeName() const override { return "MovementParams"; }
};

// SINGLE responsibility: runtime state management
class RuntimeStateBuffer : public BufferBase {
public:
    using BufferBase::initialize; // Bring base class initialize into scope
    
    bool initialize(const VulkanContext& context, ResourceCoordinator* resourceCoordinator, uint32_t maxEntities) {
        return BufferBase::initialize(context, resourceCoordinator, maxEntities, sizeof(glm::vec4), 0);
    }
    
protected:
    const char* getBufferTypeName() const override { return "RuntimeState"; }
};

// SINGLE responsibility: color data management
class ColorBuffer : public BufferBase {
public:
    using BufferBase::initialize; // Bring base class initialize into scope
    
    bool initialize(const VulkanContext& context, ResourceCoordinator* resourceCoordinator, uint32_t maxEntities) {
        return BufferBase::initialize(context, resourceCoordinator, maxEntities, sizeof(glm::vec4), 0);
    }
    
protected:
    const char* getBufferTypeName() const override { return "Color"; }
};

// SINGLE responsibility: model matrix management
class ModelMatrixBuffer : public BufferBase {
public:
    using BufferBase::initialize; // Bring base class initialize into scope
    
    bool initialize(const VulkanContext& context, ResourceCoordinator* resourceCoordinator, uint32_t maxEntities) {
        return BufferBase::initialize(context, resourceCoordinator, maxEntities, sizeof(glm::mat4), 0);
    }
    
protected:
    const char* getBufferTypeName() const override { return "ModelMatrix"; }
};

// SINGLE responsibility: position data management
class PositionBuffer : public BufferBase {
public:
    using BufferBase::initialize; // Bring base class initialize into scope
    
    bool initialize(const VulkanContext& context, ResourceCoordinator* resourceCoordinator, uint32_t maxEntities) {
        return BufferBase::initialize(context, resourceCoordinator, maxEntities, sizeof(glm::vec4), 0);
    }
    
protected:
    const char* getBufferTypeName() const override { return "Position"; }
};

// SINGLE responsibility: spatial map data management
class SpatialMapBuffer : public BufferBase {
public:
    using BufferBase::initialize; // Bring base class initialize into scope
    
    bool initialize(const VulkanContext& context, ResourceCoordinator* resourceCoordinator, uint32_t gridSize = 4096) {
        // Spatial map uses uvec2 (8 bytes per cell) for linked list storage
        return BufferBase::initialize(context, resourceCoordinator, gridSize, sizeof(glm::uvec2), 0);
    }
    
protected:
    const char* getBufferTypeName() const override { return "SpatialMap"; }
};

// SINGLE responsibility: per-entity control parameters
class ControlParamsBuffer : public BufferBase {
public:
    using BufferBase::initialize; // Bring base class initialize into scope

    bool initialize(const VulkanContext& context, ResourceCoordinator* resourceCoordinator, uint32_t maxEntities) {
        return BufferBase::initialize(context, resourceCoordinator, maxEntities, sizeof(glm::vec4), 0);
    }

protected:
    const char* getBufferTypeName() const override { return "ControlParams"; }
};
