#pragma once

#include "buffer_base.h"
#include <glm/glm.hpp>
#include <cstdint>

struct GPUDistanceConstraint {
    uint32_t a;
    uint32_t b;
    float restLength;
    float stiffness;
};

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

// SINGLE responsibility: spatial linked-list next indices
class SpatialNextBuffer : public BufferBase {
public:
    using BufferBase::initialize; // Bring base class initialize into scope

    bool initialize(const VulkanContext& context, ResourceCoordinator* resourceCoordinator, uint32_t maxEntities) {
        return BufferBase::initialize(context, resourceCoordinator, maxEntities, sizeof(uint32_t), 0);
    }

protected:
    const char* getBufferTypeName() const override { return "SpatialNext"; }
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

// SINGLE responsibility: particle velocity data management (per particle)
class ParticleVelocityBuffer : public BufferBase {
public:
    using BufferBase::initialize;

    bool initialize(const VulkanContext& context, ResourceCoordinator* resourceCoordinator, uint32_t maxParticles) {
        return BufferBase::initialize(context, resourceCoordinator, maxParticles, sizeof(glm::vec4), 0);
    }

protected:
    const char* getBufferTypeName() const override { return "ParticleVelocity"; }
};

// SINGLE responsibility: particle inverse mass data management (per particle)
class ParticleInvMassBuffer : public BufferBase {
public:
    using BufferBase::initialize;

    bool initialize(const VulkanContext& context, ResourceCoordinator* resourceCoordinator, uint32_t maxParticles) {
        return BufferBase::initialize(context, resourceCoordinator, maxParticles, sizeof(float), 0);
    }

protected:
    const char* getBufferTypeName() const override { return "ParticleInvMass"; }
};

// SINGLE responsibility: particle→body mapping data management (per particle)
class ParticleBodyBuffer : public BufferBase {
public:
    using BufferBase::initialize;

    bool initialize(const VulkanContext& context, ResourceCoordinator* resourceCoordinator, uint32_t maxParticles) {
        return BufferBase::initialize(context, resourceCoordinator, maxParticles, sizeof(uint32_t), 0);
    }

protected:
    const char* getBufferTypeName() const override { return "ParticleBody"; }
};

// SINGLE responsibility: body metadata (offsets/counts)
class BodyDataBuffer : public BufferBase {
public:
    using BufferBase::initialize;

    bool initialize(const VulkanContext& context, ResourceCoordinator* resourceCoordinator, uint32_t maxBodies) {
        return BufferBase::initialize(context, resourceCoordinator, maxBodies, sizeof(glm::uvec4), 0);
    }

protected:
    const char* getBufferTypeName() const override { return "BodyData"; }
};

// SINGLE responsibility: body simulation parameters
class BodyParamsBuffer : public BufferBase {
public:
    using BufferBase::initialize;

    bool initialize(const VulkanContext& context, ResourceCoordinator* resourceCoordinator, uint32_t maxBodies) {
        return BufferBase::initialize(context, resourceCoordinator, maxBodies, sizeof(glm::vec4), 0);
    }

protected:
    const char* getBufferTypeName() const override { return "BodyParams"; }
};

// SINGLE responsibility: distance constraint data management
class DistanceConstraintBuffer : public BufferBase {
public:
    using BufferBase::initialize;

    bool initialize(const VulkanContext& context, ResourceCoordinator* resourceCoordinator, uint32_t maxConstraints) {
        return BufferBase::initialize(context, resourceCoordinator, maxConstraints, sizeof(GPUDistanceConstraint), 0);
    }

protected:
    const char* getBufferTypeName() const override { return "DistanceConstraint"; }
};
