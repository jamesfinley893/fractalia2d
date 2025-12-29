#pragma once

#include <cstdint>

namespace FrameGraphTypes {
    using ResourceId = uint32_t;
    using NodeId = uint32_t;
    
    constexpr ResourceId INVALID_RESOURCE = 0;
    constexpr NodeId INVALID_NODE = 0;
}

// Resource access patterns for dependency tracking
enum class ResourceAccess {
    Read,
    Write,
    ReadWrite
};

// Pipeline stages for synchronization
enum class PipelineStage {
    ComputeShader,
    VertexShader,
    FragmentShader,
    ColorAttachment,
    DepthAttachment,
    Transfer
};

// Resource classification for allocation strategies
enum class ResourceCriticality {
    Critical,    // Must be device local, fail fast if not possible
    Important,   // Prefer device local, allow limited fallback
    Flexible     // Accept any memory type for allocation success
};

// Resource dependency descriptor
struct ResourceDependency {
    FrameGraphTypes::ResourceId resourceId;
    ResourceAccess access;
    PipelineStage stage;
};

// Frame-scoped context passed to nodes
struct FrameContext {
    uint32_t frameIndex = 0;
    float time = 0.0f;
    float deltaTime = 0.0f;
    uint32_t globalFrame = 0;
};
