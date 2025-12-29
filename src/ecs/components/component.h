#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <cstdint>
#include <cstring>

// Transform component - consolidates position/rotation for better cache locality
struct Transform {
    glm::vec3 position{0.0f, 0.0f, 0.0f};
    glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f}; // w, x, y, z (identity quaternion)
    glm::vec3 scale{1.0f, 1.0f, 1.0f};
    
    // Cached transform matrix - updated when dirty
    mutable glm::mat4 matrix{1.0f};
    mutable bool dirty{true};
    
    const glm::mat4& getMatrix() const {
        if (dirty) {
            glm::mat4 translationMatrix = glm::translate(glm::mat4(1.0f), position);
            glm::mat4 rotationMatrix = glm::mat4_cast(rotation);
            glm::mat4 scaleMatrix = glm::scale(glm::mat4(1.0f), scale);
            matrix = translationMatrix * rotationMatrix * scaleMatrix;
            dirty = false;
        }
        return matrix;
    }
    
    void setPosition(const glm::vec3& pos) { position = pos; dirty = true; }
    void setRotation(const glm::quat& rot) { rotation = rot; dirty = true; }
    void setRotation(const glm::vec3& eulerAngles) { 
        rotation = glm::quat(eulerAngles);
        dirty = true; 
    }
    void setScale(const glm::vec3& scl) { scale = scl; dirty = true; }
};

// Velocity component for physics
struct Velocity {
    glm::vec3 linear{0.0f, 0.0f, 0.0f};
    glm::vec3 angular{0.0f, 0.0f, 0.0f};
};

// Render component - optimized for batch rendering
struct Renderable {
    glm::vec4 color{1.0f, 1.0f, 1.0f, 1.0f};
    uint32_t layer{0}; // For depth sorting
    bool visible{true};
    
    // Transform matrix for GPU upload
    glm::mat4 modelMatrix{1.0f};
    
    // Change tracking for optimization
    mutable uint32_t version{0};
    mutable bool dirty{true};
    void markDirty() const { ++version; dirty = true; }
};

// Lifetime management component
struct Lifetime {
    float maxAge{-1.0f}; // -1 = infinite
    float currentAge{0.0f};
    bool autoDestroy{false};
};

// Physics bounds for collision detection
struct Bounds {
    glm::vec3 min{-0.5f, -0.5f, -0.5f};
    glm::vec3 max{0.5f, 0.5f, 0.5f};
    bool dynamic{true}; // Updates with transform changes
};

// Movement types matching GPU compute shader
enum class MovementType {
    RandomWalk = 0  // GPU-driven random walk from center point (simplified to single type)
};

struct MovementPattern {
    MovementType type{MovementType::RandomWalk};
    MovementType movementType{MovementType::RandomWalk}; // For compatibility
    
    // Basic movement parameters
    float amplitude{1.0f};      // Size/scale of pattern
    float frequency{1.0f};      // Speed/frequency of oscillation
    float phase{0.0f};          // Phase offset for variation
    float timeOffset{0.0f};     // Individual timing offset
    
    // Center point for movement
    glm::vec3 center{0.0f};
    
    // Runtime state
    mutable float totalTime{0.0f};
    mutable float currentTime{0.0f};
    mutable bool initialized{false};
};

// Player control marker with tunable speed
struct Player {
    float speed{1.5f};
};

// Per-player control state driven by CPU input
struct PlayerControl {
    glm::vec2 desiredVelocity{0.0f};
    bool active{false};
    float renderScale{1.0f};
};

// GPU index mapping for per-entity GPU updates
struct GPUIndex {
    uint32_t index{0};
};


// Input system components for ECS-based input handling
struct KeyboardInput {
    static constexpr size_t MAX_KEYS = 512;
    
    // Current frame key states
    bool keys[MAX_KEYS] = {false};
    bool keysPressed[MAX_KEYS] = {false};   // Key pressed this frame
    bool keysReleased[MAX_KEYS] = {false};  // Key released this frame
    
    // Modifier states
    bool shift = false;
    bool ctrl = false;
    bool alt = false;
    
    // Helper methods
    bool isKeyDown(int scancode) const {
        return scancode >= 0 && static_cast<size_t>(scancode) < MAX_KEYS && keys[scancode];
    }
    
    bool isKeyPressed(int scancode) const {
        return scancode >= 0 && static_cast<size_t>(scancode) < MAX_KEYS && keysPressed[scancode];
    }
    
    bool isKeyReleased(int scancode) const {
        return scancode >= 0 && static_cast<size_t>(scancode) < MAX_KEYS && keysReleased[scancode];
    }
    
    void clearFrameStates() {
        std::memset(keysPressed, false, sizeof(keysPressed));
        std::memset(keysReleased, false, sizeof(keysReleased));
    }
};

struct MouseInput {
    // Mouse position
    glm::vec2 position{0.0f, 0.0f};
    glm::vec2 deltaPosition{0.0f, 0.0f};
    glm::vec2 worldPosition{0.0f, 0.0f}; // Position in world coordinates
    
    // Mouse buttons (left, middle, right, x1, x2)
    static constexpr size_t MAX_BUTTONS = 8;
    bool buttons[MAX_BUTTONS] = {false};
    bool buttonsPressed[MAX_BUTTONS] = {false};
    bool buttonsReleased[MAX_BUTTONS] = {false};
    
    // Mouse wheel
    glm::vec2 wheelDelta{0.0f, 0.0f};
    
    // Mouse state
    bool isInWindow = true;
    bool isRelativeMode = false;
    
    // Helper methods
    bool isButtonDown(int button) const {
        return button >= 0 && static_cast<size_t>(button) < MAX_BUTTONS && buttons[button];
    }
    
    bool isButtonPressed(int button) const {
        return button >= 0 && static_cast<size_t>(button) < MAX_BUTTONS && buttonsPressed[button];
    }
    
    bool isButtonReleased(int button) const {
        return button >= 0 && static_cast<size_t>(button) < MAX_BUTTONS && buttonsReleased[button];
    }
    
    void clearFrameStates() {
        std::memset(buttonsPressed, false, sizeof(buttonsPressed));
        std::memset(buttonsReleased, false, sizeof(buttonsReleased));
        wheelDelta = glm::vec2(0.0f, 0.0f);
        deltaPosition = glm::vec2(0.0f, 0.0f);
    }
};

struct InputEvents {
    static constexpr size_t MAX_EVENTS = 64;
    
    struct Event {
        enum Type {
            QUIT,
            KEY_DOWN,
            KEY_UP,
            MOUSE_BUTTON_DOWN,
            MOUSE_BUTTON_UP,
            MOUSE_MOTION,
            MOUSE_WHEEL,
            WINDOW_RESIZE
        };
        
        Type type;
        union {
            struct { int key; bool repeat; } keyEvent;
            struct { int button; glm::vec2 position; } mouseButtonEvent;
            struct { glm::vec2 position; glm::vec2 delta; } mouseMotionEvent;
            struct { glm::vec2 delta; } mouseWheelEvent;
            struct { int width; int height; } windowResizeEvent;
        };
    };
    
    Event events[MAX_EVENTS];
    size_t eventCount = 0;
    
    void addEvent(const Event& event) {
        if (eventCount < MAX_EVENTS) {
            events[eventCount++] = event;
        }
    }
    
    void clear() {
        eventCount = 0;
    }
};

// Singleton component for global input state
struct InputState {
    bool quit = false;
    float deltaTime = 0.0f;
    uint32_t frameNumber = 0;
    
    // Input processing flags
    bool processKeyboard = true;
    bool processMouse = true;
    bool consumeEvents = true; // Whether to consume SDL events
};


// Tag components for input-responsive entities
struct KeyboardControlled {};   // Entity responds to keyboard input
struct MouseControlled {};      // Entity responds to mouse input
struct InputResponsive {};      // Entity responds to any input


// Tag components for efficient filtering
struct Static {}; // Non-moving entities
struct Dynamic {}; // Moving entities  
struct Pooled {}; // Can be recycled

// Application state management component (singleton)
struct ApplicationState {
    bool running = true;
    bool requestQuit = false;
    float globalDeltaTime = 0.0f;
    uint64_t frameCount = 0;
};

// GPU synchronization marker components
struct GPUUploadPending {};        // Entity needs GPU upload
struct GPUUploadComplete {};       // Entity has been uploaded to GPU
struct GPUEntitySync {             // Singleton component for GPU sync operations
    bool needsUpload = false;
    uint32_t pendingCount = 0;
    float deltaTime = 0.0f;
};

// Control system marker components  
struct InputProcessed {};          // Marker for input processing completion
struct ControlsProcessed {};       // Marker for control processing completion

// Backward compatibility aliases
using Position = Transform;
using Color = Renderable;
using Shape = Renderable;
