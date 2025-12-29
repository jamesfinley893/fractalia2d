#pragma once

#include "../core/service_locator.h"
#include "../components/component.h"
#include <flecs.h>
#include <glm/glm.hpp>
#include <functional>
#include <string>
#include <unordered_map>

// Forward declarations
class VulkanRenderer;
class EntityFactory;
class InputService;
class CameraService;
class RenderingService;

// Control action types
enum class ControlActionType {
    TOGGLE_MOVEMENT,
    CREATE_ENTITY,
    CREATE_SWARM,
    DEBUG_ENTITY,
    PERFORMANCE_STATS,
    GRAPHICS_TESTS,
    CAMERA_CONTROL,
    RENDERING_DEBUG
};

// Control action definition
struct ControlAction {
    ControlActionType type;
    std::string name;
    std::string description;
    std::function<void()> execute;
    bool enabled = true;
    float cooldown = 0.0f;
    float lastExecuted = 0.0f;
};

// Control state for game logic
struct ControlState {
    int currentMovementType = 0;
    glm::vec2 entityCreationPos{0.0f, 0.0f};
    bool debugMode = false;
    bool wireframeMode = false;
    bool performanceMonitoring = true;
    
    // Request flags
    bool requestEntityCreation = false;
    bool requestSwarmCreation = false;
    bool requestPerformanceStats = false;
    bool requestGraphicsTests = false;
    
    void resetRequestFlags() {
        requestEntityCreation = false;
        requestSwarmCreation = false;
        requestPerformanceStats = false;
        requestGraphicsTests = false;
    }
};

// Control service - centralized control logic using services
class GameControlService {
public:
    DECLARE_SERVICE(GameControlService);
    
    GameControlService();
    ~GameControlService();
    
    // Initialization and cleanup
    bool initialize(flecs::world& world, VulkanRenderer* renderer, EntityFactory* entityFactory);
    void cleanup();
    
    // Frame processing
    void processFrame(float deltaTime);
    void handleInput();
    void executeActions();
    
    // Action management
    void registerAction(const ControlAction& action);
    void unregisterAction(const std::string& actionName);
    void executeAction(const std::string& actionName);
    bool isActionAvailable(const std::string& actionName) const;
    void setActionEnabled(const std::string& actionName, bool enabled);
    
    // Control state management
    ControlState& getControlState() { return controlState; }
    const ControlState& getControlState() const { return controlState; }
    void setControlState(const ControlState& state) { controlState = state; }
    
    // Game logic actions
    void toggleMovementType();
    void createEntity(const glm::vec2& position);
    void createSwarm(size_t count, const glm::vec3& center, float radius);
    void debugEntityAtPosition(const glm::vec2& worldPos);
    void showPerformanceStats();
    void runGraphicsTests();
    void toggleDebugMode();
    void toggleWireframeMode();
    
    // Camera control integration
    void handleCameraControls();
    void resetCamera();
    void focusCameraOnEntities();
    
    // Configuration
    void setEntityCreationCooldown(float cooldown) { entityCreationCooldown = cooldown; }
    float getEntityCreationCooldown() const { return entityCreationCooldown; }
    void setSwarmCreationCooldown(float cooldown) { swarmCreationCooldown = cooldown; }
    float getSwarmCreationCooldown() const { return swarmCreationCooldown; }
    
    // Statistics and monitoring
    void printControlStats() const;
    void logControlState() const;
    void printControlInstructions() const;

private:
    // Core data
    flecs::world* world = nullptr;
    VulkanRenderer* renderer = nullptr;
    EntityFactory* entityFactory = nullptr;
    bool initialized = false;
    
    // Service dependencies
    InputService* inputService = nullptr;
    CameraService* cameraService = nullptr;
    RenderingService* renderingService = nullptr;
    
    // Control system state
    ControlState controlState;
    std::unordered_map<std::string, ControlAction> actions;
    
    // Timing and cooldowns
    float deltaTime = 0.0f;
    float entityCreationCooldown = 0.01f; // 10ms - minimal delay for immediate response
    float swarmCreationCooldown = 0.05f; // 50ms - fast swarm creation
    
    // Internal methods
    void initializeDefaultActions();
    void updateActionCooldowns();
    bool checkActionCooldown(const std::string& actionName) const;
    void executePendingRequests();
    void updatePlayerMovement();
    
    // Service integration helpers
    void integrateWithInputService();
    void integrateWithCameraService();
    void integrateWithRenderingService();
    
    // Action implementations
    void actionToggleMovement();
    void actionCreateEntity();
    void actionCreateSwarm();
    void actionDebugEntity();
    void actionShowStats();
    void actionGraphicsTests();
    void actionToggleDebug();
    void actionCameraReset();
    void actionCameraFocus();
};

