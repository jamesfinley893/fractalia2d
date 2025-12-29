#include "control_service.h"
#include "input_service.h"
#include "camera_service.h"
#include "rendering_service.h"
#include "../../vulkan_renderer.h"
#include "../core/entity_factory.h"
#include "../gpu/gpu_entity_manager.h"
#include "../utilities/debug.h"
#include "../utilities/profiler.h"
#include <iostream>
#include <algorithm>

// Service concept compliance verified by DECLARE_SERVICE macro

GameControlService::GameControlService() {
}

GameControlService::~GameControlService() {
    cleanup();
}

bool GameControlService::initialize(flecs::world& world, VulkanRenderer* renderer, EntityFactory* entityFactory) {
    if (initialized) {
        return true;
    }
    
    if (!renderer || !entityFactory) {
        std::cerr << "ControlService: Invalid renderer or entity factory provided" << std::endl;
        return false;
    }
    
    this->world = &world;
    this->renderer = renderer;
    this->entityFactory = entityFactory;
    
    // Get service dependencies from ServiceLocator
    auto& locator = ServiceLocator::instance();
    
    inputService = &locator.requireService<InputService>();
    cameraService = &locator.requireService<CameraService>();
    renderingService = &locator.requireService<RenderingService>();
    
    if (!inputService) {
        std::cerr << "ControlService: InputService not found in ServiceLocator" << std::endl;
        return false;
    }
    
    if (!cameraService) {
        std::cerr << "ControlService: CameraService not found in ServiceLocator" << std::endl;
        return false;
    }
    
    if (!renderingService) {
        std::cerr << "ControlService: RenderingService not found in ServiceLocator" << std::endl;
        return false;
    }
    
    // Initialize default actions and integrations
    initializeDefaultActions();
    integrateWithInputService();
    integrateWithCameraService();
    integrateWithRenderingService();
    
    initialized = true;
    // Print control instructions
    printControlInstructions();
    
    DEBUG_LOG("ControlService initialized successfully");
    return true;
}

void GameControlService::cleanup() {
    if (!initialized) {
        return;
    }
    
    actions.clear();
    controlState = ControlState{};
    
    world = nullptr;
    renderer = nullptr;
    entityFactory = nullptr;
    inputService = nullptr;
    cameraService = nullptr;
    renderingService = nullptr;
    
    initialized = false;
}

void GameControlService::processFrame(float deltaTime) {
    if (!initialized) {
        return;
    }
    
    this->deltaTime = deltaTime;
    
    // Debug: Check if service is running every 1800 frames (30 seconds)
    static int frameCount = 0;
    if (++frameCount % 1800 == 0) {
        std::cout << "GameControlService::processFrame - running (frame " << frameCount << ")" << std::endl;
    }
    
    // Update action cooldowns
    updateActionCooldowns();
    
    // Handle input for control actions
    handleInput();

    // Update player movement from input (GPU-driven simulation)
    updatePlayerMovement();
    
    // Execute any pending actions
    executeActions();
    
    // Execute pending requests from control state
    executePendingRequests();
}

void GameControlService::handleInput() {
    if (!inputService) {
        std::cout << "GameControlService::handleInput - inputService is null!" << std::endl;
        return;
    }
    
    // Debug: Show when input handling is called
    static int inputCallCount = 0;
    if (++inputCallCount % 1800 == 0) {
        std::cout << "GameControlService::handleInput - called " << inputCallCount << " times" << std::endl;
    }
    
    // Entity creation
    if (inputService->isActionJustPressed("create_entity")) {
        std::cout << "GameControlService: create_entity action triggered!" << std::endl;
        glm::vec2 mouseScreen = inputService->getMousePosition();
        controlState.entityCreationPos = inputService->getMouseWorldPosition();
        std::cout << "Mouse screen: (" << mouseScreen.x << ", " << mouseScreen.y << ") -> world: (" << controlState.entityCreationPos.x << ", " << controlState.entityCreationPos.y << ")" << std::endl;
        executeAction("create_entity");
    } else if (inputService->isActionActive("create_entity")) {
        std::cout << "GameControlService: create_entity is active but not just pressed" << std::endl;
    }
    
    // Entity debug info
    if (inputService->isActionJustPressed("debug_entity")) {
        std::cout << "GameControlService: debug_entity action triggered!" << std::endl;
        glm::vec2 mouseWorld = inputService->getMouseWorldPosition();
        std::cout << "Debug readback at world position: (" << mouseWorld.x << ", " << mouseWorld.y << ")" << std::endl;
        executeAction("debug_entity");
    }
    
    // Swarm creation  
    if (inputService->isActionJustPressed("create_swarm")) {
        std::cout << "GameControlService: create_swarm action triggered!" << std::endl;
        executeAction("create_swarm");
    } else if (inputService->isActionActive("create_swarm")) {
        std::cout << "GameControlService: create_swarm is active but not just pressed" << std::endl;
    }
    
    // Performance stats
    if (inputService->isActionJustPressed("show_stats")) {
        executeAction("show_stats");
    }
    
    // Graphics tests
    if (inputService->isActionJustPressed("graphics_tests")) {
        executeAction("graphics_tests");
    }
    
    // Debug mode toggle
    if (inputService->isActionJustPressed("toggle_debug")) {
        executeAction("toggle_debug");
    }
    
    // Camera controls
    if (inputService->isActionJustPressed("camera_reset")) {
        executeAction("camera_reset");
    }
    
    if (inputService->isActionJustPressed("camera_focus")) {
        executeAction("camera_focus");
    }
    
    // Handle continuous camera movement (analog actions)
    handleCameraControls();
}

void GameControlService::executeActions() {
    // Process any scheduled actions based on control state
    // This is where we can add more sophisticated action scheduling
}

void GameControlService::registerAction(const ControlAction& action) {
    actions[action.name] = action;
}

void GameControlService::unregisterAction(const std::string& actionName) {
    actions.erase(actionName);
}

void GameControlService::executeAction(const std::string& actionName) {
    auto it = actions.find(actionName);
    if (it != actions.end() && it->second.enabled) {
        if (checkActionCooldown(actionName)) {
            it->second.execute();
            it->second.lastExecuted = 0.0f; // Reset timer, will be updated by updateActionCooldowns
        }
    }
}

bool GameControlService::isActionAvailable(const std::string& actionName) const {
    auto it = actions.find(actionName);
    return it != actions.end() && it->second.enabled && checkActionCooldown(actionName);
}

void GameControlService::setActionEnabled(const std::string& actionName, bool enabled) {
    auto it = actions.find(actionName);
    if (it != actions.end()) {
        it->second.enabled = enabled;
    }
}

void GameControlService::initializeDefaultActions() {
    // Register default control actions
    registerAction({
        ControlActionType::CREATE_ENTITY,
        "create_entity",
        "Create entity at cursor position",
        [this]() { actionCreateEntity(); },
        true, entityCreationCooldown, 0.0f
    });
    
    registerAction({
        ControlActionType::CREATE_SWARM,
        "create_swarm",
        "Create entity swarm",
        [this]() { actionCreateSwarm(); },
        true, swarmCreationCooldown, 0.0f
    });
    
    registerAction({
        ControlActionType::DEBUG_ENTITY,
        "debug_entity",
        "Debug entity info at cursor position",
        [this]() { actionDebugEntity(); },
        true, 0.5f, 0.0f
    });
    
    registerAction({
        ControlActionType::PERFORMANCE_STATS,
        "show_stats",
        "Show performance statistics",
        [this]() { actionShowStats(); },
        true, 1.0f, 0.0f
    });
    
    registerAction({
        ControlActionType::GRAPHICS_TESTS,
        "graphics_tests",
        "Run graphics stress tests",
        [this]() { actionGraphicsTests(); },
        true, 2.0f, 0.0f
    });
    
    registerAction({
        ControlActionType::RENDERING_DEBUG,
        "toggle_debug",
        "Toggle debug rendering mode",
        [this]() { actionToggleDebug(); },
        true, 0.5f, 0.0f
    });
    
    registerAction({
        ControlActionType::CAMERA_CONTROL,
        "camera_reset",
        "Reset camera to default position",
        [this]() { actionCameraReset(); },
        true, 0.5f, 0.0f
    });
    
    registerAction({
        ControlActionType::CAMERA_CONTROL,
        "camera_focus",
        "Focus camera on entities",
        [this]() { actionCameraFocus(); },
        true, 0.5f, 0.0f
    });
}

void GameControlService::updateActionCooldowns() {
    for (auto& [name, action] : actions) {
        action.lastExecuted += deltaTime;
    }
}

bool GameControlService::checkActionCooldown(const std::string& actionName) const {
    auto it = actions.find(actionName);
    if (it != actions.end()) {
        return it->second.lastExecuted >= it->second.cooldown;
    }
    return false;
}

void GameControlService::executePendingRequests() {
    if (controlState.requestEntityCreation) {
        std::cout << "GameControlService::executePendingRequests() - Creating entity at (" << controlState.entityCreationPos.x << ", " << controlState.entityCreationPos.y << ")" << std::endl;
        createEntity(controlState.entityCreationPos);
    }
    
    if (controlState.requestSwarmCreation) {
        createSwarm(1000, glm::vec3(10.0f, 10.0f, 0.0f), 8.0f);
    }
    
    if (controlState.requestPerformanceStats) {
        showPerformanceStats();
    }
    
    if (controlState.requestGraphicsTests) {
        runGraphicsTests();
    }
    
    // Reset all request flags
    controlState.resetRequestFlags();
}

void GameControlService::integrateWithInputService() {
    if (!inputService) return;
    
    // Register control-specific input actions with the InputService
    inputService->registerAction({
        "create_entity",
        InputActionType::DIGITAL,
        "Create entity at mouse position",
        {InputBinding(InputBinding::InputType::MOUSE_BUTTON, SDL_BUTTON_LEFT)}
    });
    
    inputService->registerAction({
        "create_swarm",
        InputActionType::DIGITAL,
        "Create entity swarm",
        {InputBinding(InputBinding::InputType::KEYBOARD_KEY, SDL_SCANCODE_EQUALS)}
    });
    
    inputService->registerAction({
        "debug_entity",
        InputActionType::DIGITAL,
        "Debug entity info at mouse position",
        {InputBinding(InputBinding::InputType::MOUSE_BUTTON, SDL_BUTTON_RIGHT)}
    });
    
    inputService->registerAction({
        "show_stats",
        InputActionType::DIGITAL,
        "Show performance statistics",
        {InputBinding(InputBinding::InputType::KEYBOARD_KEY, SDL_SCANCODE_P)}
    });
    
    inputService->registerAction({
        "graphics_tests",
        InputActionType::DIGITAL,
        "Run graphics tests",
        {InputBinding(InputBinding::InputType::KEYBOARD_KEY, SDL_SCANCODE_T)}
    });
    
    inputService->registerAction({
        "toggle_debug",
        InputActionType::DIGITAL,
        "Toggle debug mode",
        {InputBinding(InputBinding::InputType::KEYBOARD_KEY, SDL_SCANCODE_F3)}
    });
    
    inputService->registerAction({
        "camera_reset",
        InputActionType::DIGITAL,
        "Reset camera",
        {InputBinding(InputBinding::InputType::KEYBOARD_KEY, SDL_SCANCODE_R)}
    });
    
    inputService->registerAction({
        "camera_focus",
        InputActionType::DIGITAL,
        "Focus camera on entities",
        {InputBinding(InputBinding::InputType::KEYBOARD_KEY, SDL_SCANCODE_F)}
    });
    
    // WASD camera movement controls - using DIGITAL for keyboard keys
    inputService->registerAction({
        "camera_move_forward",
        InputActionType::DIGITAL,
        "Move camera forward",
        {InputBinding(InputBinding::InputType::KEYBOARD_KEY, SDL_SCANCODE_W)}
    });
    
    inputService->registerAction({
        "camera_move_backward",
        InputActionType::DIGITAL,
        "Move camera backward",
        {InputBinding(InputBinding::InputType::KEYBOARD_KEY, SDL_SCANCODE_S)}
    });
    
    inputService->registerAction({
        "camera_move_left",
        InputActionType::DIGITAL,
        "Move camera left",
        {InputBinding(InputBinding::InputType::KEYBOARD_KEY, SDL_SCANCODE_A)}
    });
    
    inputService->registerAction({
        "camera_move_right",
        InputActionType::DIGITAL,
        "Move camera right",
        {InputBinding(InputBinding::InputType::KEYBOARD_KEY, SDL_SCANCODE_D)}
    });
    
    inputService->registerAction({
        "camera_zoom",
        InputActionType::ANALOG_1D,
        "Zoom camera with mouse wheel",
        {InputBinding(InputBinding::InputType::MOUSE_WHEEL_Y, 0)}
    });

    inputService->registerAction({
        "player_move_left",
        InputActionType::DIGITAL,
        "Move player left",
        {InputBinding(InputBinding::InputType::KEYBOARD_KEY, SDL_SCANCODE_LEFT)}
    });
    
    inputService->registerAction({
        "player_move_right",
        InputActionType::DIGITAL,
        "Move player right",
        {InputBinding(InputBinding::InputType::KEYBOARD_KEY, SDL_SCANCODE_RIGHT)}
    });
    
    inputService->registerAction({
        "player_move_up",
        InputActionType::DIGITAL,
        "Move player up",
        {InputBinding(InputBinding::InputType::KEYBOARD_KEY, SDL_SCANCODE_UP)}
    });
    
    inputService->registerAction({
        "player_move_down",
        InputActionType::DIGITAL,
        "Move player down",
        {InputBinding(InputBinding::InputType::KEYBOARD_KEY, SDL_SCANCODE_DOWN)}
    });
}

void GameControlService::integrateWithCameraService() {
    if (!cameraService) return;
    
    // Set up camera integration callbacks if needed
    // For now, direct service calls are sufficient
}

void GameControlService::integrateWithRenderingService() {
    if (!renderingService) return;
    
    // Set up rendering integration callbacks if needed
    // For now, direct service calls are sufficient
}

// Action implementations
void GameControlService::actionToggleMovement() {
    toggleMovementType();
}

void GameControlService::actionCreateEntity() {
    std::cout << "GameControlService::actionCreateEntity() - Setting request flag" << std::endl;
    controlState.requestEntityCreation = true;
}

void GameControlService::actionCreateSwarm() {
    controlState.requestSwarmCreation = true;
}

void GameControlService::actionDebugEntity() {
    glm::vec2 mouseWorldPos = inputService->getMouseWorldPosition();
    debugEntityAtPosition(mouseWorldPos);
}

void GameControlService::actionShowStats() {
    controlState.requestPerformanceStats = true;
}

void GameControlService::actionGraphicsTests() {
    controlState.requestGraphicsTests = true;
}

void GameControlService::actionToggleDebug() {
    toggleDebugMode();
}

void GameControlService::actionCameraReset() {
    resetCamera();
}

void GameControlService::actionCameraFocus() {
    focusCameraOnEntities();
}

// Game logic implementations
void GameControlService::toggleMovementType() {
    controlState.currentMovementType = (controlState.currentMovementType + 1) % 1; // Only RandomWalk for now
    DEBUG_LOG("Movement type: " << controlState.currentMovementType);
}

void GameControlService::createEntity(const glm::vec2& position) {
    std::cout << "GameControlService::createEntity() - Called with position (" << position.x << ", " << position.y << ")" << std::endl;
    
    if (!entityFactory) {
        std::cout << "ERROR: entityFactory is null!" << std::endl;
        return;
    }
    if (!renderer) {
        std::cout << "ERROR: renderer is null!" << std::endl;
        return;
    }
    
    glm::vec3 pos3d(position.x, position.y, 0.0f);
    flecs::entity entity = entityFactory->createExactEntity(pos3d);
    std::cout << "Created single entity at exact position from EntityFactory" << std::endl;
    
    auto* gpuEntityManager = renderer->getGPUEntityManager();
    if (gpuEntityManager) {
        std::vector<flecs::entity> entities = {entity};
        gpuEntityManager->addEntitiesFromECS(entities);
        gpuEntityManager->uploadPendingEntities();
        std::cout << "Added entity to GPU manager and uploaded" << std::endl;
    } else {
        std::cout << "ERROR: gpuEntityManager is null!" << std::endl;
    }
    
    DEBUG_LOG("Created entity at (" << position.x << ", " << position.y << ")");
}

void GameControlService::createSwarm(size_t count, const glm::vec3& center, float radius) {
    if (!entityFactory || !renderer) return;
    
    auto entities = entityFactory->createSwarm(count, center, radius);
    
    auto* gpuEntityManager = renderer->getGPUEntityManager();
    if (gpuEntityManager) {
        gpuEntityManager->addEntitiesFromECS(entities);
        gpuEntityManager->uploadPendingEntities();
    }
    
    DEBUG_LOG("Created swarm of " << count << " entities");
}

void GameControlService::showPerformanceStats() {
    if (!world) return;
    
    float avgFrameTime = Profiler::getInstance().getFrameTime();
    size_t activeEntities = static_cast<size_t>(world->count<Transform>());
    float fps = avgFrameTime > 0.0f ? (1000.0f / avgFrameTime) : 0.0f;
    
    std::cout << "\n=== Performance Statistics ===" << std::endl;
    std::cout << "FPS: " << fps << std::endl;
    std::cout << "Frame Time: " << avgFrameTime << "ms" << std::endl;
    std::cout << "Active Entities: " << activeEntities << std::endl;
    
    if (renderingService) {
        const auto& renderStats = renderingService->getRenderStats();
        const auto& cullingStats = renderingService->getCullingStats();
        
        std::cout << "Draw Calls: " << renderStats.totalDrawCalls << std::endl;
        std::cout << "Visible Entities: " << cullingStats.visibleEntities << std::endl;
        std::cout << "Culling Ratio: " << (cullingStats.getCullingRatio() * 100.0f) << "%" << std::endl;
    }
    std::cout << "=========================" << std::endl;
}

void GameControlService::runGraphicsTests() {
    DEBUG_LOG("Running graphics stress tests...");
    
    // Create additional test entities
    if (entityFactory && renderer) {
        auto testEntities = entityFactory->createSwarm(5000, glm::vec3(0.0f, 0.0f, 0.0f), 15.0f);
        
        auto* gpuEntityManager = renderer->getGPUEntityManager();
        if (gpuEntityManager) {
            gpuEntityManager->addEntitiesFromECS(testEntities);
            gpuEntityManager->uploadPendingEntities();
        }
        
        DEBUG_LOG("Created 5000 test entities for graphics testing");
    }
}

void GameControlService::toggleDebugMode() {
    controlState.debugMode = !controlState.debugMode;
    
    if (renderingService) {
        renderingService->setDebugVisualization(controlState.debugMode);
    }
    
    DEBUG_LOG("Debug mode: " << (controlState.debugMode ? "ON" : "OFF"));
}

void GameControlService::toggleWireframeMode() {
    controlState.wireframeMode = !controlState.wireframeMode;
    
    if (renderingService) {
        renderingService->setWireframeMode(controlState.wireframeMode);
    }
    
    DEBUG_LOG("Wireframe mode: " << (controlState.wireframeMode ? "ON" : "OFF"));
}

void GameControlService::handleCameraControls() {
    if (!cameraService || !inputService) return;
    
    const float baseMoveSpeed = 30.0f; // base units per second at 1.0x zoom (increased)
    const float maxMoveSpeed = 100.0f; // cap speed when zoomed way out
    const float zoomSensitivity = 0.05f;  // zoom sensitivity per wheel tick (smoother)
    
    // Get current zoom level to adjust movement speed
    float currentZoom = cameraService->getCameraZoom(cameraService->getActiveCameraID());
    float zoomAdjustedMoveSpeed = std::min(maxMoveSpeed, baseMoveSpeed / currentZoom); // Inverse relationship with cap
    
    glm::vec3 movement(0.0f);
    float zoomDelta = 0.0f;
    
    // Calculate movement based on input - zoom-adjusted for fine control when zoomed in
    if (inputService->isActionActive("camera_move_forward")) {
        movement.y += zoomAdjustedMoveSpeed * deltaTime;
    }
    if (inputService->isActionActive("camera_move_backward")) {
        movement.y -= zoomAdjustedMoveSpeed * deltaTime;
    }
    if (inputService->isActionActive("camera_move_left")) {
        movement.x -= zoomAdjustedMoveSpeed * deltaTime;
    }
    if (inputService->isActionActive("camera_move_right")) {
        movement.x += zoomAdjustedMoveSpeed * deltaTime;
    }
    
    // Calculate zoom from mouse wheel (positive = zoom in, negative = zoom out)
    float wheelDelta = inputService->getActionAnalog1D("camera_zoom");
    if (std::abs(wheelDelta) > 0.01f) {
        std::cout << "Wheel delta: " << wheelDelta << std::endl;
        zoomDelta = wheelDelta * zoomSensitivity; // Much smaller sensitivity for fine control
    }
    
    // Apply movement if any
    if (movement.x != 0.0f || movement.y != 0.0f || movement.z != 0.0f) {
        glm::vec3 currentPos = cameraService->getCameraPosition(cameraService->getActiveCameraID());
        cameraService->setCameraPosition(cameraService->getActiveCameraID(), currentPos + movement);
    }
    
    // Apply zoom if any
    if (zoomDelta != 0.0f) {
        float currentZoom = cameraService->getCameraZoom(cameraService->getActiveCameraID());
        
        // Use exponential zoom for smoother feel (like most applications)
        // Positive delta = zoom in (multiply by >1), negative = zoom out (multiply by <1)
        float zoomMultiplier = std::pow(1.1f, zoomDelta * 10.0f); // 1.1^(delta*10) for smooth exponential scaling
        float newZoom = std::max(0.05f, std::min(20.0f, currentZoom * zoomMultiplier)); // Wider range: 0.05x to 20x
        
        std::cout << "Zoom: " << currentZoom << " -> " << newZoom << " (multiplier: " << zoomMultiplier << ")" << std::endl;
        cameraService->setCameraZoom(cameraService->getActiveCameraID(), newZoom);
    }
}

void GameControlService::updatePlayerMovement() {
    if (!renderer || !world) {
        return;
    }

    auto* gpuEntityManager = renderer->getGPUEntityManager();
    if (!gpuEntityManager) {
        return;
    }

    auto inputEntity = world->lookup("InputEntity");
    if (!inputEntity.is_valid()) {
        return;
    }

    const auto* inputState = inputEntity.get<InputState>();
    if (inputState && !inputState->processKeyboard) {
        return;
    }

    const auto* keyboardInput = inputEntity.get<KeyboardInput>();
    if (!keyboardInput) {
        return;
    }

    glm::vec2 direction(0.0f);
    // Player movement uses raw key state to avoid input context conflicts.
    if (keyboardInput->isKeyDown(SDL_SCANCODE_LEFT)) {
        direction.x -= 1.0f;
    }
    if (keyboardInput->isKeyDown(SDL_SCANCODE_RIGHT)) {
        direction.x += 1.0f;
    }
    if (keyboardInput->isKeyDown(SDL_SCANCODE_UP)) {
        direction.y += 1.0f;
    }
    if (keyboardInput->isKeyDown(SDL_SCANCODE_DOWN)) {
        direction.y -= 1.0f;
    }

    if (direction.x != 0.0f || direction.y != 0.0f) {
        direction = glm::normalize(direction);
    }

    world->each([&](flecs::entity e, Player& player, PlayerControl& control, GPUIndex& gpuIndex) {
        glm::vec2 velocity = direction * player.speed;
        control.desiredVelocity = velocity;
        control.active = (direction.x != 0.0f || direction.y != 0.0f);
        gpuEntityManager->updateControlParamsForEntity(
            gpuIndex.index,
            glm::vec4(control.desiredVelocity.x, control.desiredVelocity.y, 1.0f, control.renderScale)
        );
    });
}

void GameControlService::resetCamera() {
    if (!cameraService) return;
    
    // Reset to default camera position
    cameraService->setCameraPosition(cameraService->getActiveCameraID(), glm::vec3(0.0f, 0.0f, 10.0f));
    cameraService->setCameraZoom(cameraService->getActiveCameraID(), 1.0f);
    
    DEBUG_LOG("Camera reset to default position");
}

void GameControlService::focusCameraOnEntities() {
    if (!cameraService || !world) return;
    
    // Calculate center of all entities
    glm::vec3 center(0.0f);
    size_t entityCount = 0;
    
    world->each<Transform>([&](flecs::entity e, Transform& transform) {
        center += transform.position;
        entityCount++;
    });
    
    if (entityCount > 0) {
        center /= static_cast<float>(entityCount);
        cameraService->focusCameraOn(cameraService->getActiveCameraID(), center, 1.0f);
        DEBUG_LOG("Camera focused on entity center: (" << center.x << ", " << center.y << ", " << center.z << ")");
    }
}

void GameControlService::printControlStats() const {
    std::cout << "\n=== Control Service Statistics ===" << std::endl;
    std::cout << "Registered Actions: " << actions.size() << std::endl;
    std::cout << "Debug Mode: " << (controlState.debugMode ? "ON" : "OFF") << std::endl;
    std::cout << "Wireframe Mode: " << (controlState.wireframeMode ? "ON" : "OFF") << std::endl;
    std::cout << "Movement Type: " << controlState.currentMovementType << std::endl;
    std::cout << "==================================" << std::endl;
}

void GameControlService::logControlState() const {
    DEBUG_LOG("ControlService State - Debug:" << controlState.debugMode 
              << " Wireframe:" << controlState.wireframeMode 
              << " Movement:" << controlState.currentMovementType);
}

void GameControlService::printControlInstructions() const {
    std::cout << "\n=== Flecs GPU Compute Movement Demo Controls ===" << std::endl;
    std::cout << "ESC: Exit" << std::endl;
    std::cout << "P: Print detailed performance report" << std::endl;
    std::cout << "+/=: Add 1000 more GPU entities" << std::endl;
    std::cout << "Left Click: Create GPU entity with movement at mouse position" << std::endl;
    std::cout << "All entities use random walk movement pattern" << std::endl;
    std::cout << "T: Run graphics buffer overflow tests" << std::endl;
    std::cout << "F3: Toggle debug mode" << std::endl;
    std::cout << "R: Reset camera" << std::endl;
    std::cout << "F: Focus camera on entities" << std::endl;
    std::cout << "WASD: Move camera" << std::endl;
    std::cout << "Right Click: Debug entity info at mouse position" << std::endl;
    std::cout << "Mouse Wheel: Zoom camera" << std::endl;
    std::cout << "Arrow Keys: Move player" << std::endl;
    std::cout << "===============================================\n" << std::endl;
}

void GameControlService::debugEntityAtPosition(const glm::vec2& worldPos) {
    if (!renderer) {
        std::cerr << "GameControlService::debugEntityAtPosition - No renderer available" << std::endl;
        return;
    }
    
    auto* gpuEntityManager = renderer->getGPUEntityManager();
    if (!gpuEntityManager) {
        std::cerr << "GameControlService::debugEntityAtPosition - No GPUEntityManager available" << std::endl;
        return;
    }
    
    // Get the entity buffer manager for readback
    auto& bufferManager = gpuEntityManager->getBufferManager();
    
    // Perform GPU readback at the clicked position with synchronization
    EntityBufferManager::EntityDebugInfo debugInfo;
    if (bufferManager.readbackEntityAtPositionSafe(worldPos, debugInfo)) {
        // Get ECS entity ID from GPU index
        auto ecsEntity = gpuEntityManager->getECSEntityFromGPUIndex(debugInfo.entityId);
        
        std::cout << "\n=== ENTITY DEBUG INFO ===" << std::endl;
        std::cout << "World Position: (" << worldPos.x << ", " << worldPos.y << ")" << std::endl;
        std::cout << "GPU Buffer Index: " << debugInfo.entityId << std::endl;
        std::cout << "ECS Entity ID: " << std::hex << ecsEntity.id() << std::dec;
        if (ecsEntity.is_valid()) {
            std::cout << " (valid)";
        } else {
            std::cout << " (invalid/unmapped)";
        }
        std::cout << std::endl;
        std::cout << "Position: (" << debugInfo.position.x << ", " << debugInfo.position.y 
                  << ", " << debugInfo.position.z << ")" << std::endl;
        std::cout << "Velocity: (" << debugInfo.velocity.x << ", " << debugInfo.velocity.y 
                  << ") | Damping: " << debugInfo.velocity.z << std::endl;
        std::cout << "Spatial Cell: " << debugInfo.spatialCell << std::endl;
        
        // Calculate spatial cell coordinates for readability
        const uint32_t GRID_WIDTH = 64;
        uint32_t cellX = debugInfo.spatialCell % GRID_WIDTH;
        uint32_t cellY = debugInfo.spatialCell / GRID_WIDTH;
        std::cout << "Spatial Grid: (" << cellX << ", " << cellY << ")" << std::endl;
        std::cout << "========================\n" << std::endl;
    } else {
        std::cout << "No entity found at world position (" << worldPos.x << ", " << worldPos.y << ")" << std::endl;
    }
}
