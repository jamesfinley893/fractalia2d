#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <iostream>
#include <chrono>
#include <thread>

#include "vulkan_renderer.h"
#include "ecs/utilities/debug.h"
#include <flecs.h>
#include "ecs/core/entity_factory.h"
#include "ecs/systems/lifetime_system.h"
#include "ecs/components/component.h"
#include "ecs/utilities/profiler.h"
#include "ecs/gpu/gpu_entity_manager.h"

// New service-based architecture includes
#include "ecs/core/world_manager.h"
#include "ecs/core/service_locator.h"
#include "ecs/services/input_service.h"
#include "ecs/services/camera_service.h"
#include "ecs/services/rendering_service.h"
// TESTING RENAMED CONTROL SERVICE
#include "ecs/services/control_service.h"

int main(int argc, char* argv[]) {
    constexpr int TARGET_FPS = 60;
    constexpr float TARGET_FRAME_TIME = 1000.0f / TARGET_FPS; // 16.67ms
    
    // Set SDL vsync hint to 0 for safety (ignored with pure Vulkan, but good practice)
    SDL_SetHint(SDL_HINT_RENDER_VSYNC, "0");
    
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        std::cerr << "Failed to initialize SDL: " << SDL_GetError() << std::endl;
        return -1;
    }
    
    // Check Vulkan support
    uint32_t extensionCount = 0;
    const char* const* extensions = SDL_Vulkan_GetInstanceExtensions(&extensionCount);
    if (!extensions || extensionCount == 0) {
        std::cerr << "Vulkan is not supported or no Vulkan extensions available" << std::endl;
        std::cerr << "Make sure Vulkan drivers are installed" << std::endl;
        SDL_Quit();
        return -1;
    }
    
    SDL_Window* window = SDL_CreateWindow(
        "Fractalia2 - SDL3 + Vulkan + Flecs",
        800, 600,
        SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE
    );

    if (!window) {
        std::cerr << "Failed to create window: " << SDL_GetError() << std::endl;
        SDL_Quit();
        return -1;
    }
    
    VulkanRenderer renderer;
    if (!renderer.initialize(window)) {
        std::cerr << "Failed to initialize Vulkan renderer" << std::endl;
        SDL_DestroyWindow(window);
        SDL_Quit();
        return -1;
    }

    // Initialize service-based architecture with proper priorities
    auto& serviceLocator = ServiceLocator::instance();
    
    auto worldManager = serviceLocator.createAndRegister<WorldManager>("WorldManager", 100);
    auto inputService = serviceLocator.createAndRegister<InputService>("InputService", 90);
    auto cameraService = serviceLocator.createAndRegister<CameraService>("CameraService", 80);
    auto renderingService = serviceLocator.createAndRegister<RenderingService>("RenderingService", 70);
    // TESTING RENAMED CONTROL SERVICE
    auto controlService = serviceLocator.createAndRegister<GameControlService>("GameControlService", 60);
    
    // Initialize world manager first to get the world reference
    if (!worldManager->initialize()) {
        std::cerr << "Failed to initialize WorldManager" << std::endl;
        return -1;
    }
    
    // Create EntityFactory early as it's needed by ControlService
    flecs::world& world = worldManager->getWorld();
    EntityFactory entityFactory(world);
    
    // Declare service dependencies
    serviceLocator.declareDependencies<InputService, WorldManager>();
    serviceLocator.declareDependencies<CameraService, WorldManager>();
    serviceLocator.declareDependencies<RenderingService, WorldManager>();
    // RESTORED WITH NEW NAME
    serviceLocator.declareDependencies<GameControlService, WorldManager, InputService, CameraService, RenderingService>();
    
    // Validate service dependencies
    if (!serviceLocator.validateDependencies()) {
        std::cerr << "Service dependency validation failed" << std::endl;
        return -1;
    }
    
    // Initialize services in dependency order with comprehensive error handling
    try {
        // WorldManager already initialized above
        serviceLocator.setServiceLifecycle<WorldManager>(ServiceLifecycle::INITIALIZED);
        
        serviceLocator.setServiceLifecycle<CameraService>(ServiceLifecycle::INITIALIZING);
        if (!cameraService->initialize(worldManager->getWorld())) {
            throw std::runtime_error("Failed to initialize CameraService");
        }
        serviceLocator.setServiceLifecycle<CameraService>(ServiceLifecycle::INITIALIZED);
        
        serviceLocator.setServiceLifecycle<InputService>(ServiceLifecycle::INITIALIZING);
        if (!inputService->initialize(worldManager->getWorld(), window)) {
            throw std::runtime_error("Failed to initialize InputService");
        }
        serviceLocator.setServiceLifecycle<InputService>(ServiceLifecycle::INITIALIZED);
        
        serviceLocator.setServiceLifecycle<RenderingService>(ServiceLifecycle::INITIALIZING);
        if (!renderingService->initialize(worldManager->getWorld(), &renderer)) {
            throw std::runtime_error("Failed to initialize RenderingService");
        }
        serviceLocator.setServiceLifecycle<RenderingService>(ServiceLifecycle::INITIALIZED);
        
        // RESTORED WITH NEW NAME
        serviceLocator.setServiceLifecycle<GameControlService>(ServiceLifecycle::INITIALIZING);
        if (!controlService->initialize(worldManager->getWorld(), &renderer, &entityFactory)) {
            throw std::runtime_error("Failed to initialize GameControlService");
        }
        serviceLocator.setServiceLifecycle<GameControlService>(ServiceLifecycle::INITIALIZED);
        
        // Final service validation
        if (!serviceLocator.initializeAllServices()) {
            throw std::runtime_error("Service initialization validation failed");
        }
        
        DEBUG_LOG("All services initialized successfully");
        serviceLocator.printServiceStatus();
        
    } catch (const std::exception& e) {
        std::cerr << "Service initialization error: " << e.what() << std::endl;
        serviceLocator.clear();
        return -1;
    }
    
    // Simple system registration - no complex scheduling needed
    world.system<Lifetime>("LifetimeSystem")
        .each(lifetime_system);
    
    DEBUG_LOG("Camera entities: " << world.count<Camera>());
    
    renderer.setWorld(&world);
    
    Profiler::getInstance().setTargetFrameTime(TARGET_FRAME_TIME);

    constexpr size_t ENTITY_COUNT = 10;
    
    DEBUG_LOG("Creating " << ENTITY_COUNT << " GPU entities for stress testing...");
    
    auto swarmEntities = entityFactory.createSwarm(
        ENTITY_COUNT,
        glm::vec3(10.0f, 10.0f, 0.0f),
        8.0f
    );
    
    auto* gpuEntityManager = renderer.getGPUEntityManager();
    gpuEntityManager->addEntitiesFromECS(swarmEntities);
    gpuEntityManager->uploadPendingEntities();
    
    DEBUG_LOG("Created " << swarmEntities.size() << " GPU entities!");

    flecs::entity playerEntity = entityFactory.createExactEntity(glm::vec3(0.0f, 0.0f, 0.0f));
    playerEntity.set<Player>({2.0f});
    playerEntity.set<PlayerControl>({glm::vec2(0.0f), false, 1.8f});
    if (auto* renderable = playerEntity.get_mut<Renderable>()) {
        renderable->color = glm::vec4(1.0f, 0.2f, 0.1f, 1.0f);
        renderable->markDirty();
    }
    if (auto* transform = playerEntity.get_mut<Transform>()) {
        transform->setScale(glm::vec3(2.0f, 2.0f, 1.0f));
    }
    gpuEntityManager->addEntitiesFromECS({playerEntity});
    gpuEntityManager->uploadPendingEntities();
    if (auto* gpuIndex = playerEntity.get<GPUIndex>()) {
        gpuEntityManager->updateControlParamsForEntity(
            gpuIndex->index,
            glm::vec4(0.0f, 0.0f, 1.0f, 1.8f)
        );
        gpuEntityManager->updateRuntimeStateForEntity(gpuIndex->index, glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
    }
    DEBUG_LOG("Total services active: " << ServiceLocator::instance().getServiceCount());
    
    bool running = true;
    
    DEBUG_LOG("\n🚀 Service-based architecture ready\n");
    int frameCount = 0;
    auto lastFrameTime = std::chrono::high_resolution_clock::now();
    
    while (running) {
        auto frameStartTime = std::chrono::high_resolution_clock::now();
        float deltaTime = std::chrono::duration<float>(frameStartTime - lastFrameTime).count();
        lastFrameTime = frameStartTime;
        
        deltaTime = std::min(deltaTime, 1.0f / 30.0f);
        
        inputService->processSDLEvents();
        // Frame cleanup for input (clear justPressed flags, etc.)
        inputService->processFrame(deltaTime);
        
        auto* appState = world.get<ApplicationState>();
        if (appState && (appState->requestQuit || !appState->running)) {
            running = false;
        }
        
        renderer.setDeltaTime(deltaTime);
        
        // RESTORED WITH NEW NAME - DEBUG CHECK
        if (controlService) {
            controlService->processFrame(deltaTime);
        } else {
            std::cout << "ERROR: controlService is null!" << std::endl;
        }
        
        // Handle window resize for camera aspect ratio
        int width, height;
        if (inputService->hasWindowResizeEvent(width, height)) {
            cameraService->handleWindowResize(width, height);
            renderer.updateAspectRatio(width, height);
            renderer.setFramebufferResized(true);
            DEBUG_LOG("Window resized to " << width << "x" << height);
        }

        PROFILE_BEGIN_FRAME();
        
        {
            PROFILE_SCOPE("ECS Update");
            worldManager->executeFrame(deltaTime);
        }
        
        {
            PROFILE_SCOPE("Input Cleanup");
            // Input cleanup is handled by services - no manual cleanup needed
            // Service-based architecture handles frame state management internally
        }

        {
            PROFILE_SCOPE("Vulkan Rendering");
            renderer.drawFrame();
        }

        frameCount++;
        PROFILE_END_FRAME();
        
        if (frameCount % 300 == 0) {
            float avgFrameTime = Profiler::getInstance().getFrameTime();
            size_t activeEntities = static_cast<size_t>(world.count<Transform>());
            size_t estimatedMemory = activeEntities * (sizeof(Transform) + sizeof(Renderable) + sizeof(MovementPattern));
            
            Profiler::getInstance().updateMemoryUsage(estimatedMemory);
            
            float fps = avgFrameTime > 0.0f ? (1000.0f / avgFrameTime) : 0.0f;
            std::cout << "Frame " << frameCount 
                      << ": Avg " << avgFrameTime << "ms"
                      << " (" << fps << " FPS)"
                      << " | Entities: " << activeEntities
                      << " | Est Memory: " << (estimatedMemory / 1024) << "KB"
                      << std::endl;
        }
        
        auto frameEndTime = std::chrono::high_resolution_clock::now();
        float frameTimeMs = std::chrono::duration<float, std::milli>(frameEndTime - frameStartTime).count();
        
        constexpr float MIN_FRAME_TIME = 11.0f;
        if (frameTimeMs < MIN_FRAME_TIME) {
            float remainingMs = MIN_FRAME_TIME - frameTimeMs;
            if (remainingMs > 0.5f) {
                SDL_Delay(static_cast<int>(remainingMs));
            }
        }
    }


    renderer.cleanup();
    
    // Cleanup services in proper order
    DEBUG_LOG("Shutting down services...");
    ServiceLocator::instance().clear();
    DEBUG_LOG("All services shut down successfully");
    
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
