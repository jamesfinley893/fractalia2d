#include "src/ecs/services/control_service.h"
#include "src/ecs/core/service_locator.h"

int main() {
    static_assert(Service<GameControlService>);
    return 0;
}
