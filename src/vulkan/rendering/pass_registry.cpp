#include "pass_registry.h"

void RenderPassRegistry::registerPass(
    const std::string& name,
    std::function<void(FrameGraph&, const RenderPassContext&)> registerFn) {
    passes.push_back(RenderPassDefinition{name, std::move(registerFn)});
}

void RenderPassRegistry::build(FrameGraph& frameGraph, const RenderPassContext& context) const {
    for (const auto& pass : passes) {
        if (pass.registerPass) {
            pass.registerPass(frameGraph, context);
        }
    }
}
