#pragma once

#include <memory>

namespace mlir {
class Pass;
}

namespace lcl {
std::unique_ptr<mlir::Pass> createSPIRToGPUPass();
}
