#pragma once

#include <memory>

namespace mlir {
class Pass;
class RewritePatternSet;
class TypeConverter;
}

namespace lcl {
void populateSPIRToGPUTypeConversions(mlir::TypeConverter &);
void populateSPIRToGPUConversionPatterns(mlir::TypeConverter &,
                                         mlir::RewritePatternSet &);

std::unique_ptr<mlir::Pass> createSPIRToGPUPass();
std::unique_ptr<mlir::Pass> createAIRKernelABIPass();
}
