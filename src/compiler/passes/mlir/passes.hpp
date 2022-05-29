#pragma once

#include <memory>

namespace mlir {
class Pass;
class RewritePatternSet;
class TypeConverter;
namespace spirv {
class TargetEnvAttr;
}
}

namespace lcl {
void populateSPIRToGPUTypeConversions(mlir::TypeConverter &);
void populateSPIRToGPUConversionPatterns(mlir::TypeConverter &,
                                         mlir::RewritePatternSet &);

void populateRawMemoryToSPIRVTypeConversions(mlir::TypeConverter &,
                                             mlir::spirv::TargetEnvAttr);
void populateRawMemoryToSPIRVConversionPatterns(mlir::TypeConverter &,
                                                mlir::RewritePatternSet &);

std::unique_ptr<mlir::Pass> createSPIRToGPUPass();
std::unique_ptr<mlir::Pass> createAIRKernelABIPass();
std::unique_ptr<mlir::Pass> createExpandOpenCLFunctionsPass();
std::unique_ptr<mlir::Pass> createGPUToSPIRVPass();
}
