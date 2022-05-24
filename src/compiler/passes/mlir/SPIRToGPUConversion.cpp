#include "passes.hpp"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct GPUTarget : public ConversionTarget {
  GPUTarget(MLIRContext &ctx) : ConversionTarget(ctx) {
    addLegalDialect<gpu::GPUDialect>();
    addLegalDialect<cf::ControlFlowDialect>();
    addLegalDialect<arith::ArithmeticDialect>();
    addLegalDialect<memref::MemRefDialect>();

    addDynamicallyLegalOp<ModuleOp>([](ModuleOp op) {
      auto &body = op.getBodyRegion();
      auto &entryBlock = body.front();

      size_t numOps = std::distance(entryBlock.begin(), entryBlock.end());

      if (numOps != 2)
        return false;

      if (!llvm::isa<gpu::GPUModuleOp>(entryBlock.front()))
        return false;

      return true;
    });

    addIllegalDialect<LLVM::LLVMDialect>();
  }
};

class ConvertSPIRToGPUPass : public PassWrapper<ConvertSPIRToGPUPass, mlir::OperationPass<ModuleOp>> {
  void runOnOperation() override {
    TypeConverter typeConverter{};

    GPUTarget target{getContext()};

    RewritePatternSet patterns{&getContext()};

    lcl::populateSPIRToGPUTypeConversions(typeConverter);
    lcl::populateSPIRToGPUConversionPatterns(typeConverter, patterns);

    ModuleOp module = getOperation();

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

struct ModuleConversionPattern : public OpConversionPattern<ModuleOp> {
  using Base = OpConversionPattern<ModuleOp>;
  using Base::OpConversionPattern;

  LogicalResult matchAndRewrite(ModuleOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    ModuleOp newModule = rewriter.create<ModuleOp>(op.getLoc());
    gpu::GPUModuleOp gpuModule;
    {
      mlir::OpBuilder::InsertionGuard _{rewriter};
      rewriter.setInsertionPointToStart(&newModule.getBodyRegion().front());

      gpuModule = rewriter.create<gpu::GPUModuleOp>(op.getLoc(), "ocl_program");
    }

    rewriter.inlineRegionBefore(newModule.getBodyRegion(), gpuModule.body(), gpuModule.body().end());

    rewriter.eraseOp(newModule);
    return success();
  }
};
struct FuncConversionPattern : public OpConversionPattern<LLVM::LLVMFuncOp> {
  using Base = OpConversionPattern<LLVM::LLVMFuncOp>;
  using Base::OpConversionPattern;

  LogicalResult matchAndRewrite(LLVM::LLVMFuncOp func, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    if (func.getName().startswith("llvm.lifetime")) {
      rewriter.eraseOp(func);
      return success();
    }

    auto funcType = func.getFunctionType();

    SmallVector<Type, 8> convertedArgs;

    assert(getTypeConverter());
    for (auto &t : funcType.getParams()) {
      convertedArgs.push_back(getTypeConverter()->convertType(t));
    }

    SmallVector<Type, 1> retTypes;
    if (!funcType.getReturnType().isa<LLVM::LLVMVoidType>())
      retTypes.push_back(getTypeConverter()->convertType(funcType.getReturnType()));

    auto gpuType = FunctionType::get(func.getContext(), convertedArgs, retTypes);

    StringRef name = func.getName();

    auto gpuFunc = rewriter.create<gpu::GPUFuncOp>(func.getLoc(), name, gpuType);

    /*
    rewriter.inlineRegionBefore(func.getBody(), gpuFunc.getBody(),
                                gpuFunc.end());
    TypeConverter &typeConverter = *getTypeConverter();
    TypeConverter::SignatureConversion signatureConverter(
        funcType.getNumParams());
    if (failed(rewriter.convertRegionTypes(&gpuFunc.getBody(), typeConverter,
                                           &signatureConverter))) {
      return failure();
    }
    */

    // gpuFunc.dump();

    rewriter.eraseOp(func);

    return success();
  }
};
}

namespace lcl {
void populateSPIRToGPUTypeConversions(TypeConverter &converter) {
  converter.addConversion([](LLVM::LLVMPointerType type) {
    Type elementType;
    // TODO are opaque pointers this big of a deal?
    if (type.isOpaque()) {
      elementType = IntegerType::get(type.getContext(), 8);
    } else {
      elementType = type.getElementType();
    }
    MemRefType::Builder builder({-1}, elementType);
    builder.setMemorySpace(type.getAddressSpace());

    return static_cast<MemRefType>(builder);
  });
  converter.addConversion([](IntegerType t) { return t; });
  converter.addConversion([](FloatType t) { return t; });
  converter.addConversion([](VectorType t) { return t; });
}

void populateSPIRToGPUConversionPatterns(TypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<ModuleConversionPattern, FuncConversionPattern>(converter, patterns.getContext());
}

std::unique_ptr<mlir::Pass> createSPIRToGPUPass() {
  return std::make_unique<ConvertSPIRToGPUPass>();
}
}
