#include "passes.hpp"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {
void expandGetGlobalId(func::FuncOp func) {
  OpBuilder builder(func.getContext());

  func->setAttr("ocl_builtin", builder.getUnitAttr());

  Block &entry = func.getBody().emplaceBlock();
  entry.addArgument(builder.getI32Type(), builder.getUnknownLoc());

  builder.setInsertionPointToStart(&entry);

  Type retType = builder.getIndexType();

  Value zero = builder.create<arith::ConstantOp>(
      builder.getUnknownLoc(), builder.getIntegerAttr(builder.getI32Type(), 0),
      builder.getI32Type());
  Value isX = builder.create<arith::CmpIOp>(builder.getUnknownLoc(),
                                            arith::CmpIPredicate::eq,
                                            entry.getArgument(0), zero);
  scf::IfOp retValue = builder.create<scf::IfOp>(
      builder.getUnknownLoc(), TypeRange{retType}, isX, /*withElse*/ true);
  // TODO this is a hack for Vulkan SPIR-V target
  Value res32 = builder.create<arith::IndexCastOp>(
      builder.getUnknownLoc(), builder.getI32Type(),
      retValue.getResults().front());
  Value res = builder.create<arith::ExtSIOp>(builder.getUnknownLoc(),
                                             builder.getI64Type(), res32);
  builder.create<func::ReturnOp>(builder.getUnknownLoc(), res);

  builder.setInsertionPointToStart(&retValue.getThenRegion().front());
  Value xDim = builder.create<gpu::GlobalIdOp>(builder.getUnknownLoc(),
                                               gpu::Dimension::x);
  builder.create<scf::YieldOp>(builder.getUnknownLoc(), xDim);

  builder.setInsertionPointToStart(&retValue.getElseRegion().front());

  Value one = builder.create<arith::ConstantOp>(
      builder.getUnknownLoc(), builder.getIntegerAttr(builder.getI32Type(), 1),
      builder.getI32Type());
  Value isY = builder.create<arith::CmpIOp>(builder.getUnknownLoc(),
                                            arith::CmpIPredicate::eq,
                                            entry.getArgument(0), one);
  scf::IfOp yBranch = builder.create<scf::IfOp>(
      builder.getUnknownLoc(), TypeRange{retType}, isY, /*withElse*/ true);
  builder.create<scf::YieldOp>(builder.getUnknownLoc(), yBranch.getResults());

  builder.setInsertionPointToStart(&yBranch.getThenRegion().front());

  Value yDim = builder.create<gpu::GlobalIdOp>(builder.getUnknownLoc(),
                                               gpu::Dimension::y);
  builder.create<scf::YieldOp>(builder.getUnknownLoc(), yDim);

  builder.setInsertionPointToStart(&yBranch.getElseRegion().front());
  Value two = builder.create<arith::ConstantOp>(
      builder.getUnknownLoc(), builder.getIntegerAttr(builder.getI32Type(), 2),
      builder.getI32Type());
  Value isZ = builder.create<arith::CmpIOp>(builder.getUnknownLoc(),
                                            arith::CmpIPredicate::eq,
                                            entry.getArgument(0), two);
  scf::IfOp zBranch = builder.create<scf::IfOp>(
      builder.getUnknownLoc(), TypeRange{retType}, isZ, /*withElse*/ true);
  builder.create<scf::YieldOp>(builder.getUnknownLoc(), zBranch.getResults());

  builder.setInsertionPointToStart(&zBranch.getThenRegion().front());

  Value zDim = builder.create<gpu::GlobalIdOp>(builder.getUnknownLoc(),
                                               gpu::Dimension::z);
  builder.create<scf::YieldOp>(builder.getUnknownLoc(), zDim);

  builder.setInsertionPointToStart(&zBranch.getElseRegion().front());

  Value defaultVal = builder.create<arith::ConstantOp>(
      builder.getUnknownLoc(), builder.getIndexAttr(0), builder.getIndexType());
  builder.create<scf::YieldOp>(builder.getUnknownLoc(), defaultVal);
}

struct ExpandOpenCLFunctionsPass
    : public PassWrapper<ExpandOpenCLFunctionsPass,
                         OperationPass<gpu::GPUModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExpandOpenCLFunctionsPass);

  void runOnOperation() override {
    gpu::GPUModuleOp module = getOperation();

    auto getGlobalId = module.lookupSymbol<func::FuncOp>("_Z13get_global_idj");
    if (getGlobalId)
      expandGetGlobalId(getGlobalId);
  }
};
} // namespace

namespace lcl {
std::unique_ptr<Pass> createExpandOpenCLFunctionsPass() {
  return std::make_unique<ExpandOpenCLFunctionsPass>();
}
} // namespace lcl
