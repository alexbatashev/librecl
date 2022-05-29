#include "passes.hpp"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
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

  Block &entry = func.getBody().emplaceBlock();
  entry.addArgument(builder.getI32Type(), builder.getUnknownLoc());

  builder.setInsertionPointToStart(&entry);

  Block &elseX = func.getBody().emplaceBlock();
  Block &elseY = func.getBody().emplaceBlock();

  Block &blockX = func.getBody().emplaceBlock();
  Block &blockY = func.getBody().emplaceBlock();
  Block &blockZ = func.getBody().emplaceBlock();

  Block &exit = func.getBody().emplaceBlock();
  exit.addArgument(builder.getIndexType(), builder.getUnknownLoc());

  Value zero = builder.create<arith::ConstantOp>(
      builder.getUnknownLoc(), builder.getIntegerAttr(builder.getI32Type(), 0),
      builder.getI32Type());
  Value isX = builder.create<arith::CmpIOp>(builder.getUnknownLoc(),
                                            arith::CmpIPredicate::eq,
                                            entry.getArgument(0), zero);
  builder.create<cf::CondBranchOp>(builder.getUnknownLoc(), isX, &blockX,
                                   ValueRange{}, &elseX, ValueRange{});

  builder.setInsertionPointToStart(&elseX);

  Value one = builder.create<arith::ConstantOp>(
      builder.getUnknownLoc(), builder.getIntegerAttr(builder.getI32Type(), 1),
      builder.getI32Type());
  Value isY = builder.create<arith::CmpIOp>(builder.getUnknownLoc(),
                                            arith::CmpIPredicate::eq,
                                            entry.getArgument(0), zero);
  builder.create<cf::CondBranchOp>(builder.getUnknownLoc(), isY, &blockY,
                                   ValueRange{}, &elseY, ValueRange{});

  builder.setInsertionPointToStart(&elseY);
  Value two = builder.create<arith::ConstantOp>(
      builder.getUnknownLoc(), builder.getIntegerAttr(builder.getI32Type(), 2),
      builder.getI32Type());
  Value defaultVal = builder.create<arith::ConstantOp>(
      builder.getUnknownLoc(), builder.getIndexAttr(0), builder.getIndexType());
  Value isZ = builder.create<arith::CmpIOp>(builder.getUnknownLoc(),
                                            arith::CmpIPredicate::eq,
                                            entry.getArgument(0), zero);
  builder.create<cf::CondBranchOp>(builder.getUnknownLoc(), isZ, &blockZ,
                                   ValueRange{}, &exit, ValueRange{defaultVal});
  // builder.create<cf::SwitchOp>(builder.getUnknownLoc(), entry.getArgument(0),
  // &exit, defaultVal, ArrayRef<int32_t>{0, 1, 2}, BlockRange{&blockX, &blockY,
  // &blockZ}, ArrayRef<ValueRange>{ValueRange{}, ValueRange{}, ValueRange{}});
  // TODO replace with cf::SwitchOp once OpSwitch is supported in SPIR-V
  // conversion

  builder.setInsertionPointToStart(&blockX);
  auto xDim = builder.create<gpu::GlobalIdOp>(builder.getUnknownLoc(),
                                              gpu::Dimension::x);
  builder.create<cf::BranchOp>(builder.getUnknownLoc(), &exit,
                               ValueRange{xDim});

  builder.setInsertionPointToStart(&blockY);
  auto yDim = builder.create<gpu::GlobalIdOp>(builder.getUnknownLoc(),
                                              gpu::Dimension::y);
  builder.create<cf::BranchOp>(builder.getUnknownLoc(), &exit,
                               ValueRange{yDim});

  builder.setInsertionPointToStart(&blockZ);
  auto zDim = builder.create<gpu::GlobalIdOp>(builder.getUnknownLoc(),
                                              gpu::Dimension::z);
  builder.create<cf::BranchOp>(builder.getUnknownLoc(), &exit,
                               ValueRange{zDim});

  builder.setInsertionPointToStart(&exit);
  auto res = builder.create<arith::IndexCastOp>(
      builder.getUnknownLoc(), builder.getI64Type(), exit.getArgument(0));
  builder.create<func::ReturnOp>(builder.getUnknownLoc(), res.getOut());
}

struct ExpandOpenCLFunctionsPass
    : public PassWrapper<ExpandOpenCLFunctionsPass,
                         OperationPass<gpu::GPUModuleOp>> {
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
