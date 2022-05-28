#include "passes.hpp"

#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {
void expandGetGlobalId(func::FuncOp func) {
  OpBuilder builder(func.getContext());

  Block &entry = func.getBody().emplaceBlock();
  entry.addArgument(builder.getI32Type(), builder.getUnknownLoc());

  builder.setInsertionPointToStart(&entry);


  Block &blockX = func.getBody().emplaceBlock();
  Block &blockY = func.getBody().emplaceBlock();
  Block &blockZ = func.getBody().emplaceBlock();

  Block &exit = func.getBody().emplaceBlock();
  exit.addArgument(builder.getIndexType(), builder.getUnknownLoc());

  Value defaultVal = builder.create<arith::ConstantOp>(builder.getUnknownLoc(), builder.getIndexAttr(0), builder.getIndexType());
  builder.create<cf::SwitchOp>(builder.getUnknownLoc(), entry.getArgument(0), &exit, defaultVal, ArrayRef<int32_t>{0, 1, 2}, BlockRange{&blockX, &blockY, &blockZ}, ArrayRef<ValueRange>{ValueRange{}, ValueRange{}, ValueRange{}});


  builder.setInsertionPointToStart(&blockX);
  auto xDim = builder.create<gpu::GlobalIdOp>(builder.getUnknownLoc(), gpu::Dimension::x);
  builder.create<cf::BranchOp>(builder.getUnknownLoc(), &exit, ValueRange{xDim});

  builder.setInsertionPointToStart(&blockY);
  auto yDim = builder.create<gpu::GlobalIdOp>(builder.getUnknownLoc(), gpu::Dimension::y);
  builder.create<cf::BranchOp>(builder.getUnknownLoc(), &exit, ValueRange{yDim});

  builder.setInsertionPointToStart(&blockZ);
  auto zDim = builder.create<gpu::GlobalIdOp>(builder.getUnknownLoc(), gpu::Dimension::z);
  builder.create<cf::BranchOp>(builder.getUnknownLoc(), &exit, ValueRange{zDim});

  builder.setInsertionPointToStart(&exit);
  auto res = builder.create<arith::IndexCastOp>(builder.getUnknownLoc(), builder.getI64Type(), exit.getArgument(0));
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
}

namespace lcl {
std::unique_ptr<Pass> createExpandOpenCLFunctionsPass() {
  return std::make_unique<ExpandOpenCLFunctionsPass>();
}
}
