//===- StructureCFGPass.cpp -------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "passes.hpp"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVEnums.h>

using namespace mlir;

namespace {
static spirv::BranchConditionalOp findInnermostIf(Operation *op) {
  PostDominanceInfo info{op};

  spirv::BranchConditionalOp result = nullptr;
  op->walk([&result, &info](spirv::BranchConditionalOp branch) {
    if (branch->getParentOfType<spirv::SelectionOp>() ||
        branch->getParentOfType<spirv::LoopOp>()) {
      return WalkResult::skip();
    }
    if (!info.postDominates(branch.falseTarget(), branch.trueTarget())) {
      return WalkResult::advance();
    }

    // TODO strictly speaking this is not innermost, but I don't care for now.
    result = branch;

    return WalkResult::interrupt();
  });

  return result;
}

struct StructureCFGPass
    : public PassWrapper<StructureCFGPass, OperationPass<spirv::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StructureCFGPass);

  void runOnOperation() override {
    spirv::ModuleOp module = getOperation();

    OpBuilder builder{&getContext()};

    while (auto cond = findInnermostIf(module)) {
      builder.setInsertionPoint(cond);
      auto selection = builder.create<spirv::SelectionOp>(
          cond.getLoc(), spirv::SelectionControl::None);
      auto header = builder.createBlock(&selection.body());
      auto body = builder.createBlock(&selection.body());
      auto merge = builder.createBlock(&selection.body());

      builder.setInsertionPointToStart(header);
      builder.create<spirv::BranchConditionalOp>(
          cond->getLoc(), cond.condition(), body, cond.trueTargetOperands(),
          merge, ValueRange{});
      builder.setInsertionPointToStart(body);
      BlockAndValueMapping mapping;
      for (Operation &op : *cond.trueTarget()) {
        if (&op != cond.trueTarget()->getTerminator()) {
          Operation *clone = builder.clone(op, mapping);
          for (auto res : llvm::enumerate(op.getResults())) {
            mapping.map(res.value(), clone->getResult(res.index()));
          }
        }
      }
      builder.create<spirv::BranchOp>(cond.getLoc(), merge, ValueRange{});
      builder.setInsertionPointToStart(merge);
      builder.create<spirv::MergeOp>(cond.getLoc());

      builder.setInsertionPoint(cond);
      builder.create<spirv::BranchOp>(cond.getLoc(), cond.falseTarget(),
                                      ValueRange());
      Block *trueTarget = cond.trueTarget();
      cond.erase();
      trueTarget->erase();
    }
  }
};
} // namespace

namespace lcl {
std::unique_ptr<Pass> createStructureCFGPass() {
  return std::make_unique<StructureCFGPass>();
}
} // namespace lcl
