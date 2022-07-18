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
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/SCF.h"

using namespace mlir;

namespace {
static cf::CondBranchOp findInnermostIf(Operation *op) {
  PostDominanceInfo info{op};

  cf::CondBranchOp result = nullptr;
  op->walk([&result, &info](cf::CondBranchOp branch) {
    /*
    if (branch->getParentOfType<spirv::SelectionOp>() ||
        branch->getParentOfType<spirv::LoopOp>()) {
      return WalkResult::skip();
    }
    */
    // Skip if-else if blocks
    if (!llvm::isa<cf::BranchOp>(branch.getTrueDest()->getTerminator())) {
      return WalkResult::skip();
    }
    if (!info.postDominates(branch.getFalseDest(), branch.getTrueDest())) {
      return WalkResult::advance();
    }

    // TODO strictly speaking this is not innermost, but I don't care for now.
    result = branch;

    return WalkResult::interrupt();
  });

  return result;
}

struct StructureCFGPass
    : public PassWrapper<StructureCFGPass, OperationPass<gpu::GPUModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StructureCFGPass);

  static constexpr llvm::StringLiteral getArgumentName() {
    return llvm::StringLiteral("structure-cfg");
  }
  llvm::StringRef getArgument() const override { return "structure-cfg"; }

  llvm::StringRef getDescription() const override {
    return "Attempt to convert Control Flow operations to their structured counterparts";
  }

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect>();
    registry.insert<cf::ControlFlowDialect>();
  }

  void runOnOperation() override {
    gpu::GPUModuleOp module = getOperation();

    OpBuilder builder{&getContext()};

    while (auto cond = findInnermostIf(module)) {
      // Even though we only consider a simple if-then pattern here,
      // false destination may still have some arguments. Thus we must take them
      // into account by creating an artificial else branch with a single yield
      // operation, that would return our block arguments.
      bool hasElseBlock = !cond.getFalseDestOperands().empty();

      builder.setInsertionPoint(cond);
      auto scfIf = builder.create<scf::IfOp>(cond->getLoc(), cond.getCondition(), hasElseBlock);
      builder.setInsertionPointToStart(&scfIf.getThenRegion().front());

      mlir::BlockAndValueMapping mapping;
      for (mlir::Operation &op : *cond.getTrueDest()) {
        if (!llvm::isa<cf::BranchOp>(op)) {
          auto newOp = builder.clone(op, mapping);
          // mapping.map(op, newOp);
          for (auto res : llvm::zip(op.getResults(), newOp->getResults())) {
            mapping.map(std::get<0>(res), std::get<1>(res));
          }
        }
      }

      SmallVector<Value, 5> results;
      for (auto res : cond.getTrueDest()->getTerminator()->getResults()) {
        results.push_back(mapping.lookupOrNull(res));
      }

      auto yieldOp = builder.create<scf::YieldOp>(cond.getTrueDest()->getTerminator()->getLoc(), results);

      scfIf.getThenRegion().front().back().erase();

      if (hasElseBlock) {
        builder.setInsertionPointToStart(&scfIf.getElseRegion().front());
        scfIf.getElseRegion().front().getTerminator()->erase();
        builder.create<scf::YieldOp>(cond.getLoc(), cond.getFalseDestOperands());
      }

      auto *trueDest = cond.getTrueDest();
      auto *falseDest = cond.getFalseDest();

      cond.erase();
      // We suppose that no other branch points to this block
      assert(trueDest->hasNoPredecessors());
      trueDest->erase();

      builder.setInsertionPointAfter(scfIf);
      builder.createOrFold<cf::BranchOp>(yieldOp.getLoc(), falseDest, scfIf.getResults());
    }
  }
};
} // namespace

namespace lcl {
std::unique_ptr<Pass> createStructureCFGPass() {
  return std::make_unique<StructureCFGPass>();
}
} // namespace lcl
