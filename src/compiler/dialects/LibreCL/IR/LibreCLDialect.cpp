//===- LibreCLDialect.cpp - LibreCL dialect ---------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LibreCLDialect.h"
#include "LibreCLOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace lcl;

#include "LibreCL/IR/LibreCLOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// LibreCL dialect.
//===----------------------------------------------------------------------===//

void LibreCLDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "LibreCL/IR/LibreCLOps.cpp.inc"
      >();
}
