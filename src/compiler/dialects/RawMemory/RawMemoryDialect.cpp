//===- RawMemoryDialect.cpp - Raw Memory dialect ----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RawMemoryDialect.h"
#include "RawMemoryOps.h"

using namespace mlir;
using namespace mlir::rawmem;

#include "RawMemory/RawMemoryOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// RawMemory dialect.
//===----------------------------------------------------------------------===//

void RawMemoryDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "RawMemory/RawMemoryOps.cpp.inc"
      >();
}
