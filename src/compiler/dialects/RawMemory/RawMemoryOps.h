//===- RawMemoryOps.h - RawMemory dialect ops -------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef RAWMEMORY_RAWMEMORYOPS_H
#define RAWMEMORY_RAWMEMORYOPS_H

#include "RawMemoryTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "RawMemory/RawMemoryOps.h.inc"

#endif // RAWMEMORY_RAWMEMORYOPS_H
