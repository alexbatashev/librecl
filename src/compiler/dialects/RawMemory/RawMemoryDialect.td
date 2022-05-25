//===-- RawMemoryDialect.td - Raw memory dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef RAWMEMORY_DIALECT
#define RAWMEMORY_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Raw memory dialect definition.
//===----------------------------------------------------------------------===//

def RawMemory_Dialect : Dialect {
    let name = "rawmem";
    let summary = "Raw memory manipulation dialect.";
    let description = [{
      TODO description
    }];
    let cppNamespace = "::mlir::rawmem";
}

// Type constraint accepting any pointer type.
def RawMem_AnyPointer : Type<CPred<"$_self.isa<::mlir::rawmem::PointerType>()">,
                                 "Pointer type", "::mlir::rawmem::PointerType">;

//===----------------------------------------------------------------------===//
// Base raw memory operation definition.
//===----------------------------------------------------------------------===//

class RawMemory_Op<string mnemonic, list<Trait> traits = []> :
        Op<RawMemory_Dialect, mnemonic, traits>;

#endif // RAWMEMORY_DIALECT
