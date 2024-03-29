//===- StructOps.td - Raw memory dialect ops -----000-------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STRUCT_OPS
#define STRUCT_OPS

include "StructDialect.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
// TODO figure out how to remove this hack
include "../RawMemory/RawMemoryDialect.td"

// Memory-related operations.

// TODO do we need structure allocation?!

def Struct_AddressOfOp: Struct_Op<"address_of"> {
  let arguments = (ins Arg<Struct_AnyStruct, "", [MemRead]>:$addr,
                   SI32Attr:$index);
  let results = (outs RawMem_AnyPointer:$res);

  let assemblyFormat = "$addr `[` $index `]` attr-dict `:` type($addr) `,` type($res)";
  // let hasVerifier = 1;
  // let hasCanonicalizeMethod = 1;
}

def Struct_LoadOp : Struct_Op<"load"> {
  let arguments = (ins Arg<Struct_AnyStruct, "", [MemRead]>:$addr,
                   SI32Attr:$index);
  let results = (outs AnyType:$res);

  let assemblyFormat = "$addr `[` $index `]` attr-dict `:` type($addr) `,` type($res)";
  // let hasVerifier = 1;
  // let hasCanonicalizeMethod = 1;
}

def Struct_StoreOp : Struct_Op<"store"> {
  let arguments = (ins AnyType:$value,
                   Arg<Struct_AnyStruct, "", [MemWrite]>:$addr,
                   SI32Attr:$index);

  let assemblyFormat = "$value `,` $addr `[` $index `]` attr-dict `:` type($value) `,` type($addr)";
  // let hasVerifier = 1;
  // let hasCanonicalizeMethod = 1;
}

#endif // STRUCT_OPS

