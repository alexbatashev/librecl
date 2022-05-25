//===- StandaloneOps.td - Standalone dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef RAWMEMORY_OPS
#define RAWMEMORY_OPS

include "RawMemoryDialect.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Memory-related operations.
def RawMemory_AllocaOp : RawMemory_Op<"alloca"> {
  let arguments = (ins AnyInteger:$arraySize,
                   OptionalAttr<I64Attr>:$alignment,
                   OptionalAttr<TypeAttr>:$elem_type);
  let results = (outs Res<RawMem_AnyPointer, "",
                          [MemAlloc<AutomaticAllocationScopeResource>]>:$res);

  let builders = [
    OpBuilder<(ins "Type":$resultType, "Value":$arraySize,
               "unsigned":$alignment),
    [{
      assert(!resultType.cast<PointerType>().isOpaque() &&
             "pass the allocated type explicitly if opaque pointers are used");
      if (alignment == 0)
        return build($_builder, $_state, resultType, arraySize, IntegerAttr(),
                     TypeAttr());
      build($_builder, $_state, resultType, arraySize,
        $_builder.getI64IntegerAttr(alignment), TypeAttr());
    }]>,
    OpBuilder<(ins "Type":$resultType, "Type":$elementType, "Value":$arraySize,
               CArg<"unsigned", "0">:$alignment),
    [{
      TypeAttr elemTypeAttr =
          resultType.cast<PointerType>().isOpaque() ?
          TypeAttr::get(elementType) : TypeAttr();
      build($_builder, $_state, resultType, arraySize,
            alignment == 0 ? IntegerAttr()
                           : $_builder.getI64IntegerAttr(alignment),
            elemTypeAttr);

    }]>
    ];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}
#endif // RAWMEMORY_OPS
