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

def RawMemory_LoadOp : RawMemory_Op<"load"> {
  let arguments = (ins Arg<RawMem_AnyPointer, "", [MemRead]>:$addr,
                   Variadic<Index>:$indices, UnitAttr:$volatility);
  let results = (outs AnyType:$res);

  let builders = [
    OpBuilder<(ins "Value":$addr, CArg<"ValueRange", "{}">:$indices,
        CArg<"bool","false">:$isVolatile), [{
      $_state.addOperands(addr);
      if (indices.size())
        $_state.addOperands(indices);

      auto type = addr.getType().cast<PointerType>().getElementType();
      $_state.addTypes(type);
    }]>,
  ];

  // TODO volatile
  let assemblyFormat = "(`volatile` $volatility^)? $addr (`[` $indices^ `]`)? attr-dict `:` type($addr) `,` type($res)";
  let hasVerifier = 1;
}

def RawMemory_StoreOp : RawMemory_Op<"store"> {
  let arguments = (ins AnyType:$value,
                   Arg<RawMem_AnyPointer, "", [MemWrite]>:$addr,
                   Variadic<Index>:$indices, UnitAttr:$volatility);

  /*
  let builders = [
    OpBuilder<(ins "Value":$value, "Value":$addr, "ValueRange":$indices,
        CArg<"bool","false">:$isVolatile), [{
      $_state.addOperands(value);
      $_state.addOperands(addr);
      $_state.addOperands(indices);
      if (isVolatile)
        $_state.addAttribute(getVolatileAttrName(), $_builder.getUnitAttr());
    }]>,
  ];
  */
  let extraClassDeclaration = [{
    static StringRef getVolatileAttrName() { return "volatility"; }
  }];

  let assemblyFormat = [{
    (`volatile` $volatility^)? $value `,` $addr (`[` $indices^ `]`)? attr-dict
      `:` type($value) `,` type($addr)
  }];
  let hasVerifier = 1;
}

def RawMemory_ReinterpretCastOp : RawMemory_Op<"reinterpret_cast"> {
  let arguments = (ins RawMem_AnyPointer:$addr);
  let results = (outs RawMem_AnyPointer:$out);

  let assemblyFormat = [{
    $addr attr-dict type($addr) `to` type($out)
  }];
}

def RawMemory_OffsetOp : RawMemory_Op<"offset", [
  TypesMatchWith<"result types matches with base addres", "addr", "out",
  "$_self">
]> {
  let arguments = (ins RawMem_AnyPointer:$addr, Index:$offset);
  let results = (outs RawMem_AnyPointer:$out);

  let assemblyFormat = [{
    $addr `[` $offset `]` attr-dict `:` type($addr)
  }];
}

// TODO specify more constraints
/*
def RawMemory_LoadOp : RawMemory_Op<"load"> {
  let arguments = (ins Arg<RawMem_AnyPointer, "", [MemRead]>:$addr,
                   OptionalAttr<SymbolRefArrayAttr>:$access_groups,
                   OptionalAttr<SymbolRefArrayAttr>:$alias_scopes,
                   OptionalAttr<SymbolRefArrayAttr>:$noalias_scopes,
                   OptionalAttr<I64Attr>:$alignment, UnitAttr:$volatile_,
                   UnitAttr:$nontemporal);
  let results = (outs AnyType:$res);
  let builders = [
    OpBuilder<(ins "Value":$addr, CArg<"unsigned", "0">:$alignment,
      CArg<"bool", "false">:$isVolatile, CArg<"bool", "false">:$isNonTemporal),
    [{
      auto type = addr.getType().cast<PointerType>().getElementType();
      assert(type && "must provide explicit element type to the constructor "
                     "when the pointer type is opaque");
      build($_builder, $_state, type, addr, alignment, isVolatile, isNonTemporal);
    }]>,
    OpBuilder<(ins "Type":$t, "Value":$addr,
      CArg<"unsigned", "0">:$alignment, CArg<"bool", "false">:$isVolatile,
      CArg<"bool", "false">:$isNonTemporal)>,];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def RawMemory_StoreOp : RawMemory_Op<"store"> {
  let arguments = (ins AnyType:$value,
                   Arg<RawMem_AnyPointer,"",[MemWrite]>:$addr,
                   OptionalAttr<SymbolRefArrayAttr>:$access_groups,
                   OptionalAttr<SymbolRefArrayAttr>:$alias_scopes,
                   OptionalAttr<SymbolRefArrayAttr>:$noalias_scopes,
                   OptionalAttr<I64Attr>:$alignment, UnitAttr:$volatile_,
                   UnitAttr:$nontemporal);
  let builders = [
    OpBuilder<(ins "Value":$value, "Value":$addr,
      CArg<"unsigned", "0">:$alignment, CArg<"bool", "false">:$isVolatile,
      CArg<"bool", "false">:$isNonTemporal)>
    ];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}
*/
#endif // RAWMEMORY_OPS
