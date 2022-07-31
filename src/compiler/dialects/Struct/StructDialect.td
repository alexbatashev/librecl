//===-- StructDialect.td - Struct dialect ------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STRUCT_DIALECT
#define STRUCT_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Struct dialect definition.
//===----------------------------------------------------------------------===//

def Struct_Dialect : Dialect {
  let name = "struct";
  let summary = "Data structure manipulation dialect.";
  let description = [{
    TODO description
  }];

  let useDefaultTypePrinterParser = 1;

  let cppNamespace = "::mlir::structure";
}

// Type constraint accepting any pointer type.
def Struct_AnyStruct : Type<CPred<"$_self.isa<::mlir::structure::StructType>()">,
                                 "Data structure type", "::mlir::structure::StructType">;

//===----------------------------------------------------------------------===//
// Base struct operation definition.
//===----------------------------------------------------------------------===//

class Struct_Op<string mnemonic, list<Trait> traits = []> :
        Op<Struct_Dialect, mnemonic, traits>;

#endif // STRUCT_DIALECT
