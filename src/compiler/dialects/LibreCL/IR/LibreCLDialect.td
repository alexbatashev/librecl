//===-- LibreCLDialect.td - LibreCL dialect ----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIBRECL_DIALECT
#define LIBRECL_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// LibreCL dialect definition.
//===----------------------------------------------------------------------===//

def LibreCL_Dialect : Dialect {
  let name = "lcl";
  let summary = "LibreCL additional ops dialect.";
  let description = [{
    TODO description
  }];

  // let useDefaultTypePrinterParser = 1;

  let cppNamespace = "::lcl";
}

//===----------------------------------------------------------------------===//
// Base LibreCL operation definition.
//===----------------------------------------------------------------------===//

class LibreCL_Op<string mnemonic, list<Trait> traits = []> :
        Op<LibreCL_Dialect, mnemonic, traits>;

#endif // RAWMEMORY_DIALECT

