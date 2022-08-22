//===- LibreCLOps.td - LibreCL dialect ops -----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIBRECL_OPS
#define LIBRECL_OPS

include "LibreCLDialect.td"

def LibreCL_AnyCastOp : LibreCL_Op<"any_cast"> {
  // TODO restrict types
  let arguments = (ins AnyType:$val);
  let results = (outs AnyType:$res);

  let assemblyFormat = "attr-dict $val `:` type($val) `->` type($res)";
}

def LibreCL_AssumeOp : LibreCL_Op<"assume"> {
  // TODO restrict types
  let arguments = (ins AnyI1:$val);
  let results = (outs);

  let assemblyFormat = "attr-dict $val `:` type($val)";
}

#endif // LIBRECL_OPS

