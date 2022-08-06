//===- RawMemoryDialect.cpp - Raw Memory dialect ----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RawMemoryDialect.h"
#include "../Struct/StructTypes.h"
#include "RawMemoryOps.h"
#include "RawMemoryTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/TypeSwitch.h"

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
  addTypes<PointerType>();
}

static void dispatchPrint(AsmPrinter &printer, Type type);
static Type dispatchParse(AsmParser &parser, bool allowAny = true);
static ParseResult dispatchParse(AsmParser &parser, Type &type);

static StringRef getTypeKeyword(Type type) {
  return TypeSwitch<Type, StringRef>(type)
      .Case<rawmem::PointerType>([&](Type) { return "ptr"; })
      .Default([](Type) -> StringRef {
        llvm_unreachable("unexpected 'rawmem' type kind");
      });
}

namespace mlir {
namespace rawmem {
namespace detail {
static void printType(Type type, AsmPrinter &printer) {
  if (!type)
    printer << "<<NULL-TYPE>>";

  printer << getTypeKeyword(type);

  if (auto ptrType = type.dyn_cast<PointerType>()) {
    if (ptrType.isOpaque()) {
      if (ptrType.getAddressSpace() != 0)
        printer << '<' << ptrType.getAddressSpace() << '>';
      return;
    }

    printer << '<';
    dispatchPrint(printer, ptrType.getElementType());
    if (ptrType.getAddressSpace() != 0)
      printer << ", " << ptrType.getAddressSpace();
    printer << '>';
    return;
  }
}
static Type parseType(DialectAsmParser &parser) {
  SMLoc loc = parser.getCurrentLocation();
  Type type = dispatchParse(parser, /*allowAny=*/false);
  if (!type)
    return type;
  return type;
}
} // namespace detail
} // namespace rawmem
} // namespace mlir

/// Parse a type registered to this dialect.
Type RawMemoryDialect::parseType(DialectAsmParser &parser) const {
  return detail::parseType(parser);
}

void dispatchPrint(AsmPrinter &printer, Type type) {
  if (!type.isa<IntegerType, FloatType, VectorType, structure::StructType>())
    return mlir::rawmem::detail::printType(type, printer);
  printer.printType(type);
}

static PointerType parsePointerType(AsmParser &parser) {
  SMLoc loc = parser.getCurrentLocation();
  Type elementType;
  if (parser.parseOptionalLess()) {
    return parser.getChecked<PointerType>(loc, parser.getContext(),
                                          /*addressSpace=*/0);
  }

  unsigned addressSpace = 0;
  OptionalParseResult opr = parser.parseOptionalInteger(addressSpace);
  if (opr.hasValue()) {
    if (failed(*opr) || parser.parseGreater())
      return PointerType();
    return parser.getChecked<PointerType>(loc, parser.getContext(),
                                          addressSpace);
  }

  if (dispatchParse(parser, elementType))
    return PointerType();

  if (succeeded(parser.parseOptionalComma()) &&
      failed(parser.parseInteger(addressSpace)))
    return PointerType();
  if (failed(parser.parseGreater()))
    return PointerType();
  return parser.getChecked<PointerType>(loc, elementType, addressSpace);
}

static Type dispatchParse(AsmParser &parser, bool allowAny) {
  SMLoc keyLoc = parser.getCurrentLocation();

  // Try parsing any MLIR type.
  Type type;
  OptionalParseResult result = parser.parseOptionalType(type);
  if (result.hasValue()) {
    if (failed(result.getValue()))
      return nullptr;
    if (!allowAny) {
      parser.emitError(keyLoc) << "unexpected type, expected keyword";
      return nullptr;
    }
    return type;
  }

  // If no type found, fallback to the shorthand form.
  StringRef key;
  if (failed(parser.parseKeyword(&key)))
    return Type();

  MLIRContext *ctx = parser.getContext();
  return StringSwitch<function_ref<Type()>>(key)
      .Case("ptr", [&] { return parsePointerType(parser); })
      .Default([&] {
        parser.emitError(keyLoc) << "unknown Raw Memory type: " << key;
        return Type();
      })();
}

/// Helper to use in parse lists.
static ParseResult dispatchParse(AsmParser &parser, Type &type) {
  type = dispatchParse(parser, true);
  return success(type != nullptr);
}

/// Print a type registered to this dialect.
void RawMemoryDialect::printType(Type type, DialectAsmPrinter &os) const {
  detail::printType(type, os);
}
