//===- StructDialect.cpp - Struct dialect -----------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "StructDialect.h"
#include "StructOps.h"
#include "StructTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"

#include <type_traits>
#include <utility>

using namespace mlir;
using namespace mlir::structure;

#include "Struct/StructOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Struct dialect.
//===----------------------------------------------------------------------===//

void StructDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Struct/StructOps.cpp.inc"
      >();
  addTypes<StructType>();
}

static void dispatchPrint(AsmPrinter &printer, Type type);
static Type dispatchParse(AsmParser &parser, bool allowAny = true);
static ParseResult dispatchParse(AsmParser &parser, Type &type);

static StringRef getTypeKeyword(Type type) {
  return TypeSwitch<Type, StringRef>(type)
      .Case<structure::StructType>([&](Type) { return "struct"; })
      .Default([](Type) -> StringRef {
        llvm_unreachable("unexpected 'llvm' type kind");
      });
}

namespace mlir {
namespace structure {
namespace detail {
static void printType(Type genericType, AsmPrinter &printer) {
  StructType type = genericType.cast<StructType>();

  if (!type) {
    printer << "<<NULL-TYPE>>";
    return;
  }

  thread_local SetVector<StringRef> knownStructNames;
  unsigned stackSize = knownStructNames.size();
  (void)stackSize;
  auto guard = llvm::make_scope_exit([&]() {
    assert(knownStructNames.size() == stackSize &&
           "malformed identified stack when printing recursive structs");
  });

  printer << "<";
  if (type.isIdentified()) {
    printer << '"' << type.getName() << '"';
    // If we are printing a reference to one of the enclosing structs, just
    // print the name and stop to avoid infinitely long output.
    if (knownStructNames.count(type.getName())) {
      printer << '>';
      return;
    }
    printer << ", ";
  }

  /*
  if (type.isIdentified() && type.isOpaque()) {
    printer << "opaque>";
    return;
  }
  */

  if (type.isPacked())
    printer << "packed ";

  // Put the current type on stack to avoid infinite recursion.
  printer << '(';
  if (type.isIdentified())
    knownStructNames.insert(type.getName());
  llvm::interleaveComma(type.getBody(), printer.getStream(),
                        [&](Type subtype) { dispatchPrint(printer, subtype); });
  if (type.isIdentified())
    knownStructNames.pop_back();
  printer << ')';
  printer << '>';
}

static Type parseType(DialectAsmParser &parser) {
  // SMLoc loc = parser.getCurrentLocation();
  Type type = dispatchParse(parser, /*allowAny=*/false);
  if (!type)
    return type;
  return type;
}
} // namespace detail
} // namespace structure
} // namespace mlir

/// Parse a type registered to this dialect.
Type StructDialect::parseType(DialectAsmParser &parser) const {
  return detail::parseType(parser);
}

void dispatchPrint(AsmPrinter &printer, Type type) {
  if (!type.isa<IntegerType, FloatType, VectorType>())
    return mlir::structure::detail::printType(type, printer);
  printer.printType(type);
}

static StructType trySetStructBody(StructType type, ArrayRef<Type> subtypes,
                                   bool isPacked, AsmParser &parser,
                                   SMLoc subtypesLoc) {
  for (Type t : subtypes) {
    if (!StructType::isValidElementType(t)) {
      parser.emitError(subtypesLoc) << "invalid structure element type: " << t;
      return StructType();
    }
  }

  if (succeeded(type.setBody(subtypes, isPacked)))
    return type;

  parser.emitError(subtypesLoc)
      << "identified type already used with a different body";
  return StructType();
}

static StructType parseStructType(AsmParser &parser) {
  thread_local SetVector<StringRef> knownStructNames;
  unsigned stackSize = knownStructNames.size();
  (void)stackSize;
  auto guard = llvm::make_scope_exit([&]() {
    assert(knownStructNames.size() == stackSize &&
           "malformed identified stack when parsing recursive structs");
  });

  Location loc = parser.getEncodedSourceLoc(parser.getCurrentLocation());

  if (failed(parser.parseLess()))
    return StructType();

  // If we are parsing a self-reference to a recursive struct, i.e. the parsing
  // stack already contains a struct with the same identifier, bail out after
  // the name.
  std::string name;
  bool isIdentified = succeeded(parser.parseOptionalString(&name));
  if (isIdentified) {
    if (knownStructNames.count(name)) {
      if (failed(parser.parseGreater()))
        return StructType();
      return StructType::getIdentifiedChecked([loc] { return emitError(loc); },
                                              loc.getContext(), name);
    }
    if (failed(parser.parseComma()))
      return StructType();
  }

  // Handle intentionally opaque structs.
  SMLoc kwLoc = parser.getCurrentLocation();
  /*
  if (succeeded(parser.parseOptionalKeyword("opaque"))) {
    if (!isIdentified)
      return parser.emitError(kwLoc, "only identified structs can be opaque"),
             LLVMStructType();
    if (failed(parser.parseGreater()))
      return LLVMStructType();
    auto type = LLVMStructType::getOpaqueChecked(
        [loc] { return emitError(loc); }, loc.getContext(), name);
    if (!type.isOpaque()) {
      parser.emitError(kwLoc, "redeclaring defined struct as opaque");
      return LLVMStructType();
    }
    return type;
  }
  */

  // Check for packedness.
  bool isPacked = succeeded(parser.parseOptionalKeyword("packed"));
  if (failed(parser.parseLParen()))
    return StructType();

  // Fast pass for structs with zero subtypes.
  if (succeeded(parser.parseOptionalRParen())) {
    if (failed(parser.parseGreater()))
      return StructType();
    if (!isIdentified)
      return StructType::getLiteralChecked([loc] { return emitError(loc); },
                                           loc.getContext(), {}, isPacked);
    auto type = StructType::getIdentifiedChecked(
        [loc] { return emitError(loc); }, loc.getContext(), name);
    return trySetStructBody(type, {}, isPacked, parser, kwLoc);
  }

  // Parse subtypes. For identified structs, put the identifier of the struct on
  // the stack to support self-references in the recursive calls.
  SmallVector<Type, 4> subtypes;
  SMLoc subtypesLoc = parser.getCurrentLocation();
  do {
    if (isIdentified)
      knownStructNames.insert(name);
    Type type;
    if (dispatchParse(parser, type))
      return StructType();
    subtypes.push_back(type);
    if (isIdentified)
      knownStructNames.pop_back();
  } while (succeeded(parser.parseOptionalComma()));

  if (parser.parseRParen() || parser.parseGreater())
    return StructType();

  // Construct the struct with body.
  if (!isIdentified)
    return StructType::getLiteralChecked([loc] { return emitError(loc); },
                                         loc.getContext(), subtypes, isPacked);
  auto type = StructType::getIdentifiedChecked([loc] { return emitError(loc); },
                                               loc.getContext(), name);
  return trySetStructBody(type, subtypes, isPacked, parser, subtypesLoc);
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

  // MLIRContext *ctx = parser.getContext();
  return StringSwitch<function_ref<Type()>>(key)
      .Case("struct", [&] { return parseStructType(parser); })
      .Default([&] {
        parser.emitError(keyLoc) << "unknown Struct type: " << key;
        return Type();
      })();
}

/// Helper to use in parse lists.
static ParseResult dispatchParse(AsmParser &parser, Type &type) {
  type = dispatchParse(parser, true);
  return success(type != nullptr);
}

/// Print a type registered to this dialect.
void StructDialect::printType(Type type, DialectAsmPrinter &os) const {
  detail::printType(type, os);
}
