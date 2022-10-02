//===- RawMemoryOps.cpp - RawMemory dialect ops -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RawMemoryOps.h"
#include "RawMemoryDialect.h"
// #include "Struct/StructDialect.h"
#include "Struct/StructOps.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

#define GET_OP_CLASSES
#include "RawMemory/RawMemoryOps.cpp.inc"

using namespace mlir;
using namespace mlir::rawmem;

// TODO get rid of LLVM stuff
static constexpr const char kElemTypeAttrName[] = "elem_type";

//===----------------------------------------------------------------------===//
// Printing, parsing and verification for rawmem::AllocaOp.
//===----------------------------------------------------------------------===//

void AllocaOp::print(OpAsmPrinter &p) {
  Type elemTy = getType().cast<rawmem::PointerType>().getElementType();
  if (!elemTy)
    elemTy = *getElemType();

  auto funcTy =
      FunctionType::get(getContext(), {getArraySize().getType()}, {getType()});

  p << ' ' << getArraySize() << " x " << elemTy;
  if (getAlignment().hasValue() && *getAlignment() != 0)
    p.printOptionalAttrDict((*this)->getAttrs(), {kElemTypeAttrName});
  else
    p.printOptionalAttrDict((*this)->getAttrs(),
                            {"alignment", kElemTypeAttrName});
  p << " : " << funcTy;
}

// <operation> ::= `llvm.alloca` ssa-use `x` type attribute-dict?
//                 `:` type `,` type
ParseResult AllocaOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand arraySize;
  Type type, elemType;
  SMLoc trailingTypeLoc;
  if (parser.parseOperand(arraySize) || parser.parseKeyword("x") ||
      parser.parseType(elemType) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.getCurrentLocation(&trailingTypeLoc) || parser.parseType(type))
    return failure();

  Optional<NamedAttribute> alignmentAttr =
      result.attributes.getNamed("alignment");
  if (alignmentAttr.hasValue()) {
    auto alignmentInt =
        alignmentAttr.getValue().getValue().dyn_cast<IntegerAttr>();
    if (!alignmentInt)
      return parser.emitError(parser.getNameLoc(),
                              "expected integer alignment");
    if (alignmentInt.getValue().isNullValue())
      result.attributes.erase("alignment");
  }

  // Extract the result type from the trailing function type.
  auto funcType = type.dyn_cast<FunctionType>();
  if (!funcType || funcType.getNumInputs() != 1 ||
      funcType.getNumResults() != 1)
    return parser.emitError(
        trailingTypeLoc,
        "expected trailing function type with one argument and one result");

  if (parser.resolveOperand(arraySize, funcType.getInput(0), result.operands))
    return failure();

  Type resultType = funcType.getResult(0);
  if (auto ptrResultType = resultType.dyn_cast<PointerType>()) {
    if (ptrResultType.isOpaque())
      result.addAttribute(kElemTypeAttrName, TypeAttr::get(elemType));
  }

  result.addTypes({funcType.getResult(0)});
  return success();
}

/// Checks that the elemental type is present in either the pointer type or
/// the attribute, but not both.
static LogicalResult verifyOpaquePtr(Operation *op, PointerType ptrType,
                                     Optional<Type> ptrElementType) {
  if (ptrType.isOpaque() && !ptrElementType.hasValue()) {
    return op->emitOpError() << "expected '" << kElemTypeAttrName
                             << "' attribute if opaque pointer type is used";
  }
  if (!ptrType.isOpaque() && ptrElementType.hasValue()) {
    return op->emitOpError()
           << "unexpected '" << kElemTypeAttrName
           << "' attribute when non-opaque pointer type is used";
  }
  return success();
}

LogicalResult AllocaOp::verify() {
  return verifyOpaquePtr(getOperation(), getType().cast<PointerType>(),
                         getElemType());
}

//===----------------------------------------------------------------------===//
// Verification for rawmem::LoadOp.
//===----------------------------------------------------------------------===//
LogicalResult LoadOp::verify() {
  // TODO real verifier
  return success();
}

//===----------------------------------------------------------------------===//
// Verification for rawmem::StoreOp.
//===----------------------------------------------------------------------===//
LogicalResult StoreOp::verify() {
  // TODO real verifier
  return success();
}

OpFoldResult ReinterpretCastOp::fold(ArrayRef<Attribute> operands) {
  if (getAddr().getDefiningOp() &&
      llvm::isa<ReinterpretCastOp>(getAddr().getDefiningOp())) {
    auto otherCast = getAddr().getDefiningOp<ReinterpretCastOp>();
    if (otherCast.getAddr().getType() == getResult().getType()) {
      return otherCast.getAddr();
    }
  }
  return nullptr;
}

LogicalResult ReinterpretCastOp::canonicalize(ReinterpretCastOp op,
                                              PatternRewriter &rewriter) {
  if (op.getResult().user_begin() == op.getResult().user_end()) {
    rewriter.eraseOp(op);
    return success();
  }
  return failure();
}

LogicalResult LoadOp::canonicalize(LoadOp op, PatternRewriter &rewriter) {
  if (op.getAddr().getDefiningOp() &&
      llvm::isa<OffsetOp>(op.getAddr().getDefiningOp())) {
    auto offsetOp = op.getAddr().getDefiningOp<OffsetOp>();
    rewriter.replaceOpWithNewOp<LoadOp>(op, offsetOp.getAddr(),
                                        ValueRange{offsetOp.getOffset()});
    return success();
  }
  if (op.getAddr().getDefiningOp() &&
      llvm::isa<structure::AddressOfOp>(op.getAddr().getDefiningOp())) {
    auto addrOp = op.getAddr().getDefiningOp<structure::AddressOfOp>();
    rewriter.replaceOpWithNewOp<structure::LoadOp>(
        op, op.getResult().getType(), addrOp.getAddr(), addrOp.getIndex());
    return success();
  }
  return failure();
}

LogicalResult StoreOp::canonicalize(StoreOp op, PatternRewriter &rewriter) {
  if (op.getAddr().getDefiningOp() &&
      llvm::isa<OffsetOp>(op.getAddr().getDefiningOp())) {
    auto offsetOp = op.getAddr().getDefiningOp<OffsetOp>();
    rewriter.replaceOpWithNewOp<StoreOp>(op, op.getValue(), offsetOp.getAddr(),
                                         ValueRange{offsetOp.getOffset()},
                                         op.getVolatility());
    return success();
  }
  if (op.getAddr().getDefiningOp() &&
      llvm::isa<structure::AddressOfOp>(op.getAddr().getDefiningOp())) {
    auto addrOp = op.getAddr().getDefiningOp<structure::AddressOfOp>();
    rewriter.replaceOpWithNewOp<structure::StoreOp>(
        op, op.getValue(), addrOp.getAddr(), addrOp.getIndex());
    return success();
  }
  return failure();
}

LogicalResult OffsetOp::canonicalize(OffsetOp op, PatternRewriter &rewriter) {
  if (op.getResult().user_begin() == op.getResult().user_end()) {
    rewriter.eraseOp(op);
    return success();
  }
  return failure();
}
