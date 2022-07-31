//===- StructTypes.cpp - Struct dialect types -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "StructTypes.h"

using namespace mlir;
using namespace mlir::structure;

#include "Struct/StructOpsTypes.cpp.inc"

constexpr const static unsigned kBitsInByte = 8;

bool StructType::isValidElementType(Type type) {
  // TODO figure out if we need this func
  return true;
  /*
  return !type.isa<LLVMVoidType, LLVMLabelType, LLVMMetadataType,
                   LLVMFunctionType, LLVMTokenType, LLVMScalableVectorType>();
                   */
}

StructType StructType::getIdentified(MLIRContext *context, StringRef name) {
  return Base::get(context, name, /*opaque=*/false);
}

StructType
StructType::getIdentifiedChecked(function_ref<InFlightDiagnostic()> emitError,
                                 MLIRContext *context, StringRef name) {
  return Base::getChecked(emitError, context, name, /*opaque=*/false);
}

StructType StructType::getNewIdentified(MLIRContext *context, StringRef name,
                                        ArrayRef<Type> elements,
                                        bool isPacked) {
  std::string stringName = name.str();
  unsigned counter = 0;
  do {
    auto type = StructType::getIdentified(context, stringName);
    if (type.isInitialized() || failed(type.setBody(elements, isPacked))) {
      counter += 1;
      stringName = (Twine(name) + "." + std::to_string(counter)).str();
      continue;
    }
    return type;
  } while (true);
}

StructType StructType::getLiteral(MLIRContext *context, ArrayRef<Type> types,
                                  bool isPacked) {
  return Base::get(context, types, isPacked);
}

StructType
StructType::getLiteralChecked(function_ref<InFlightDiagnostic()> emitError,
                              MLIRContext *context, ArrayRef<Type> types,
                              bool isPacked) {
  return Base::getChecked(emitError, context, types, isPacked);
}

LogicalResult StructType::setBody(ArrayRef<Type> types, bool isPacked) {
  assert(isIdentified() && "can only set bodies of identified structs");
  assert(llvm::all_of(types, StructType::isValidElementType) &&
         "expected valid body types");
  return Base::mutate(types, isPacked);
}

bool StructType::isPacked() const { return getImpl()->isPacked(); }
bool StructType::isIdentified() const { return getImpl()->isIdentified(); }

bool StructType::isInitialized() { return getImpl()->isInitialized(); }
StringRef StructType::getName() { return getImpl()->getIdentifier(); }
ArrayRef<Type> StructType::getBody() const {
  return isIdentified() ? getImpl()->getIdentifiedStructBody()
                        : getImpl()->getTypeList();
}

LogicalResult StructType::verify(function_ref<InFlightDiagnostic()>, StringRef,
                                 bool) {
  return success();
}

LogicalResult StructType::verify(function_ref<InFlightDiagnostic()> emitError,
                                 ArrayRef<Type> types, bool) {
  for (Type t : types)
    if (!isValidElementType(t))
      return emitError() << "invalid structure element type: " << t;

  return success();
}

unsigned StructType::getTypeSizeInBits(const DataLayout &dataLayout,
                                       DataLayoutEntryListRef params) const {
  unsigned structSize = 0;
  unsigned structAlignment = 1;
  for (Type element : getBody()) {
    unsigned elementAlignment =
        isPacked() ? 1 : dataLayout.getTypeABIAlignment(element);
    // Add padding to the struct size to align it to the abi alignment of the
    // element type before than adding the size of the element
    structSize = llvm::alignTo(structSize, elementAlignment);
    structSize += dataLayout.getTypeSize(element);

    // The alignment requirement of a struct is equal to the strictest alignment
    // requirement of its elements.
    structAlignment = std::max(elementAlignment, structAlignment);
  }
  // At the end, add padding to the struct to satisfy its own alignment
  // requirement. Otherwise structs inside of arrays would be misaligned.
  structSize = llvm::alignTo(structSize, structAlignment);
  return structSize * kBitsInByte;
}

namespace {
enum class StructDLEntryPos { Abi = 0, Preferred = 1 };
} // namespace

static Optional<unsigned>
getStructDataLayoutEntry(DataLayoutEntryListRef params, StructType type,
                         StructDLEntryPos pos) {
  const auto *currentEntry =
      llvm::find_if(params, [](DataLayoutEntryInterface entry) {
        return entry.isTypeEntry();
      });
  if (currentEntry == params.end())
    return llvm::None;

  auto attr = currentEntry->getValue().cast<DenseIntElementsAttr>();
  if (pos == StructDLEntryPos::Preferred &&
      attr.size() <= static_cast<unsigned>(StructDLEntryPos::Preferred))
    // If no preferred was specified, fall back to abi alignment
    pos = StructDLEntryPos::Abi;

  return attr.getValues<unsigned>()[static_cast<unsigned>(pos)];
}

static unsigned calculateStructAlignment(const DataLayout &dataLayout,
                                         DataLayoutEntryListRef params,
                                         StructType type,
                                         StructDLEntryPos pos) {
  // Packed structs always have an abi alignment of 1
  if (pos == StructDLEntryPos::Abi && type.isPacked()) {
    return 1;
  }

  // The alignment requirement of a struct is equal to the strictest alignment
  // requirement of its elements.
  unsigned structAlignment = 1;
  for (Type iter : type.getBody()) {
    structAlignment =
        std::max(dataLayout.getTypeABIAlignment(iter), structAlignment);
  }

  // Entries are only allowed to be stricter than the required alignment
  if (Optional<unsigned> entryResult =
          getStructDataLayoutEntry(params, type, pos))
    return std::max(*entryResult / kBitsInByte, structAlignment);

  return structAlignment;
}

unsigned StructType::getABIAlignment(const DataLayout &dataLayout,
                                     DataLayoutEntryListRef params) const {
  return calculateStructAlignment(dataLayout, params, *this,
                                  StructDLEntryPos::Abi);
}

unsigned
StructType::getPreferredAlignment(const DataLayout &dataLayout,
                                  DataLayoutEntryListRef params) const {
  return calculateStructAlignment(dataLayout, params, *this,
                                  StructDLEntryPos::Preferred);
}

static unsigned extractStructSpecValue(Attribute attr, StructDLEntryPos pos) {
  return attr.cast<DenseIntElementsAttr>()
      .getValues<unsigned>()[static_cast<unsigned>(pos)];
}

bool StructType::areCompatible(DataLayoutEntryListRef oldLayout,
                               DataLayoutEntryListRef newLayout) const {
  for (DataLayoutEntryInterface newEntry : newLayout) {
    if (!newEntry.isTypeEntry())
      continue;

    const auto *previousEntry =
        llvm::find_if(oldLayout, [](DataLayoutEntryInterface entry) {
          return entry.isTypeEntry();
        });
    if (previousEntry == oldLayout.end())
      continue;

    unsigned abi = extractStructSpecValue(previousEntry->getValue(),
                                          StructDLEntryPos::Abi);
    unsigned newAbi =
        extractStructSpecValue(newEntry.getValue(), StructDLEntryPos::Abi);
    if (abi < newAbi || abi % newAbi != 0)
      return false;
  }
  return true;
}

LogicalResult StructType::verifyEntries(DataLayoutEntryListRef entries,
                                        Location loc) const {
  for (DataLayoutEntryInterface entry : entries) {
    if (!entry.isTypeEntry())
      continue;

    auto key = entry.getKey().get<Type>().cast<StructType>();
    auto values = entry.getValue().dyn_cast<DenseIntElementsAttr>();
    if (!values || (values.size() != 2 && values.size() != 1)) {
      return emitError(loc)
             << "expected layout attribute for " << entry.getKey().get<Type>()
             << " to be a dense integer elements attribute of 1 or 2 elements";
    }

    if (key.isIdentified() || !key.getBody().empty()) {
      return emitError(loc) << "unexpected layout attribute for struct " << key;
    }

    if (values.size() == 1)
      continue;

    if (extractStructSpecValue(values, StructDLEntryPos::Abi) >
        extractStructSpecValue(values, StructDLEntryPos::Preferred)) {
      return emitError(loc) << "preferred alignment is expected to be at least "
                               "as large as ABI alignment";
    }
  }
  return mlir::success();
}
