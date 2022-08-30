//===- RawMemoryTypes.cpp - Raw Memory dialect types ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RawMemoryTypes.h"

using namespace mlir;
using namespace mlir::rawmem;

#include "RawMemory/RawMemoryOpsTypes.cpp.inc"

constexpr const static unsigned kBitsInByte = 8;

bool PointerType::isValidElementType(Type type) {
  // TODO actual diagnostics
  return true;
  // if (!type)
  //  return true;
  // return isCompatibleOuterType(type) type.isa<PointerElementTypeInterface>();
}

PointerType PointerType::get(Type pointee, unsigned addressSpace) {
  assert(pointee && "expected non-null subtype, pass the context instead if "
                    "the opaque pointer type is desired");
  return Base::get(pointee.getContext(), pointee, addressSpace);
}

PointerType PointerType::get(MLIRContext *context, unsigned addressSpace) {
  return Base::get(context, Type(), addressSpace);
}

PointerType
PointerType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                        Type pointee, unsigned addressSpace) {
  return Base::getChecked(emitError, pointee.getContext(), pointee,
                          addressSpace);
}

PointerType
PointerType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                        MLIRContext *context, unsigned addressSpace) {
  return Base::getChecked(emitError, context, Type(), addressSpace);
}

Type PointerType::getElementType() const { return getImpl()->pointeeType; }

bool PointerType::isOpaque() const { return !getImpl()->pointeeType; }

unsigned PointerType::getAddressSpace() const {
  return getImpl()->addressSpace;
}

LogicalResult PointerType::verify(function_ref<InFlightDiagnostic()> emitError,
                                  Type pointee, unsigned) {
  if (!isValidElementType(pointee))
    return emitError() << "invalid pointer element type: " << pointee;
  return success();
}

namespace {
/// The positions of different values in the data layout entry.
enum class DLEntryPos { Size = 0, Abi = 1, Preferred = 2, Address = 3 };
} // namespace

constexpr const static unsigned kDefaultPointerSizeBits = 64;
constexpr const static unsigned kDefaultPointerAlignment = 8;

/// Returns the value that corresponds to named position `pos` from the
/// attribute `attr` assuming it's a dense integer elements attribute.
static unsigned extractPointerSpecValue(Attribute attr, DLEntryPos pos) {
  return attr.cast<DenseIntElementsAttr>()
      .getValues<unsigned>()[static_cast<unsigned>(pos)];
}

/// Returns the part of the data layout entry that corresponds to `pos` for the
/// given `type` by interpreting the list of entries `params`. For the pointer
/// type in the default address space, returns the default value if the entries
/// do not provide a custom one, for other address spaces returns None.
static Optional<unsigned>
getPointerDataLayoutEntry(DataLayoutEntryListRef params, PointerType type,
                          DLEntryPos pos) {
  // First, look for the entry for the pointer in the current address space.
  Attribute currentEntry;
  for (DataLayoutEntryInterface entry : params) {
    if (!entry.isTypeEntry())
      continue;
    if (entry.getKey().get<Type>().cast<PointerType>().getAddressSpace() ==
        type.getAddressSpace()) {
      currentEntry = entry.getValue();
      break;
    }
  }
  if (currentEntry) {
    return extractPointerSpecValue(currentEntry, pos) /
           (pos == DLEntryPos::Size ? 1 : kBitsInByte);
  }

  // If not found, and this is the pointer to the default memory space, assume
  // 64-bit pointers.
  if (type.getAddressSpace() == 0) {
    return pos == DLEntryPos::Size ? kDefaultPointerSizeBits
                                   : kDefaultPointerAlignment;
  }

  return llvm::None;
}

unsigned PointerType::getTypeSizeInBits(const DataLayout &dataLayout,
                                        DataLayoutEntryListRef params) const {
  if (Optional<unsigned> size =
          getPointerDataLayoutEntry(params, *this, DLEntryPos::Size))
    return *size;

  // For other memory spaces, use the size of the pointer to the default memory
  // space.
  if (isOpaque())
    return dataLayout.getTypeSizeInBits(get(getContext()));
  return dataLayout.getTypeSizeInBits(get(getElementType()));
}

unsigned PointerType::getABIAlignment(const DataLayout &dataLayout,
                                      DataLayoutEntryListRef params) const {
  if (Optional<unsigned> alignment =
          getPointerDataLayoutEntry(params, *this, DLEntryPos::Abi))
    return *alignment;

  if (isOpaque())
    return dataLayout.getTypeABIAlignment(get(getContext()));
  return dataLayout.getTypeABIAlignment(get(getElementType()));
}

unsigned
PointerType::getPreferredAlignment(const DataLayout &dataLayout,
                                   DataLayoutEntryListRef params) const {
  if (Optional<unsigned> alignment =
          getPointerDataLayoutEntry(params, *this, DLEntryPos::Preferred))
    return *alignment;

  if (isOpaque())
    return dataLayout.getTypePreferredAlignment(get(getContext()));
  return dataLayout.getTypePreferredAlignment(get(getElementType()));
}

bool PointerType::areCompatible(DataLayoutEntryListRef oldLayout,
                                DataLayoutEntryListRef newLayout) const {
  for (DataLayoutEntryInterface newEntry : newLayout) {
    if (!newEntry.isTypeEntry())
      continue;
    unsigned size = kDefaultPointerSizeBits;
    unsigned abi = kDefaultPointerAlignment;
    auto newType = newEntry.getKey().get<Type>().cast<PointerType>();
    const auto *it =
        llvm::find_if(oldLayout, [&](DataLayoutEntryInterface entry) {
          if (auto type = entry.getKey().dyn_cast<Type>()) {
            return type.cast<PointerType>().getAddressSpace() ==
                   newType.getAddressSpace();
          }
          return false;
        });
    if (it == oldLayout.end()) {
      llvm::find_if(oldLayout, [&](DataLayoutEntryInterface entry) {
        if (auto type = entry.getKey().dyn_cast<Type>()) {
          return type.cast<PointerType>().getAddressSpace() == 0;
        }
        return false;
      });
    }
    if (it != oldLayout.end()) {
      size = extractPointerSpecValue(*it, DLEntryPos::Size);
      abi = extractPointerSpecValue(*it, DLEntryPos::Abi);
    }

    Attribute newSpec = newEntry.getValue().cast<DenseIntElementsAttr>();
    unsigned newSize = extractPointerSpecValue(newSpec, DLEntryPos::Size);
    unsigned newAbi = extractPointerSpecValue(newSpec, DLEntryPos::Abi);
    if (size != newSize || abi < newAbi || abi % newAbi != 0)
      return false;
  }
  return true;
}

LogicalResult PointerType::verifyEntries(DataLayoutEntryListRef entries,
                                         Location loc) const {
  for (DataLayoutEntryInterface entry : entries) {
    if (!entry.isTypeEntry())
      continue;
    auto key = entry.getKey().get<Type>().cast<PointerType>();
    auto values = entry.getValue().dyn_cast<DenseIntElementsAttr>();
    if (!values || (values.size() != 3 && values.size() != 4)) {
      return emitError(loc)
             << "expected layout attribute for " << entry.getKey().get<Type>()
             << " to be a dense integer elements attribute with 3 or 4 "
                "elements";
    }
    if (key.getElementType() && !key.getElementType().isInteger(8)) {
      return emitError(loc) << "unexpected layout attribute for pointer to "
                            << key.getElementType();
    }
    if (extractPointerSpecValue(values, DLEntryPos::Abi) >
        extractPointerSpecValue(values, DLEntryPos::Preferred)) {
      return emitError(loc) << "preferred alignment is expected to be at least "
                               "as large as ABI alignment";
    }
  }
  return success();
}

StaticArrayType StaticArrayType::get(Type pointee, unsigned size) {
  assert(pointee && "expected non-null subtype");
  return Base::get(pointee.getContext(), pointee, size);
}

StaticArrayType
StaticArrayType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                            Type pointee, unsigned size) {
  return Base::getChecked(emitError, pointee.getContext(), pointee, size);
}

Type StaticArrayType::getElementType() const { return getImpl()->elementType; }

unsigned StaticArrayType::getSize() const { return getImpl()->size; }

LogicalResult StaticArrayType::verify(function_ref<InFlightDiagnostic()>, Type,
                                      unsigned) {
  return success();
}

/// Returns the part of the data layout entry that corresponds to `pos` for the
/// given `type` by interpreting the list of entries `params`. For the pointer
/// type in the default address space, returns the default value if the entries
/// do not provide a custom one, for other address spaces returns None.
static Optional<unsigned> getArrayDataLayoutEntry(DataLayoutEntryListRef params,
                                                  StaticArrayType type,
                                                  DLEntryPos pos) {
  // First, look for the entry for the pointer in the current address space.
  Attribute currentEntry;
  for (DataLayoutEntryInterface entry : params) {
    if (!entry.isTypeEntry())
      continue;
    if (entry.getKey().get<Type>().cast<StaticArrayType>().getSize() ==
        type.getSize()) {
      currentEntry = entry.getValue();
      break;
    }
  }
  if (currentEntry) {
    return extractPointerSpecValue(currentEntry, pos) /
           (pos == DLEntryPos::Size ? 1 : kBitsInByte);
  }

  return llvm::None;
}

unsigned
StaticArrayType::getTypeSizeInBits(const DataLayout &dataLayout,
                                   DataLayoutEntryListRef params) const {
  if (Optional<unsigned> size =
          getArrayDataLayoutEntry(params, *this, DLEntryPos::Size))
    return *size;

  return dataLayout.getTypeSizeInBits(getElementType()) * getSize();
}

unsigned StaticArrayType::getABIAlignment(const DataLayout &dataLayout,
                                          DataLayoutEntryListRef params) const {
  if (Optional<unsigned> alignment =
          getArrayDataLayoutEntry(params, *this, DLEntryPos::Abi))
    return *alignment;

  return dataLayout.getTypeABIAlignment(get(getElementType(), getSize()));
}

unsigned
StaticArrayType::getPreferredAlignment(const DataLayout &dataLayout,
                                       DataLayoutEntryListRef params) const {
  if (Optional<unsigned> alignment =
          getArrayDataLayoutEntry(params, *this, DLEntryPos::Preferred))
    return *alignment;

  return dataLayout.getTypePreferredAlignment(get(getElementType(), getSize()));
}

bool StaticArrayType::areCompatible(DataLayoutEntryListRef oldLayout,
                                    DataLayoutEntryListRef newLayout) const {
  for (DataLayoutEntryInterface newEntry : newLayout) {
    if (!newEntry.isTypeEntry())
      continue;
    unsigned size = kDefaultPointerSizeBits;
    unsigned abi = kDefaultPointerAlignment;
    auto newType = newEntry.getKey().get<Type>().cast<StaticArrayType>();
    const auto *it =
        llvm::find_if(oldLayout, [&](DataLayoutEntryInterface entry) {
          if (auto type = entry.getKey().dyn_cast<Type>()) {
            return type.cast<StaticArrayType>().getSize() == newType.getSize();
          }
          return false;
        });
    if (it == oldLayout.end()) {
      return false;
    }
    if (it != oldLayout.end()) {
      size = extractPointerSpecValue(*it, DLEntryPos::Size);
      abi = extractPointerSpecValue(*it, DLEntryPos::Abi);
    }

    Attribute newSpec = newEntry.getValue().cast<DenseIntElementsAttr>();
    unsigned newSize = extractPointerSpecValue(newSpec, DLEntryPos::Size);
    unsigned newAbi = extractPointerSpecValue(newSpec, DLEntryPos::Abi);
    if (size != newSize || abi < newAbi || abi % newAbi != 0)
      return false;
  }
  return true;
}

LogicalResult StaticArrayType::verifyEntries(DataLayoutEntryListRef entries,
                                             Location loc) const {
  for (DataLayoutEntryInterface entry : entries) {
    if (!entry.isTypeEntry())
      continue;
    auto key = entry.getKey().get<Type>().cast<StaticArrayType>();
    auto values = entry.getValue().dyn_cast<DenseIntElementsAttr>();
    if (!values || (values.size() != 3 && values.size() != 4)) {
      return emitError(loc)
             << "expected layout attribute for " << entry.getKey().get<Type>()
             << " to be a dense integer elements attribute with 3 or 4 "
                "elements";
    }
    if (key.getElementType() && !key.getElementType().isInteger(8)) {
      return emitError(loc) << "unexpected layout attribute for pointer to "
                            << key.getElementType();
    }
    if (extractPointerSpecValue(values, DLEntryPos::Abi) >
        extractPointerSpecValue(values, DLEntryPos::Preferred)) {
      return emitError(loc) << "preferred alignment is expected to be at least "
                               "as large as ABI alignment";
    }
  }
  return success();
}
