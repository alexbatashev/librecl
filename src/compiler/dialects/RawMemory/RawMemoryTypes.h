//===- RawMemoryTypes.h - Raw Memory dialect types --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_RAWMEMORY_RAWMEMORYTYPES_H_
#define MLIR_DIALECT_RAWMEMORY_RAWMEMORYTYPES_H_

#include "mlir/IR/Types.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"

namespace mlir {
namespace rawmem {
namespace detail {
/// Storage type for LLVM dialect pointer types. These are uniqued by a pair of
/// element type and address space. The element type may be null indicating that
/// the pointer is opaque.
struct PointerTypeStorage : public TypeStorage {
  using KeyTy = std::tuple<Type, unsigned>;

  PointerTypeStorage(const KeyTy &key)
      : pointeeType(std::get<0>(key)), addressSpace(std::get<1>(key)) {}

  static PointerTypeStorage *construct(TypeStorageAllocator &allocator,
                                       const KeyTy &key) {
    return new (allocator.allocate<PointerTypeStorage>())
        PointerTypeStorage(key);
  }

  bool operator==(const KeyTy &key) const {
    return std::make_tuple(pointeeType, addressSpace) == key;
  }

  Type pointeeType;
  unsigned addressSpace;
};
} // namespace detail
} // namespace rawmem
} // namespace mlir

#include "RawMemory/RawMemoryOpsTypes.h.inc"

namespace mlir {
namespace rawmem {

/// Raw Memory dialect pointer type. This type typically represents a reference
/// to an object in memory. Pointers may be opaque or parameterized by the
/// element type. Both opaque and non-opaque pointers are additionally
/// parameterized by the address space.
class PointerType
    : public Type::TypeBase<PointerType, Type, detail::PointerTypeStorage,
                            DataLayoutTypeInterface::Trait> {
public:
  /// Inherit base constructors.
  using Base::Base;

  /// Checks if the given type can have a pointer type pointing to it.
  static bool isValidElementType(Type type);

  /// Gets or creates an instance of LLVM dialect pointer type pointing to an
  /// object of `pointee` type in the given address space. The pointer type is
  /// created in the same context as `pointee`. If the pointee is not provided,
  /// creates an opaque pointer in the given context and address space.
  static PointerType get(MLIRContext *context, unsigned addressSpace = 0);
  static PointerType get(Type pointee, unsigned addressSpace = 0);
  static PointerType getChecked(function_ref<InFlightDiagnostic()> emitError,
                                Type pointee, unsigned addressSpace = 0);
  static PointerType getChecked(function_ref<InFlightDiagnostic()> emitError,
                                MLIRContext *context,
                                unsigned addressSpace = 0);

  /// Returns the pointed-to type. It may be null if the pointer is opaque.
  Type getElementType() const;

  /// Returns `true` if this type is the opaque pointer type, i.e., it has no
  /// pointed-to type.
  bool isOpaque() const;

  /// Returns the address space of the pointer.
  unsigned getAddressSpace() const;

  /// Verifies that the type about to be constructed is well-formed.
  static LogicalResult verify(function_ref<InFlightDiagnostic()> emitError,
                              Type pointee, unsigned);
  static LogicalResult verify(function_ref<InFlightDiagnostic()> emitError,
                              MLIRContext *context, unsigned) {
    return success();
  }

  /// Hooks for DataLayoutTypeInterface. Should not be called directly. Obtain a
  /// DataLayout instance and query it instead.
  unsigned getTypeSizeInBits(const DataLayout &dataLayout,
                             DataLayoutEntryListRef params) const;
  unsigned getABIAlignment(const DataLayout &dataLayout,
                           DataLayoutEntryListRef params) const;
  unsigned getPreferredAlignment(const DataLayout &dataLayout,
                                 DataLayoutEntryListRef params) const;
  bool areCompatible(DataLayoutEntryListRef oldLayout,
                     DataLayoutEntryListRef newLayout) const;
  LogicalResult verifyEntries(DataLayoutEntryListRef entries,
                              Location loc) const;
};
} // namespace rawmem
} // namespace mlir

#endif
