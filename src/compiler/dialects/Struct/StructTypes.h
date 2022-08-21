//===- StructTypes.h - Struct dialect types ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_STRUCT_STRUCTTYPES_H_
#define MLIR_DIALECT_STRUCT_STRUCTTYPES_H_

#include "mlir/IR/Types.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "llvm/ADT/Bitfields.h"
#include "llvm/ADT/PointerIntPair.h"

namespace mlir {
namespace structure {
namespace detail {
struct StructTypeStorage : public TypeStorage {
public:
  /// Construction/uniquing key class for Struct dialect structure storage. Note
  /// that this is a transient helper data structure that is NOT stored.
  /// Therefore, it intentionally avoids bit manipulation and type erasure in
  /// pointers to make manipulation more straightforward. Not all elements of
  /// the key participate in uniquing, but all elements participate in
  /// construction.
  class Key {
  public:
    /// Constructs a key for an identified struct.
    Key(StringRef name, bool opaque)
        : name(name), identified(true), packed(false), opaque(opaque) {}
    /// Constructs a key for a literal struct.
    Key(ArrayRef<Type> types, bool packed)
        : types(types), identified(false), packed(packed), opaque(false) {}

    /// Checks a specific property of the struct.
    bool isIdentified() const { return identified; }
    bool isPacked() const {
      assert(!isIdentified() &&
             "'packed' bit is not part of the key for identified structs");
      return packed;
    }
    bool isOpaque() const {
      assert(isIdentified() &&
             "'opaque' bit is meaningless on literal structs");
      return opaque;
    }

    /// Returns the identifier of a key for identified structs.
    StringRef getIdentifier() const {
      assert(isIdentified() &&
             "non-identified struct key cannot have an identifier");
      return name;
    }

    /// Returns the list of type contained in the key of a literal struct.
    ArrayRef<Type> getTypeList() const {
      assert(!isIdentified() &&
             "identified struct key cannot have a type list");
      return types;
    }

    /// Returns the hash value of the key. This combines various flags into a
    /// single value: the identified flag sets the first bit, and the packedness
    /// flag sets the second bit. Opacity bit is only used for construction and
    /// does not participate in uniquing.
    llvm::hash_code hashValue() const {
      constexpr static unsigned kIdentifiedHashFlag = 1;
      constexpr static unsigned kPackedHashFlag = 2;

      unsigned flags = 0;
      if (isIdentified()) {
        flags |= kIdentifiedHashFlag;
        return llvm::hash_combine(flags, getIdentifier());
      }
      if (isPacked())
        flags |= kPackedHashFlag;
      return llvm::hash_combine(flags, getTypeList());
    }

    /// Compares two keys.
    bool operator==(const Key &other) const {
      if (isIdentified())
        return other.isIdentified() &&
               other.getIdentifier().equals(getIdentifier());

      return !other.isIdentified() && other.isPacked() == isPacked() &&
             other.getTypeList() == getTypeList();
    }

    /// Copies dynamically-sized components of the key into the given allocator.
    Key copyIntoAllocator(TypeStorageAllocator &allocator) const {
      if (isIdentified())
        return Key(allocator.copyInto(name), opaque);
      return Key(allocator.copyInto(types), packed);
    }

  private:
    ArrayRef<Type> types;
    StringRef name;
    bool identified;
    bool packed;
    bool opaque;
  };
  using KeyTy = Key;

  /// Returns the string identifier of an identified struct.
  StringRef getIdentifier() const {
    assert(isIdentified() && "requested identifier on a non-identified struct");
    return StringRef(static_cast<const char *>(keyPtr), keySize());
  }

  /// Returns the list of types (partially) identifying a literal struct.
  ArrayRef<Type> getTypeList() const {
    // If this triggers, use getIdentifiedStructBody() instead.
    assert(!isIdentified() && "requested typelist on an identified struct");
    return ArrayRef<Type>(static_cast<const Type *>(keyPtr), keySize());
  }

  /// Returns the list of types contained in an identified struct.
  ArrayRef<Type> getIdentifiedStructBody() const {
    // If this triggers, use getTypeList() instead.
    assert(isIdentified() &&
           "requested struct body on a non-identified struct");
    return ArrayRef<Type>(identifiedBodyArray, identifiedBodySize());
  }

  /// Checks whether the struct is identified.
  bool isIdentified() const {
    return llvm::Bitfield::get<KeyFlagIdentified>(keySizeAndFlags);
  }

  /// Checks whether the struct is packed (both literal and identified structs).
  bool isPacked() const {
    return isIdentified() ? llvm::Bitfield::get<MutableFlagPacked>(
                                identifiedBodySizeAndFlags)
                          : llvm::Bitfield::get<KeyFlagPacked>(keySizeAndFlags);
  }

  /// Checks whether a struct is marked as intentionally opaque (an
  /// uninitialized struct is also considered opaque by the user, call
  /// isInitialized to check that).
  bool isOpaque() const {
    return llvm::Bitfield::get<MutableFlagOpaque>(identifiedBodySizeAndFlags);
  }

  /// Checks whether an identified struct has been explicitly initialized either
  /// by setting its body or by marking it as intentionally opaque.
  bool isInitialized() const {
    return llvm::Bitfield::get<MutableFlagInitialized>(
        identifiedBodySizeAndFlags);
  }

  /// Constructs the storage from the given key. This sets up the uniquing key
  /// components and optionally the mutable component if they construction key
  /// has the relevant information. In the latter case, the struct is considered
  /// as initialized and can no longer be mutated.
  StructTypeStorage(const KeyTy &key) {
    if (!key.isIdentified()) {
      ArrayRef<Type> types = key.getTypeList();
      keyPtr = static_cast<const void *>(types.data());
      setKeySize(types.size());
      llvm::Bitfield::set<KeyFlagPacked>(keySizeAndFlags, key.isPacked());
      return;
    }

    StringRef name = key.getIdentifier();
    keyPtr = static_cast<const void *>(name.data());
    setKeySize(name.size());
    llvm::Bitfield::set<KeyFlagIdentified>(keySizeAndFlags, true);

    // If the struct is being constructed directly as opaque, mark it as
    // initialized.
    llvm::Bitfield::set<MutableFlagInitialized>(identifiedBodySizeAndFlags,
                                                key.isOpaque());
    llvm::Bitfield::set<MutableFlagOpaque>(identifiedBodySizeAndFlags,
                                           key.isOpaque());
  }

  /// Hook into the type uniquing infrastructure.
  bool operator==(const KeyTy &other) const { return getKey() == other; };
  static llvm::hash_code hashKey(const KeyTy &key) { return key.hashValue(); }
  static StructTypeStorage *construct(TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    return new (allocator.allocate<StructTypeStorage>())
        StructTypeStorage(key.copyIntoAllocator(allocator));
  }

  /// Sets the body of an identified struct. If the struct is already
  /// initialized, succeeds only if the body is equal to the current body. Fails
  /// if the struct is marked as intentionally opaque. The struct will be marked
  /// as initialized as a result of this operation and can no longer be changed.
  LogicalResult mutate(TypeStorageAllocator &allocator, ArrayRef<Type> body,
                       bool packed) {
    if (!isIdentified())
      return failure();
    if (isInitialized())
      return success(!isOpaque() && body == getIdentifiedStructBody() &&
                     packed == isPacked());

    llvm::Bitfield::set<MutableFlagInitialized>(identifiedBodySizeAndFlags,
                                                true);
    llvm::Bitfield::set<MutableFlagPacked>(identifiedBodySizeAndFlags, packed);

    ArrayRef<Type> typesInAllocator = allocator.copyInto(body);
    identifiedBodyArray = typesInAllocator.data();
    setIdentifiedBodySize(typesInAllocator.size());

    return success();
  }

private:
  /// Returns the number of elements in the key.
  unsigned keySize() const {
    return llvm::Bitfield::get<KeySize>(keySizeAndFlags);
  }

  /// Sets the number of elements in the key.
  void setKeySize(unsigned value) {
    llvm::Bitfield::set<KeySize>(keySizeAndFlags, value);
  }

  /// Returns the number of types contained in an identified struct.
  unsigned identifiedBodySize() const {
    return llvm::Bitfield::get<MutableSize>(identifiedBodySizeAndFlags);
  }
  /// Sets the number of types contained in an identified struct.
  void setIdentifiedBodySize(unsigned value) {
    llvm::Bitfield::set<MutableSize>(identifiedBodySizeAndFlags, value);
  }

  /// Returns the key for the current storage.
  Key getKey() const {
    if (isIdentified())
      return Key(getIdentifier(), isOpaque());
    return Key(getTypeList(), isPacked());
  }

  /// Bitfield elements for `keyAndSizeFlags`:
  ///   - bit 0: identified key flag;
  ///   - bit 1: packed key flag;
  ///   - bits 2..bitwidth(unsigned): size of the key.
  using KeyFlagIdentified =
      llvm::Bitfield::Element<bool, /*Offset=*/0, /*Size=*/1>;
  using KeyFlagPacked = llvm::Bitfield::Element<bool, /*Offset=*/1, /*Size=*/1>;
  using KeySize =
      llvm::Bitfield::Element<unsigned, /*Offset=*/2,
                              std::numeric_limits<unsigned>::digits - 2>;

  /// Bitfield elements for `identifiedBodySizeAndFlags`:
  ///   - bit 0: opaque flag;
  ///   - bit 1: packed mutable flag;
  ///   - bit 2: initialized flag;
  ///   - bits 3..bitwidth(unsigned): size of the identified body.
  using MutableFlagOpaque =
      llvm::Bitfield::Element<bool, /*Offset=*/0, /*Size=*/1>;
  using MutableFlagPacked =
      llvm::Bitfield::Element<bool, /*Offset=*/1, /*Size=*/1>;
  using MutableFlagInitialized =
      llvm::Bitfield::Element<bool, /*Offset=*/2, /*Size=*/1>;
  using MutableSize =
      llvm::Bitfield::Element<unsigned, /*Offset=*/3,
                              std::numeric_limits<unsigned>::digits - 3>;

  /// Pointer to the first element of the uniquing key.
  // Note: cannot use PointerUnion because bump-ptr allocator does not guarantee
  // address alignment.
  const void *keyPtr = nullptr;

  /// Pointer to the first type contained in an identified struct.
  const Type *identifiedBodyArray = nullptr;

  /// Size of the uniquing key combined with identified/literal and
  /// packedness bits. Must only be used through the Key* bitfields.
  unsigned keySizeAndFlags = 0;

  /// Number of the types contained in an identified struct combined with
  /// mutable flags. Must only be used through the Mutable* bitfields.
  unsigned identifiedBodySizeAndFlags = 0;
};
} // namespace detail

struct StructType
    : public Type::TypeBase<StructType, Type, detail::StructTypeStorage,
                            DataLayoutTypeInterface::Trait, TypeTrait::IsMutable> {
  using Base::Base;

  /// Checks if the given type can be contained in a structure type.
  static bool isValidElementType(Type type);

  /// Gets or creates an identified struct with the given name in the provided
  /// context. Note that unlike llvm::StructType::create, this function will
  /// _NOT_ rename a struct in case a struct with the same name already exists
  /// in the context. Instead, it will just return the existing struct,
  /// similarly to the rest of MLIR type ::get methods.
  static StructType getIdentified(MLIRContext *context, StringRef name);
  static StructType
  getIdentifiedChecked(function_ref<InFlightDiagnostic()> emitError,
                       MLIRContext *context, StringRef name);

  /// Gets a new identified struct with the given body. The body _cannot_ be
  /// changed later. If a struct with the given name already exists, renames
  /// the struct by appending a `.` followed by a number to the name. Renaming
  /// happens even if the existing struct has the same body.
  static StructType getNewIdentified(MLIRContext *context, StringRef name,
                                     ArrayRef<Type> elements,
                                     bool isPacked = false);

  /// Gets or creates a literal struct with the given body in the provided
  /// context.
  static StructType getLiteral(MLIRContext *context, ArrayRef<Type> types,
                               bool isPacked = false);
  static StructType
  getLiteralChecked(function_ref<InFlightDiagnostic()> emitError,
                    MLIRContext *context, ArrayRef<Type> types,
                    bool isPacked = false);

  /// Set the body of an identified struct. Returns failure if the body could
  /// not be set, e.g. if the struct already has a body or if it was marked as
  /// intentionally opaque. This might happen in a multi-threaded context when a
  /// different thread modified the struct after it was created. Most callers
  /// are likely to assert this always succeeds, but it is possible to implement
  /// a local renaming scheme based on the result of this call.
  LogicalResult setBody(ArrayRef<Type> types, bool isPacked);

  /// Checks if a struct is packed.
  bool isPacked() const;

  /// Checks if a struct is identified.
  bool isIdentified() const;

  /// Checks if a struct is initialized.
  bool isInitialized();

  /// Returns the name of an identified struct.
  StringRef getName();

  /// Returns the list of element types contained in a non-opaque struct.
  ArrayRef<Type> getBody() const;

  /// Verifies that the type about to be constructed is well-formed.
  static LogicalResult verify(function_ref<InFlightDiagnostic()> emitError,
                              StringRef, bool);
  static LogicalResult verify(function_ref<InFlightDiagnostic()> emitError,
                              ArrayRef<Type> types, bool);

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
} // namespace structure
} // namespace mlir

#endif // MLIR_DIALECT_STRUCT_STRUCTTYPES_H_
