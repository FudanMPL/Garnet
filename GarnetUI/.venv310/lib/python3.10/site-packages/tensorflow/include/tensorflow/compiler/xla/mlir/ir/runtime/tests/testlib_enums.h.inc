/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Declarations                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace xla {
namespace runtime {
// Custom Call Testlib Enum Type.
enum class EnumType : uint32_t {
  Foo = 0,
  Bar = 1,
  Baz = 2,
};

::llvm::Optional<EnumType> symbolizeEnumType(uint32_t);
::llvm::StringRef stringifyEnumType(EnumType);
::llvm::Optional<EnumType> symbolizeEnumType(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForEnumType() {
  return 2;
}


inline ::llvm::StringRef stringifyEnum(EnumType enumValue) {
  return stringifyEnumType(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<EnumType> symbolizeEnum<EnumType>(::llvm::StringRef str) {
  return symbolizeEnumType(str);
}
} // namespace runtime
} // namespace xla

namespace llvm {
template<> struct DenseMapInfo<::xla::runtime::EnumType> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::xla::runtime::EnumType getEmptyKey() {
    return static_cast<::xla::runtime::EnumType>(StorageInfo::getEmptyKey());
  }

  static inline ::xla::runtime::EnumType getTombstoneKey() {
    return static_cast<::xla::runtime::EnumType>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::xla::runtime::EnumType &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::xla::runtime::EnumType &lhs, const ::xla::runtime::EnumType &rhs) {
    return lhs == rhs;
  }
};
}

namespace xla {
namespace runtime {
// Another Custom Call Testlib Enum Type.
enum class EnumType2 : uint32_t {
  Foo = 0,
  Bar = 1,
  Baz = 2,
};

::llvm::Optional<EnumType2> symbolizeEnumType2(uint32_t);
::llvm::StringRef stringifyEnumType2(EnumType2);
::llvm::Optional<EnumType2> symbolizeEnumType2(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForEnumType2() {
  return 2;
}


inline ::llvm::StringRef stringifyEnum(EnumType2 enumValue) {
  return stringifyEnumType2(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<EnumType2> symbolizeEnum<EnumType2>(::llvm::StringRef str) {
  return symbolizeEnumType2(str);
}
} // namespace runtime
} // namespace xla

namespace llvm {
template<> struct DenseMapInfo<::xla::runtime::EnumType2> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::xla::runtime::EnumType2 getEmptyKey() {
    return static_cast<::xla::runtime::EnumType2>(StorageInfo::getEmptyKey());
  }

  static inline ::xla::runtime::EnumType2 getTombstoneKey() {
    return static_cast<::xla::runtime::EnumType2>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::xla::runtime::EnumType2 &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::xla::runtime::EnumType2 &lhs, const ::xla::runtime::EnumType2 &rhs) {
    return lhs == rhs;
  }
};
}

