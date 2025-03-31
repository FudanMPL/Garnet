/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Declarations                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace chlo {
// Which comparison operation to perform.
enum class ComparisonDirection : uint32_t {
  EQ = 0,
  NE = 1,
  GE = 2,
  GT = 3,
  LE = 4,
  LT = 5,
};

::llvm::Optional<ComparisonDirection> symbolizeComparisonDirection(uint32_t);
::llvm::StringRef stringifyComparisonDirection(ComparisonDirection);
::llvm::Optional<ComparisonDirection> symbolizeComparisonDirection(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForComparisonDirection() {
  return 5;
}


inline ::llvm::StringRef stringifyEnum(ComparisonDirection enumValue) {
  return stringifyComparisonDirection(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<ComparisonDirection> symbolizeEnum<ComparisonDirection>(::llvm::StringRef str) {
  return symbolizeComparisonDirection(str);
}
} // namespace chlo
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::chlo::ComparisonDirection> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::chlo::ComparisonDirection getEmptyKey() {
    return static_cast<::mlir::chlo::ComparisonDirection>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::chlo::ComparisonDirection getTombstoneKey() {
    return static_cast<::mlir::chlo::ComparisonDirection>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::chlo::ComparisonDirection &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::chlo::ComparisonDirection &lhs, const ::mlir::chlo::ComparisonDirection &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace chlo {
// Which comparison type to use.
enum class ComparisonType : uint32_t {
  NOTYPE = 0,
  FLOAT = 1,
  TOTALORDER = 2,
  SIGNED = 3,
  UNSIGNED = 4,
};

::llvm::Optional<ComparisonType> symbolizeComparisonType(uint32_t);
::llvm::StringRef stringifyComparisonType(ComparisonType);
::llvm::Optional<ComparisonType> symbolizeComparisonType(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForComparisonType() {
  return 4;
}


inline ::llvm::StringRef stringifyEnum(ComparisonType enumValue) {
  return stringifyComparisonType(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<ComparisonType> symbolizeEnum<ComparisonType>(::llvm::StringRef str) {
  return symbolizeComparisonType(str);
}
} // namespace chlo
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::chlo::ComparisonType> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::chlo::ComparisonType getEmptyKey() {
    return static_cast<::mlir::chlo::ComparisonType>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::chlo::ComparisonType getTombstoneKey() {
    return static_cast<::mlir::chlo::ComparisonType>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::chlo::ComparisonType &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::chlo::ComparisonType &lhs, const ::mlir::chlo::ComparisonType &rhs) {
    return lhs == rhs;
  }
};
}

