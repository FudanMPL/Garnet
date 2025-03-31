/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Declarations                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace stablehlo {
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
} // namespace stablehlo
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::stablehlo::ComparisonDirection> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::stablehlo::ComparisonDirection getEmptyKey() {
    return static_cast<::mlir::stablehlo::ComparisonDirection>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::stablehlo::ComparisonDirection getTombstoneKey() {
    return static_cast<::mlir::stablehlo::ComparisonDirection>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::stablehlo::ComparisonDirection &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::stablehlo::ComparisonDirection &lhs, const ::mlir::stablehlo::ComparisonDirection &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace stablehlo {
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
} // namespace stablehlo
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::stablehlo::ComparisonType> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::stablehlo::ComparisonType getEmptyKey() {
    return static_cast<::mlir::stablehlo::ComparisonType>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::stablehlo::ComparisonType getTombstoneKey() {
    return static_cast<::mlir::stablehlo::ComparisonType>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::stablehlo::ComparisonType &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::stablehlo::ComparisonType &lhs, const ::mlir::stablehlo::ComparisonType &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace stablehlo {
// Custom call API version
enum class CustomCallApiVersion : uint32_t {
  API_VERSION_UNSPECIFIED = 0,
  API_VERSION_ORIGINAL = 1,
  API_VERSION_STATUS_RETURNING = 2,
  API_VERSION_STATUS_RETURNING_UNIFIED = 3,
};

::llvm::Optional<CustomCallApiVersion> symbolizeCustomCallApiVersion(uint32_t);
::llvm::StringRef stringifyCustomCallApiVersion(CustomCallApiVersion);
::llvm::Optional<CustomCallApiVersion> symbolizeCustomCallApiVersion(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForCustomCallApiVersion() {
  return 3;
}


inline ::llvm::StringRef stringifyEnum(CustomCallApiVersion enumValue) {
  return stringifyCustomCallApiVersion(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<CustomCallApiVersion> symbolizeEnum<CustomCallApiVersion>(::llvm::StringRef str) {
  return symbolizeCustomCallApiVersion(str);
}

class CustomCallApiVersionAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = CustomCallApiVersion;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static CustomCallApiVersionAttr get(::mlir::MLIRContext *context, CustomCallApiVersion val);
  CustomCallApiVersion getValue() const;
};
} // namespace stablehlo
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::stablehlo::CustomCallApiVersion> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::stablehlo::CustomCallApiVersion getEmptyKey() {
    return static_cast<::mlir::stablehlo::CustomCallApiVersion>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::stablehlo::CustomCallApiVersion getTombstoneKey() {
    return static_cast<::mlir::stablehlo::CustomCallApiVersion>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::stablehlo::CustomCallApiVersion &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::stablehlo::CustomCallApiVersion &lhs, const ::mlir::stablehlo::CustomCallApiVersion &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace stablehlo {
// XLA fast fourier transform type.
enum class FftType : uint32_t {
  FFT = 0,
  IFFT = 1,
  RFFT = 2,
  IRFFT = 3,
};

::llvm::Optional<FftType> symbolizeFftType(uint32_t);
::llvm::StringRef stringifyFftType(FftType);
::llvm::Optional<FftType> symbolizeFftType(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForFftType() {
  return 3;
}


inline ::llvm::StringRef stringifyEnum(FftType enumValue) {
  return stringifyFftType(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<FftType> symbolizeEnum<FftType>(::llvm::StringRef str) {
  return symbolizeFftType(str);
}
} // namespace stablehlo
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::stablehlo::FftType> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::stablehlo::FftType getEmptyKey() {
    return static_cast<::mlir::stablehlo::FftType>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::stablehlo::FftType getTombstoneKey() {
    return static_cast<::mlir::stablehlo::FftType>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::stablehlo::FftType &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::stablehlo::FftType &lhs, const ::mlir::stablehlo::FftType &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace stablehlo {
// XLA precision for an operand. Has backend specific meaning.
enum class Precision : uint32_t {
  DEFAULT = 0,
  HIGH = 1,
  HIGHEST = 2,
};

::llvm::Optional<Precision> symbolizePrecision(uint32_t);
::llvm::StringRef stringifyPrecision(Precision);
::llvm::Optional<Precision> symbolizePrecision(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForPrecision() {
  return 2;
}


inline ::llvm::StringRef stringifyEnum(Precision enumValue) {
  return stringifyPrecision(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<Precision> symbolizeEnum<Precision>(::llvm::StringRef str) {
  return symbolizePrecision(str);
}
} // namespace stablehlo
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::stablehlo::Precision> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::stablehlo::Precision getEmptyKey() {
    return static_cast<::mlir::stablehlo::Precision>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::stablehlo::Precision getTombstoneKey() {
    return static_cast<::mlir::stablehlo::Precision>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::stablehlo::Precision &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::stablehlo::Precision &lhs, const ::mlir::stablehlo::Precision &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace stablehlo {
// XLA PRNG algorithm to be used.
enum class RngAlgorithm : uint32_t {
  DEFAULT = 0,
  THREE_FRY = 1,
  PHILOX = 2,
};

::llvm::Optional<RngAlgorithm> symbolizeRngAlgorithm(uint32_t);
::llvm::StringRef stringifyRngAlgorithm(RngAlgorithm);
::llvm::Optional<RngAlgorithm> symbolizeRngAlgorithm(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForRngAlgorithm() {
  return 2;
}


inline ::llvm::StringRef stringifyEnum(RngAlgorithm enumValue) {
  return stringifyRngAlgorithm(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<RngAlgorithm> symbolizeEnum<RngAlgorithm>(::llvm::StringRef str) {
  return symbolizeRngAlgorithm(str);
}
} // namespace stablehlo
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::stablehlo::RngAlgorithm> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::stablehlo::RngAlgorithm getEmptyKey() {
    return static_cast<::mlir::stablehlo::RngAlgorithm>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::stablehlo::RngAlgorithm getTombstoneKey() {
    return static_cast<::mlir::stablehlo::RngAlgorithm>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::stablehlo::RngAlgorithm &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::stablehlo::RngAlgorithm &lhs, const ::mlir::stablehlo::RngAlgorithm &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace stablehlo {
// XLA PRNG distribution to be used.
enum class RngDistribution : uint32_t {
  UNIFORM = 1,
  NORMAL = 2,
};

::llvm::Optional<RngDistribution> symbolizeRngDistribution(uint32_t);
::llvm::StringRef stringifyRngDistribution(RngDistribution);
::llvm::Optional<RngDistribution> symbolizeRngDistribution(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForRngDistribution() {
  return 2;
}


inline ::llvm::StringRef stringifyEnum(RngDistribution enumValue) {
  return stringifyRngDistribution(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<RngDistribution> symbolizeEnum<RngDistribution>(::llvm::StringRef str) {
  return symbolizeRngDistribution(str);
}
} // namespace stablehlo
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::stablehlo::RngDistribution> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::stablehlo::RngDistribution getEmptyKey() {
    return static_cast<::mlir::stablehlo::RngDistribution>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::stablehlo::RngDistribution getTombstoneKey() {
    return static_cast<::mlir::stablehlo::RngDistribution>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::stablehlo::RngDistribution &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::stablehlo::RngDistribution &lhs, const ::mlir::stablehlo::RngDistribution &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace stablehlo {
// Transpose options
enum class Transpose : uint32_t {
  TRANSPOSE_INVALID = 0,
  NO_TRANSPOSE = 1,
  TRANSPOSE = 2,
  ADJOINT = 3,
};

::llvm::Optional<Transpose> symbolizeTranspose(uint32_t);
::llvm::StringRef stringifyTranspose(Transpose);
::llvm::Optional<Transpose> symbolizeTranspose(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForTranspose() {
  return 3;
}


inline ::llvm::StringRef stringifyEnum(Transpose enumValue) {
  return stringifyTranspose(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<Transpose> symbolizeEnum<Transpose>(::llvm::StringRef str) {
  return symbolizeTranspose(str);
}
} // namespace stablehlo
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::stablehlo::Transpose> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::stablehlo::Transpose getEmptyKey() {
    return static_cast<::mlir::stablehlo::Transpose>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::stablehlo::Transpose getTombstoneKey() {
    return static_cast<::mlir::stablehlo::Transpose>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::stablehlo::Transpose &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::stablehlo::Transpose &lhs, const ::mlir::stablehlo::Transpose &rhs) {
    return lhs == rhs;
  }
};
}

