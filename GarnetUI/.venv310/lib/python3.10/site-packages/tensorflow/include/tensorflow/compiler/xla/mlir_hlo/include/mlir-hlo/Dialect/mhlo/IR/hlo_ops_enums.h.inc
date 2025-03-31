/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Declarations                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace mhlo {
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
} // namespace mhlo
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::mhlo::ComparisonDirection> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::mhlo::ComparisonDirection getEmptyKey() {
    return static_cast<::mlir::mhlo::ComparisonDirection>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::mhlo::ComparisonDirection getTombstoneKey() {
    return static_cast<::mlir::mhlo::ComparisonDirection>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::mhlo::ComparisonDirection &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::mhlo::ComparisonDirection &lhs, const ::mlir::mhlo::ComparisonDirection &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace mhlo {
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
} // namespace mhlo
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::mhlo::ComparisonType> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::mhlo::ComparisonType getEmptyKey() {
    return static_cast<::mlir::mhlo::ComparisonType>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::mhlo::ComparisonType getTombstoneKey() {
    return static_cast<::mlir::mhlo::ComparisonType>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::mhlo::ComparisonType &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::mhlo::ComparisonType &lhs, const ::mlir::mhlo::ComparisonType &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace mhlo {
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
} // namespace mhlo
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::mhlo::CustomCallApiVersion> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::mhlo::CustomCallApiVersion getEmptyKey() {
    return static_cast<::mlir::mhlo::CustomCallApiVersion>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::mhlo::CustomCallApiVersion getTombstoneKey() {
    return static_cast<::mlir::mhlo::CustomCallApiVersion>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::mhlo::CustomCallApiVersion &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::mhlo::CustomCallApiVersion &lhs, const ::mlir::mhlo::CustomCallApiVersion &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace mhlo {
// Dequantization mode. Only MIN_COMBINED is supported.
enum class DequantizeMode : uint32_t {
  MIN_COMBINED = 0,
};

::llvm::Optional<DequantizeMode> symbolizeDequantizeMode(uint32_t);
::llvm::StringRef stringifyDequantizeMode(DequantizeMode);
::llvm::Optional<DequantizeMode> symbolizeDequantizeMode(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForDequantizeMode() {
  return 0;
}


inline ::llvm::StringRef stringifyEnum(DequantizeMode enumValue) {
  return stringifyDequantizeMode(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<DequantizeMode> symbolizeEnum<DequantizeMode>(::llvm::StringRef str) {
  return symbolizeDequantizeMode(str);
}
} // namespace mhlo
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::mhlo::DequantizeMode> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::mhlo::DequantizeMode getEmptyKey() {
    return static_cast<::mlir::mhlo::DequantizeMode>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::mhlo::DequantizeMode getTombstoneKey() {
    return static_cast<::mlir::mhlo::DequantizeMode>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::mhlo::DequantizeMode &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::mhlo::DequantizeMode &lhs, const ::mlir::mhlo::DequantizeMode &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace mhlo {
// Kind of domain metatdata attached to an HLO domain.
enum class DomainKind : uint32_t {
  sharding = 0,
};

::llvm::Optional<DomainKind> symbolizeDomainKind(uint32_t);
::llvm::StringRef stringifyDomainKind(DomainKind);
::llvm::Optional<DomainKind> symbolizeDomainKind(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForDomainKind() {
  return 0;
}


inline ::llvm::StringRef stringifyEnum(DomainKind enumValue) {
  return stringifyDomainKind(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<DomainKind> symbolizeEnum<DomainKind>(::llvm::StringRef str) {
  return symbolizeDomainKind(str);
}
} // namespace mhlo
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::mhlo::DomainKind> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::mhlo::DomainKind getEmptyKey() {
    return static_cast<::mlir::mhlo::DomainKind>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::mhlo::DomainKind getTombstoneKey() {
    return static_cast<::mlir::mhlo::DomainKind>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::mhlo::DomainKind &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::mhlo::DomainKind &lhs, const ::mlir::mhlo::DomainKind &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace mhlo {
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
} // namespace mhlo
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::mhlo::FftType> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::mhlo::FftType getEmptyKey() {
    return static_cast<::mlir::mhlo::FftType>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::mhlo::FftType getTombstoneKey() {
    return static_cast<::mlir::mhlo::FftType>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::mhlo::FftType &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::mhlo::FftType &lhs, const ::mlir::mhlo::FftType &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace mhlo {
// fusion kind
enum class FusionKind : uint32_t {
  kLoop = 0,
  kInput = 1,
  kOutput = 2,
  kCustom = 3,
};

::llvm::Optional<FusionKind> symbolizeFusionKind(uint32_t);
::llvm::StringRef stringifyFusionKind(FusionKind);
::llvm::Optional<FusionKind> symbolizeFusionKind(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForFusionKind() {
  return 3;
}


inline ::llvm::StringRef stringifyEnum(FusionKind enumValue) {
  return stringifyFusionKind(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<FusionKind> symbolizeEnum<FusionKind>(::llvm::StringRef str) {
  return symbolizeFusionKind(str);
}
} // namespace mhlo
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::mhlo::FusionKind> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::mhlo::FusionKind getEmptyKey() {
    return static_cast<::mlir::mhlo::FusionKind>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::mhlo::FusionKind getTombstoneKey() {
    return static_cast<::mlir::mhlo::FusionKind>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::mhlo::FusionKind &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::mhlo::FusionKind &lhs, const ::mlir::mhlo::FusionKind &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace mhlo {
// XLA precision for an operand. Has backend specific meaning.
enum class Precision : uint32_t {
  DEFAULT = 0,
  HIGH = 1,
  HIGHEST = 2,
  PACKED_NIBBLE = 3,
};

::llvm::Optional<Precision> symbolizePrecision(uint32_t);
::llvm::StringRef stringifyPrecision(Precision);
::llvm::Optional<Precision> symbolizePrecision(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForPrecision() {
  return 3;
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
} // namespace mhlo
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::mhlo::Precision> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::mhlo::Precision getEmptyKey() {
    return static_cast<::mlir::mhlo::Precision>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::mhlo::Precision getTombstoneKey() {
    return static_cast<::mlir::mhlo::Precision>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::mhlo::Precision &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::mhlo::Precision &lhs, const ::mlir::mhlo::Precision &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace mhlo {
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
} // namespace mhlo
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::mhlo::RngAlgorithm> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::mhlo::RngAlgorithm getEmptyKey() {
    return static_cast<::mlir::mhlo::RngAlgorithm>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::mhlo::RngAlgorithm getTombstoneKey() {
    return static_cast<::mlir::mhlo::RngAlgorithm>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::mhlo::RngAlgorithm &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::mhlo::RngAlgorithm &lhs, const ::mlir::mhlo::RngAlgorithm &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace mhlo {
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
} // namespace mhlo
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::mhlo::RngDistribution> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::mhlo::RngDistribution getEmptyKey() {
    return static_cast<::mlir::mhlo::RngDistribution>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::mhlo::RngDistribution getTombstoneKey() {
    return static_cast<::mlir::mhlo::RngDistribution>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::mhlo::RngDistribution &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::mhlo::RngDistribution &lhs, const ::mlir::mhlo::RngDistribution &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace mhlo {
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
} // namespace mhlo
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::mhlo::Transpose> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::mhlo::Transpose getEmptyKey() {
    return static_cast<::mlir::mhlo::Transpose>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::mhlo::Transpose getTombstoneKey() {
    return static_cast<::mlir::mhlo::Transpose>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::mhlo::Transpose &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::mhlo::Transpose &lhs, const ::mlir::mhlo::Transpose &rhs) {
    return lhs == rhs;
  }
};
}

