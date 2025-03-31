/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Declarations                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace lmhlo_gpu {
// Activation applied with fused convolution
enum class Activation : uint32_t {
  None = 0,
  Sigmoid = 1,
  Tanh = 2,
  Relu = 3,
  Relu6 = 4,
  ReluX = 5,
  BandPass = 6,
};

::llvm::Optional<Activation> symbolizeActivation(uint32_t);
::llvm::StringRef stringifyActivation(Activation);
::llvm::Optional<Activation> symbolizeActivation(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForActivation() {
  return 6;
}


inline ::llvm::StringRef stringifyEnum(Activation enumValue) {
  return stringifyActivation(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<Activation> symbolizeEnum<Activation>(::llvm::StringRef str) {
  return symbolizeActivation(str);
}
} // namespace lmhlo_gpu
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::lmhlo_gpu::Activation> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::lmhlo_gpu::Activation getEmptyKey() {
    return static_cast<::mlir::lmhlo_gpu::Activation>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::lmhlo_gpu::Activation getTombstoneKey() {
    return static_cast<::mlir::lmhlo_gpu::Activation>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::lmhlo_gpu::Activation &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::lmhlo_gpu::Activation &lhs, const ::mlir::lmhlo_gpu::Activation &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace lmhlo_gpu {
// Epilogue for cublasLt matmul
enum class CublasLtMatmulEpilogue : uint32_t {
  Default = 0,
  Bias = 1,
};

::llvm::Optional<CublasLtMatmulEpilogue> symbolizeCublasLtMatmulEpilogue(uint32_t);
::llvm::StringRef stringifyCublasLtMatmulEpilogue(CublasLtMatmulEpilogue);
::llvm::Optional<CublasLtMatmulEpilogue> symbolizeCublasLtMatmulEpilogue(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForCublasLtMatmulEpilogue() {
  return 1;
}


inline ::llvm::StringRef stringifyEnum(CublasLtMatmulEpilogue enumValue) {
  return stringifyCublasLtMatmulEpilogue(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<CublasLtMatmulEpilogue> symbolizeEnum<CublasLtMatmulEpilogue>(::llvm::StringRef str) {
  return symbolizeCublasLtMatmulEpilogue(str);
}
} // namespace lmhlo_gpu
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::lmhlo_gpu::CublasLtMatmulEpilogue> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::lmhlo_gpu::CublasLtMatmulEpilogue getEmptyKey() {
    return static_cast<::mlir::lmhlo_gpu::CublasLtMatmulEpilogue>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::lmhlo_gpu::CublasLtMatmulEpilogue getTombstoneKey() {
    return static_cast<::mlir::lmhlo_gpu::CublasLtMatmulEpilogue>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::lmhlo_gpu::CublasLtMatmulEpilogue &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::lmhlo_gpu::CublasLtMatmulEpilogue &lhs, const ::mlir::lmhlo_gpu::CublasLtMatmulEpilogue &rhs) {
    return lhs == rhs;
  }
};
}

