/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace vector {
class MaskableOpInterface;
namespace detail {
struct MaskableOpInterfaceInterfaceTraits {
  struct Concept {
    bool (*supportsPassthru)(const Concept *impl, ::mlir::Operation *);
    mlir::VectorType (*getExpectedMaskType)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::vector::MaskableOpInterface;
    Model() : Concept{supportsPassthru, getExpectedMaskType} {}

    static inline bool supportsPassthru(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline mlir::VectorType getExpectedMaskType(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::vector::MaskableOpInterface;
    FallbackModel() : Concept{supportsPassthru, getExpectedMaskType} {}

    static inline bool supportsPassthru(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline mlir::VectorType getExpectedMaskType(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    bool supportsPassthru(::mlir::Operation *tablegen_opaque_val) const;
    mlir::VectorType getExpectedMaskType(::mlir::Operation *tablegen_opaque_val) const;
  };
};template <typename ConcreteOp>
struct MaskableOpInterfaceTrait;

} // namespace detail
class MaskableOpInterface : public ::mlir::OpInterface<MaskableOpInterface, detail::MaskableOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<MaskableOpInterface, detail::MaskableOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::MaskableOpInterfaceTrait<ConcreteOp> {};
  /// Returns true if the operation may have a passthru argument when masked.
  bool supportsPassthru();
  /// Returns the mask type expected by this operation. It requires the operation to be vectorized.
  mlir::VectorType getExpectedMaskType();
};
namespace detail {
  template <typename ConcreteOp>
  struct MaskableOpInterfaceTrait : public ::mlir::OpInterface<MaskableOpInterface, detail::MaskableOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    /// Returns true if the operation may have a passthru argument when masked.
    bool supportsPassthru() {
      return false;
    }
    /// Returns the mask type expected by this operation. It requires the operation to be vectorized.
    mlir::VectorType getExpectedMaskType() {
      // Default implementation is only aimed for operations that implement the
    // `getVectorType()` method.
      return (*static_cast<ConcreteOp *>(this)).getVectorType().cloneWith(
        /*shape=*/llvm::None, IntegerType::get((*static_cast<ConcreteOp *>(this)).getContext(), /*width=*/1));
    }
  };
}// namespace detail
template<typename ConcreteOp>
bool detail::MaskableOpInterfaceInterfaceTraits::Model<ConcreteOp>::supportsPassthru(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).supportsPassthru();
}
template<typename ConcreteOp>
mlir::VectorType detail::MaskableOpInterfaceInterfaceTraits::Model<ConcreteOp>::getExpectedMaskType(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getExpectedMaskType();
}
template<typename ConcreteOp>
bool detail::MaskableOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::supportsPassthru(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->supportsPassthru(tablegen_opaque_val);
}
template<typename ConcreteOp>
mlir::VectorType detail::MaskableOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getExpectedMaskType(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getExpectedMaskType(tablegen_opaque_val);
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::MaskableOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::supportsPassthru(::mlir::Operation *tablegen_opaque_val) const {
return false;
}
template<typename ConcreteModel, typename ConcreteOp>
mlir::VectorType detail::MaskableOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getExpectedMaskType(::mlir::Operation *tablegen_opaque_val) const {
// Default implementation is only aimed for operations that implement the
    // `getVectorType()` method.
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getVectorType().cloneWith(
        /*shape=*/llvm::None, IntegerType::get((llvm::cast<ConcreteOp>(tablegen_opaque_val)).getContext(), /*width=*/1));
}
} // namespace vector
} // namespace mlir
