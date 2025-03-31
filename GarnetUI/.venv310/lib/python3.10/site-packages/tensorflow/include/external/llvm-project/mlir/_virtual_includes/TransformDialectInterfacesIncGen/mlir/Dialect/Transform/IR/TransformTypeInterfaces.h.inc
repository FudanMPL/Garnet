/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace transform {
class TransformTypeInterface;
namespace detail {
struct TransformTypeInterfaceInterfaceTraits {
  struct Concept {
    ::mlir::DiagnosedSilenceableFailure (*checkPayload)(const Concept *impl, ::mlir::Type , ::mlir::Location, ::mlir::ArrayRef<::mlir::Operation *>);
  };
  template<typename ConcreteType>
  class Model : public Concept {
  public:
    using Interface = ::mlir::transform::TransformTypeInterface;
    Model() : Concept{checkPayload} {}

    static inline ::mlir::DiagnosedSilenceableFailure checkPayload(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::Location loc, ::mlir::ArrayRef<::mlir::Operation *> payload);
  };
  template<typename ConcreteType>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::transform::TransformTypeInterface;
    FallbackModel() : Concept{checkPayload} {}

    static inline ::mlir::DiagnosedSilenceableFailure checkPayload(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::Location loc, ::mlir::ArrayRef<::mlir::Operation *> payload);
  };
  template<typename ConcreteModel, typename ConcreteType>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteType;
  };
};template <typename ConcreteType>
struct TransformTypeInterfaceTrait;

} // namespace detail
class TransformTypeInterface : public ::mlir::TypeInterface<TransformTypeInterface, detail::TransformTypeInterfaceInterfaceTraits> {
public:
  using ::mlir::TypeInterface<TransformTypeInterface, detail::TransformTypeInterfaceInterfaceTraits>::TypeInterface;
  template <typename ConcreteType>
  struct Trait : public detail::TransformTypeInterfaceTrait<ConcreteType> {};
  /// Checks if the given list of associated Payload IR operations satisfy
  /// the conditions defined by this type. If not, produces a silenceable
  /// error at the specified location.
  ::mlir::DiagnosedSilenceableFailure checkPayload(::mlir::Location loc, ::mlir::ArrayRef<::mlir::Operation *> payload) const;

    DiagnosedSilenceableFailure emitSilenceableError(Location loc) const {
      Diagnostic diag(loc, DiagnosticSeverity::Error);
      return DiagnosedSilenceableFailure::silenceableFailure(std::move(diag));
    }
  };
namespace detail {
  template <typename ConcreteType>
  struct TransformTypeInterfaceTrait : public ::mlir::TypeInterface<TransformTypeInterface, detail::TransformTypeInterfaceInterfaceTraits>::Trait<ConcreteType> {

    DiagnosedSilenceableFailure emitSilenceableError(Location loc) const {
      Diagnostic diag(loc, DiagnosticSeverity::Error);
      return DiagnosedSilenceableFailure::silenceableFailure(std::move(diag));
    }
  
  };
}// namespace detail
template<typename ConcreteType>
::mlir::DiagnosedSilenceableFailure detail::TransformTypeInterfaceInterfaceTraits::Model<ConcreteType>::checkPayload(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::Location loc, ::mlir::ArrayRef<::mlir::Operation *> payload) {
  return (tablegen_opaque_val.cast<ConcreteType>()).checkPayload(loc, payload);
}
template<typename ConcreteType>
::mlir::DiagnosedSilenceableFailure detail::TransformTypeInterfaceInterfaceTraits::FallbackModel<ConcreteType>::checkPayload(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::Location loc, ::mlir::ArrayRef<::mlir::Operation *> payload) {
  return static_cast<const ConcreteType *>(impl)->checkPayload(tablegen_opaque_val, loc, payload);
}
} // namespace transform
} // namespace mlir
