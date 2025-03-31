/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace gml_st {
class ComposeSetInterface;
namespace detail {
struct ComposeSetInterfaceInterfaceTraits {
  struct Concept {
    ::mlir::Value (*compose)(const Concept *impl, ::mlir::Operation *, OpBuilder&);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::gml_st::ComposeSetInterface;
    Model() : Concept{compose} {}

    static inline ::mlir::Value compose(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder& builder);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::gml_st::ComposeSetInterface;
    FallbackModel() : Concept{compose} {}

    static inline ::mlir::Value compose(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder& builder);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};template <typename ConcreteOp>
struct ComposeSetInterfaceTrait;

} // namespace detail
class ComposeSetInterface : public ::mlir::OpInterface<ComposeSetInterface, detail::ComposeSetInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<ComposeSetInterface, detail::ComposeSetInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::ComposeSetInterfaceTrait<ConcreteOp> {};
  /// Returns a composition of this set with its superset operand.
  ::mlir::Value compose(OpBuilder& builder);
};
namespace detail {
  template <typename ConcreteOp>
  struct ComposeSetInterfaceTrait : public ::mlir::OpInterface<ComposeSetInterface, detail::ComposeSetInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };
}// namespace detail
template<typename ConcreteOp>
::mlir::Value detail::ComposeSetInterfaceInterfaceTraits::Model<ConcreteOp>::compose(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder& builder) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).compose(builder);
}
template<typename ConcreteOp>
::mlir::Value detail::ComposeSetInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::compose(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder& builder) {
  return static_cast<const ConcreteOp *>(impl)->compose(tablegen_opaque_val, builder);
}
} // namespace gml_st
} // namespace mlir
