/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace omp {
class OutlineableOpenMPOpInterface;
namespace detail {
struct OutlineableOpenMPOpInterfaceInterfaceTraits {
  struct Concept {
    ::mlir::Block*(*getAllocaBlock)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::omp::OutlineableOpenMPOpInterface;
    Model() : Concept{getAllocaBlock} {}

    static inline ::mlir::Block*getAllocaBlock(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::omp::OutlineableOpenMPOpInterface;
    FallbackModel() : Concept{getAllocaBlock} {}

    static inline ::mlir::Block*getAllocaBlock(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};template <typename ConcreteOp>
struct OutlineableOpenMPOpInterfaceTrait;

} // namespace detail
class OutlineableOpenMPOpInterface : public ::mlir::OpInterface<OutlineableOpenMPOpInterface, detail::OutlineableOpenMPOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<OutlineableOpenMPOpInterface, detail::OutlineableOpenMPOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::OutlineableOpenMPOpInterfaceTrait<ConcreteOp> {};
  /// Get alloca block
  ::mlir::Block*getAllocaBlock();
};
namespace detail {
  template <typename ConcreteOp>
  struct OutlineableOpenMPOpInterfaceTrait : public ::mlir::OpInterface<OutlineableOpenMPOpInterface, detail::OutlineableOpenMPOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };
}// namespace detail
template<typename ConcreteOp>
::mlir::Block*detail::OutlineableOpenMPOpInterfaceInterfaceTraits::Model<ConcreteOp>::getAllocaBlock(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return &(llvm::cast<ConcreteOp>(tablegen_opaque_val)).getRegion().front();
}
template<typename ConcreteOp>
::mlir::Block*detail::OutlineableOpenMPOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getAllocaBlock(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getAllocaBlock(tablegen_opaque_val);
}
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class ReductionClauseInterface;
namespace detail {
struct ReductionClauseInterfaceInterfaceTraits {
  struct Concept {
    ::mlir::SmallVector<::mlir::Value> (*getAllReductionVars)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::omp::ReductionClauseInterface;
    Model() : Concept{getAllReductionVars} {}

    static inline ::mlir::SmallVector<::mlir::Value> getAllReductionVars(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::omp::ReductionClauseInterface;
    FallbackModel() : Concept{getAllReductionVars} {}

    static inline ::mlir::SmallVector<::mlir::Value> getAllReductionVars(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    ::mlir::SmallVector<::mlir::Value> getAllReductionVars(::mlir::Operation *tablegen_opaque_val) const;
  };
};template <typename ConcreteOp>
struct ReductionClauseInterfaceTrait;

} // namespace detail
class ReductionClauseInterface : public ::mlir::OpInterface<ReductionClauseInterface, detail::ReductionClauseInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<ReductionClauseInterface, detail::ReductionClauseInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::ReductionClauseInterfaceTrait<ConcreteOp> {};
  /// Get reduction vars
  ::mlir::SmallVector<::mlir::Value> getAllReductionVars();
};
namespace detail {
  template <typename ConcreteOp>
  struct ReductionClauseInterfaceTrait : public ::mlir::OpInterface<ReductionClauseInterface, detail::ReductionClauseInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    /// Get reduction vars
    ::mlir::SmallVector<::mlir::Value> getAllReductionVars() {
      return (*static_cast<ConcreteOp *>(this)).getReductionVars();
    }
  };
}// namespace detail
template<typename ConcreteOp>
::mlir::SmallVector<::mlir::Value> detail::ReductionClauseInterfaceInterfaceTraits::Model<ConcreteOp>::getAllReductionVars(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getAllReductionVars();
}
template<typename ConcreteOp>
::mlir::SmallVector<::mlir::Value> detail::ReductionClauseInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getAllReductionVars(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getAllReductionVars(tablegen_opaque_val);
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::SmallVector<::mlir::Value> detail::ReductionClauseInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getAllReductionVars(::mlir::Operation *tablegen_opaque_val) const {
return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getReductionVars();
}
} // namespace omp
} // namespace mlir
