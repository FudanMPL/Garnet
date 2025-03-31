/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace xla {
namespace runtime {
class TraceAnnotationAttrInterface;
namespace detail {
struct TraceAnnotationAttrInterfaceInterfaceTraits {
  struct Concept {
  };
  template<typename ConcreteAttr>
  class Model : public Concept {
  public:
    using Interface = xla::runtime::TraceAnnotationAttrInterface;
    Model() : Concept{} {}

  };
  template<typename ConcreteAttr>
  class FallbackModel : public Concept {
  public:
    using Interface = xla::runtime::TraceAnnotationAttrInterface;
    FallbackModel() : Concept{} {}

  };
  template<typename ConcreteModel, typename ConcreteAttr>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteAttr;
  };
};template <typename ConcreteAttr>
struct TraceAnnotationAttrInterfaceTrait;

} // namespace detail
class TraceAnnotationAttrInterface : public ::mlir::AttributeInterface<TraceAnnotationAttrInterface, detail::TraceAnnotationAttrInterfaceInterfaceTraits> {
public:
  using ::mlir::AttributeInterface<TraceAnnotationAttrInterface, detail::TraceAnnotationAttrInterfaceInterfaceTraits>::AttributeInterface;
  template <typename ConcreteAttr>
  struct Trait : public detail::TraceAnnotationAttrInterfaceTrait<ConcreteAttr> {};
};
namespace detail {
  template <typename ConcreteAttr>
  struct TraceAnnotationAttrInterfaceTrait : public ::mlir::AttributeInterface<TraceAnnotationAttrInterface, detail::TraceAnnotationAttrInterfaceInterfaceTraits>::Trait<ConcreteAttr> {
  };
}// namespace detail
} // namespace runtime
} // namespace xla
