/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES


namespace mlir {
class AsmParser;
class AsmPrinter;
} // namespace mlir
namespace mlir {
namespace gml_st {
class IteratorTypeAttr;
namespace detail {
struct IteratorTypeAttrStorage;
} // namespace detail
class IteratorTypeAttr : public ::mlir::Attribute::AttrBase<IteratorTypeAttr, ::mlir::Attribute, detail::IteratorTypeAttrStorage> {
public:
  using Base::Base;
public:
  static IteratorTypeAttr get(::mlir::MLIRContext *context, ::mlir::utils::IteratorType value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"iterator_type"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::utils::IteratorType getValue() const;
};
} // namespace gml_st
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::gml_st::IteratorTypeAttr)

#endif  // GET_ATTRDEF_CLASSES

