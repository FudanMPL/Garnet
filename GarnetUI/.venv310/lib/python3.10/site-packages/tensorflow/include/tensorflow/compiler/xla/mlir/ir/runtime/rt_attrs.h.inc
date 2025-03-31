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
namespace xla {
namespace runtime {
class HloTraceAttr;
namespace detail {
struct HloTraceAttrStorage;
} // namespace detail
class HloTraceAttr : public ::mlir::Attribute::AttrBase<HloTraceAttr, ::mlir::Attribute, detail::HloTraceAttrStorage, xla::runtime::TraceAnnotationAttrInterface::Trait> {
public:
  using Base::Base;
public:
  static HloTraceAttr get(::mlir::MLIRContext *context, ::llvm::StringRef hlo_op, ::llvm::StringRef module, int64_t program_id);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"hlo_trace"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::llvm::StringRef getHloOp() const;
  ::llvm::StringRef getModule() const;
  int64_t getProgramId() const;
};
} // namespace runtime
} // namespace xla
MLIR_DECLARE_EXPLICIT_TYPE_ID(::xla::runtime::HloTraceAttr)

#endif  // GET_ATTRDEF_CLASSES

