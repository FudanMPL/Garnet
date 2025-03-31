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
class PairOfDimsAttr;
class EnumType2Attr;
class EnumTypeAttr;
namespace detail {
struct PairOfDimsAttrStorage;
} // namespace detail
class PairOfDimsAttr : public ::mlir::Attribute::AttrBase<PairOfDimsAttr, ::mlir::Attribute, detail::PairOfDimsAttrStorage> {
public:
  using Base::Base;
public:
  static PairOfDimsAttr get(::mlir::MLIRContext *context, int64_t rank, ::llvm::ArrayRef<int64_t> a, ::llvm::ArrayRef<int64_t> b);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"pair_of_dims"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  int64_t getRank() const;
  ::llvm::ArrayRef<int64_t> getA() const;
  ::llvm::ArrayRef<int64_t> getB() const;
};
namespace detail {
struct EnumType2AttrStorage;
} // namespace detail
class EnumType2Attr : public ::mlir::Attribute::AttrBase<EnumType2Attr, ::mlir::Attribute, detail::EnumType2AttrStorage> {
public:
  using Base::Base;
public:
  static EnumType2Attr get(::mlir::MLIRContext *context, ::xla::runtime::EnumType2 value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"another_enum_type"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::xla::runtime::EnumType2 getValue() const;
};
namespace detail {
struct EnumTypeAttrStorage;
} // namespace detail
class EnumTypeAttr : public ::mlir::Attribute::AttrBase<EnumTypeAttr, ::mlir::Attribute, detail::EnumTypeAttrStorage> {
public:
  using Base::Base;
public:
  static EnumTypeAttr get(::mlir::MLIRContext *context, ::xla::runtime::EnumType value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"enum_type"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::xla::runtime::EnumType getValue() const;
};
} // namespace runtime
} // namespace xla
MLIR_DECLARE_EXPLICIT_TYPE_ID(::xla::runtime::PairOfDimsAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::xla::runtime::EnumType2Attr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::xla::runtime::EnumTypeAttr)

#endif  // GET_ATTRDEF_CLASSES

