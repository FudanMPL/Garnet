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
namespace sparse_tensor {
class SparseTensorEncodingAttr;
namespace detail {
struct SparseTensorEncodingAttrStorage;
} // namespace detail
class SparseTensorEncodingAttr : public ::mlir::Attribute::AttrBase<SparseTensorEncodingAttr, ::mlir::Attribute, detail::SparseTensorEncodingAttrStorage, ::mlir::VerifiableTensorEncoding::Trait> {
public:
  using Base::Base;
  // Dimension level types. By default, each type has the unique and
  // ordered properties. Alternatives properties are indicated by
  // Nu (not-unique) and No (not-ordered).
  //
  // TODO: separate type and property in encoding
  //
  enum class DimLevelType : uint8_t {
    Dense            = 4,  // 0b001_00
    Compressed       = 8,  // 0b010_00
    CompressedNu     = 9,  // 0b010_01
    CompressedNo     = 10, // 0b010_10
    CompressedNuNo   = 11, // 0b010_11
    Singleton        = 16, // 0b100_00
    SingletonNu      = 17, // 0b100_01
    SingletonNo      = 18, // 0b100_10
    SingletonNuNo    = 19, // 0b100_11
  };
  using Base::getChecked;
public:
  static SparseTensorEncodingAttr get(::mlir::MLIRContext *context, ::llvm::ArrayRef<SparseTensorEncodingAttr::DimLevelType> dimLevelType, AffineMap dimOrdering, AffineMap higherOrdering, unsigned pointerBitWidth, unsigned indexBitWidth);
  static SparseTensorEncodingAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, ::llvm::ArrayRef<SparseTensorEncodingAttr::DimLevelType> dimLevelType, AffineMap dimOrdering, AffineMap higherOrdering, unsigned pointerBitWidth, unsigned indexBitWidth);
  static ::mlir::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::ArrayRef<SparseTensorEncodingAttr::DimLevelType> dimLevelType, AffineMap dimOrdering, AffineMap higherOrdering, unsigned pointerBitWidth, unsigned indexBitWidth);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"encoding"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::llvm::ArrayRef<SparseTensorEncodingAttr::DimLevelType> getDimLevelType() const;
  AffineMap getDimOrdering() const;
  AffineMap getHigherOrdering() const;
  unsigned getPointerBitWidth() const;
  unsigned getIndexBitWidth() const;
  ::mlir::LogicalResult verifyEncoding(ArrayRef<int64_t> shape, Type elementType, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) const;
};
} // namespace sparse_tensor
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::sparse_tensor::SparseTensorEncodingAttr)

#endif  // GET_ATTRDEF_CLASSES

