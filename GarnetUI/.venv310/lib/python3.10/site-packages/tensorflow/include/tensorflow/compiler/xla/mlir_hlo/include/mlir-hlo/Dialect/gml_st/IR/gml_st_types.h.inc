/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* TypeDef Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_TYPEDEF_CLASSES
#undef GET_TYPEDEF_CLASSES


namespace mlir {
class AsmParser;
class AsmPrinter;
} // namespace mlir
namespace mlir {
namespace gml_st {
class TileType;
namespace detail {
struct TileTypeStorage;
} // namespace detail
class TileType : public ::mlir::Type::TypeBase<TileType, ::mlir::Type, detail::TileTypeStorage> {
public:
  using Base::Base;
  unsigned getRank() const { return getShape().size(); }
  bool hasStaticShape() const {
    return llvm::none_of(getShape(), ShapedType::isDynamic);
  }
  int64_t getNumElements() const {
    return ShapedType::getNumElements(getShape());
  }
public:
  static TileType get(::mlir::MLIRContext *context, ::llvm::ArrayRef<int64_t> shape);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"tile"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::llvm::ArrayRef<int64_t> getShape() const;
};
} // namespace gml_st
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::gml_st::TileType)

#endif  // GET_TYPEDEF_CLASSES

