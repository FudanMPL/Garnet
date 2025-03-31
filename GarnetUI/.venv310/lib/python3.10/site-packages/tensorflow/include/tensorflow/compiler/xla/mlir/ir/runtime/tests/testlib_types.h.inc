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
namespace xla {
namespace runtime {
class CustomArgType;
class ValueType;
class CustomArgType : public ::mlir::Type::TypeBase<CustomArgType, ::mlir::Type, ::mlir::TypeStorage> {
public:
  using Base::Base;
public:
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"custom_arg"};
  }

};
class ValueType : public ::mlir::Type::TypeBase<ValueType, ::mlir::Type, ::mlir::TypeStorage> {
public:
  using Base::Base;
public:
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"value"};
  }

};
} // namespace runtime
} // namespace xla
MLIR_DECLARE_EXPLICIT_TYPE_ID(::xla::runtime::CustomArgType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::xla::runtime::ValueType)

#endif  // GET_TYPEDEF_CLASSES

