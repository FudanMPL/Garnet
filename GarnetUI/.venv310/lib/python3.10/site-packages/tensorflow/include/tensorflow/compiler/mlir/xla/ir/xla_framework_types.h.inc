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
namespace xla_framework {
class BufferType;
class BufferType : public ::mlir::Type::TypeBase<BufferType, ::mlir::Type, ::mlir::TypeStorage> {
public:
  using Base::Base;
public:
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"buffer"};
  }

};
} // namespace xla_framework
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::xla_framework::BufferType)

#endif  // GET_TYPEDEF_CLASSES

