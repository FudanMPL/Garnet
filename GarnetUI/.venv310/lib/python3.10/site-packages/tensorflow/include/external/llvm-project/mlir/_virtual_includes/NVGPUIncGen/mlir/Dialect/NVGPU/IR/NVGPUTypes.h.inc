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
namespace nvgpu {
class DeviceAsyncTokenType;
class DeviceAsyncTokenType : public ::mlir::Type::TypeBase<DeviceAsyncTokenType, ::mlir::Type, ::mlir::TypeStorage> {
public:
  using Base::Base;
public:
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"device.async.token"};
  }

};
} // namespace nvgpu
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::nvgpu::DeviceAsyncTokenType)

#endif  // GET_TYPEDEF_CLASSES

