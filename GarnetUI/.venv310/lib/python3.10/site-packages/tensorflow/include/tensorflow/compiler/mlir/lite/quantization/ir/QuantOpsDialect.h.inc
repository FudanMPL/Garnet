/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace quantfork {

class QuantizationForkDialect : public ::mlir::Dialect {
  explicit QuantizationForkDialect(::mlir::MLIRContext *context);

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~QuantizationForkDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("quantfork");
  }
};
} // namespace quantfork
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::quantfork::QuantizationForkDialect)
