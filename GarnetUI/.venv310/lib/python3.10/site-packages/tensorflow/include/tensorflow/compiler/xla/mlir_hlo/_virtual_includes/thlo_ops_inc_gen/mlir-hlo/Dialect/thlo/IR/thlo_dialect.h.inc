/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace thlo {

class THLODialect : public ::mlir::Dialect {
  explicit THLODialect(::mlir::MLIRContext *context);

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~THLODialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("thlo");
  }
};
} // namespace thlo
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::thlo::THLODialect)
