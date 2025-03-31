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
namespace transform {
class AnyOpType;
class OperationType;
class AnyOpType : public ::mlir::Type::TypeBase<AnyOpType, ::mlir::Type, ::mlir::TypeStorage, ::mlir::transform::TransformTypeInterface::Trait> {
public:
  using Base::Base;
public:
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"any_op"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::DiagnosedSilenceableFailure checkPayload(::mlir::Location loc, ::mlir::ArrayRef<::mlir::Operation *> payload) const;
};
namespace detail {
struct OperationTypeStorage;
} // namespace detail
class OperationType : public ::mlir::Type::TypeBase<OperationType, ::mlir::Type, detail::OperationTypeStorage, ::mlir::transform::TransformTypeInterface::Trait> {
public:
  using Base::Base;
public:
  static OperationType get(::mlir::MLIRContext *context, ::llvm::StringRef operation_name);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"op"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::llvm::StringRef getOperationName() const;
  ::mlir::DiagnosedSilenceableFailure checkPayload(::mlir::Location loc, ::mlir::ArrayRef<::mlir::Operation *> payload) const;
};
} // namespace transform
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::transform::AnyOpType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::transform::OperationType)

#endif  // GET_TYPEDEF_CLASSES

