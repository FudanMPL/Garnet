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
class ExecutionContextType;
class OpaqueType;
class StatusType;
class ExecutionContextType : public ::mlir::Type::TypeBase<ExecutionContextType, ::mlir::Type, ::mlir::TypeStorage> {
public:
  using Base::Base;
public:
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"execution_context"};
  }

};
class OpaqueType : public ::mlir::Type::TypeBase<OpaqueType, ::mlir::Type, ::mlir::TypeStorage> {
public:
  using Base::Base;
public:
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"opaque"};
  }

};
class StatusType : public ::mlir::Type::TypeBase<StatusType, ::mlir::Type, ::mlir::TypeStorage> {
public:
  using Base::Base;
public:
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"status"};
  }

};
} // namespace runtime
} // namespace xla
MLIR_DECLARE_EXPLICIT_TYPE_ID(::xla::runtime::ExecutionContextType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::xla::runtime::OpaqueType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::xla::runtime::StatusType)

#endif  // GET_TYPEDEF_CLASSES

