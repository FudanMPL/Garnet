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
namespace lmhlo {
class CustomCallTargetArgMappingAttr;
namespace detail {
struct CustomCallTargetArgMappingAttrStorage;
} // namespace detail
class CustomCallTargetArgMappingAttr : public ::mlir::Attribute::AttrBase<CustomCallTargetArgMappingAttr, ::mlir::Attribute, detail::CustomCallTargetArgMappingAttrStorage> {
public:
  using Base::Base;
public:
  static CustomCallTargetArgMappingAttr get(::mlir::MLIRContext *context, int64_t num_args, int64_t num_results, ::llvm::ArrayRef<int64_t> args_to_target_args, ::llvm::ArrayRef<int64_t> results_to_target_results);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"custom_call_target_arg_mapping"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  int64_t getNumArgs() const;
  int64_t getNumResults() const;
  ::llvm::ArrayRef<int64_t> getArgsToTargetArgs() const;
  ::llvm::ArrayRef<int64_t> getResultsToTargetResults() const;
};
} // namespace lmhlo
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::lmhlo::CustomCallTargetArgMappingAttr)

#endif  // GET_ATTRDEF_CLASSES

