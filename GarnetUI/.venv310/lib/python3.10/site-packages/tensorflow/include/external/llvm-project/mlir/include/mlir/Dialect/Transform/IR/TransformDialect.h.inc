/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace transform {

class TransformDialect : public ::mlir::Dialect {
  explicit TransformDialect(::mlir::MLIRContext *context);

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~TransformDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("transform");
  }

      /// Returns the named PDL constraint functions available in the dialect
      /// as a map from their name to the function.
      const ::llvm::StringMap<::mlir::PDLConstraintFunction> &
      getPDLConstraintHooks() const;

      /// Parses a type registered by this dialect or one of its extensions.
      ::mlir::Type parseType(::mlir::DialectAsmParser &parser) const override;

      /// Prints a type registered by this dialect or one of its extensions.
      void printType(::mlir::Type type,
                     ::mlir::DialectAsmPrinter &printer) const override;

      /// Parser callback for an individual type registered by this dialect or
      /// its extensions.
      using ExtensionTypeParsingHook = ::mlir::Type (*)(::mlir::AsmParser &);

      /// Printer callback for an individual type registered by this dialect or
      /// its extensions.
      using ExtensionTypePrintingHook =
          std::function<void (::mlir::Type, ::mlir::AsmPrinter &)>;

    private:
      /// Registers operations specified as template parameters with this
      /// dialect. Checks that they implement the required interfaces.
      template <typename... OpTys>
      void addOperationsChecked() {
        (addOperationIfNotRegistered<OpTys>(), ...);
      }
      template <typename OpTy>
      void addOperationIfNotRegistered();

      /// Reports a repeated registration error of an op with the given name.
      [[noreturn]] void reportDuplicateOpRegistration(StringRef opName);

      /// Registers the types specified as template parameters with the
      /// Transform dialect. Checks that they meet the requirements for
      /// Transform IR types.
      template <typename... TypeTys>
      void addTypesChecked() {
        (addTypeIfNotRegistered<TypeTys>(), ...);
      }
      template <typename Type>
      void addTypeIfNotRegistered();

      /// Reports a repeated registration error of a type with the given
      /// mnemonic.
      [[noreturn]] void reportDuplicateTypeRegistration(StringRef mnemonic);

      void initializeTypes();

      template <typename, typename...>
      friend class TransformDialectExtension;

      /// Takes ownership of the named PDL constraint function from the given
      /// map and makes them available for use by the operations in the dialect.
      void mergeInPDLMatchHooks(
          ::llvm::StringMap<::mlir::PDLConstraintFunction> &&constraintFns);

      //===----------------------------------------------------------------===//
      // Data fields
      //===----------------------------------------------------------------===//

      /// A container for PDL constraint function that can be used by
      /// operations in this dialect.
      ::mlir::PDLPatternModule pdlMatchHooks;

      /// A map from type mnemonic to its parsing function for the remainder of
      /// the syntax. The parser has access to the mnemonic, so it is used for
      /// further dispatch.
      ::llvm::StringMap<ExtensionTypeParsingHook> typeParsingHooks;

      /// A map from type TypeID to its printing function. No need to do string
      /// lookups when the type is fully constructed.
      ::llvm::DenseMap<::mlir::TypeID, ExtensionTypePrintingHook>
      typePrintingHooks;
  };
} // namespace transform
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::transform::TransformDialect)
