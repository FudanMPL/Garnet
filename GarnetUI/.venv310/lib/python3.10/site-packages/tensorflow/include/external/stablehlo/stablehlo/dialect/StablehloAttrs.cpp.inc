/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_LIST
#undef GET_ATTRDEF_LIST

::mlir::stablehlo::ArgResultAliasAttr,
::mlir::stablehlo::ChannelHandleAttr,
::mlir::stablehlo::ComparisonDirectionAttr,
::mlir::stablehlo::ComparisonTypeAttr,
::mlir::stablehlo::ConvDimensionNumbersAttr,
::mlir::stablehlo::DotDimensionNumbersAttr,
::mlir::stablehlo::FftTypeAttr,
::mlir::stablehlo::GatherDimensionNumbersAttr,
::mlir::stablehlo::PrecisionAttr,
::mlir::stablehlo::RngAlgorithmAttr,
::mlir::stablehlo::RngDistributionAttr,
::mlir::stablehlo::ScatterDimensionNumbersAttr,
::mlir::stablehlo::TransposeAttr,
::mlir::stablehlo::TypeExtensionsAttr

#endif  // GET_ATTRDEF_LIST

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES

static ::mlir::OptionalParseResult generatedAttributeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type type, ::mlir::Attribute &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    .Case(::mlir::stablehlo::ArgResultAliasAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::stablehlo::ArgResultAliasAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::stablehlo::ChannelHandleAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::stablehlo::ChannelHandleAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::stablehlo::ComparisonDirectionAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::stablehlo::ComparisonDirectionAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::stablehlo::ComparisonTypeAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::stablehlo::ComparisonTypeAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::stablehlo::ConvDimensionNumbersAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::stablehlo::ConvDimensionNumbersAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::stablehlo::DotDimensionNumbersAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::stablehlo::DotDimensionNumbersAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::stablehlo::FftTypeAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::stablehlo::FftTypeAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::stablehlo::GatherDimensionNumbersAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::stablehlo::GatherDimensionNumbersAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::stablehlo::PrecisionAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::stablehlo::PrecisionAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::stablehlo::RngAlgorithmAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::stablehlo::RngAlgorithmAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::stablehlo::RngDistributionAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::stablehlo::RngDistributionAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::stablehlo::ScatterDimensionNumbersAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::stablehlo::ScatterDimensionNumbersAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::stablehlo::TransposeAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::stablehlo::TransposeAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::stablehlo::TypeExtensionsAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::stablehlo::TypeExtensionsAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
      *mnemonic = keyword;
      return llvm::None;
    });
}

static ::mlir::LogicalResult generatedAttributePrinter(::mlir::Attribute def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Attribute, ::mlir::LogicalResult>(def)    .Case<::mlir::stablehlo::ArgResultAliasAttr>([&](auto t) {
      printer << ::mlir::stablehlo::ArgResultAliasAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::stablehlo::ChannelHandleAttr>([&](auto t) {
      printer << ::mlir::stablehlo::ChannelHandleAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::stablehlo::ComparisonDirectionAttr>([&](auto t) {
      printer << ::mlir::stablehlo::ComparisonDirectionAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::stablehlo::ComparisonTypeAttr>([&](auto t) {
      printer << ::mlir::stablehlo::ComparisonTypeAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::stablehlo::ConvDimensionNumbersAttr>([&](auto t) {
      printer << ::mlir::stablehlo::ConvDimensionNumbersAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::stablehlo::DotDimensionNumbersAttr>([&](auto t) {
      printer << ::mlir::stablehlo::DotDimensionNumbersAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::stablehlo::FftTypeAttr>([&](auto t) {
      printer << ::mlir::stablehlo::FftTypeAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::stablehlo::GatherDimensionNumbersAttr>([&](auto t) {
      printer << ::mlir::stablehlo::GatherDimensionNumbersAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::stablehlo::PrecisionAttr>([&](auto t) {
      printer << ::mlir::stablehlo::PrecisionAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::stablehlo::RngAlgorithmAttr>([&](auto t) {
      printer << ::mlir::stablehlo::RngAlgorithmAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::stablehlo::RngDistributionAttr>([&](auto t) {
      printer << ::mlir::stablehlo::RngDistributionAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::stablehlo::ScatterDimensionNumbersAttr>([&](auto t) {
      printer << ::mlir::stablehlo::ScatterDimensionNumbersAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::stablehlo::TransposeAttr>([&](auto t) {
      printer << ::mlir::stablehlo::TransposeAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::stablehlo::TypeExtensionsAttr>([&](auto t) {
      printer << ::mlir::stablehlo::TypeExtensionsAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace mlir {
namespace stablehlo {
namespace detail {
struct ArgResultAliasAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::llvm::ArrayRef<int64_t>, int64_t, ::llvm::ArrayRef<int64_t>, bool>;
  ArgResultAliasAttrStorage(::llvm::ArrayRef<int64_t> argTupleIndices, int64_t resultIndex, ::llvm::ArrayRef<int64_t> resultTupleIndices, bool isMustAlias) : argTupleIndices(argTupleIndices), resultIndex(resultIndex), resultTupleIndices(resultTupleIndices), isMustAlias(isMustAlias) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (argTupleIndices == std::get<0>(tblgenKey)) && (resultIndex == std::get<1>(tblgenKey)) && (resultTupleIndices == std::get<2>(tblgenKey)) && (isMustAlias == std::get<3>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey), std::get<3>(tblgenKey));
  }

  static ArgResultAliasAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto argTupleIndices = std::get<0>(tblgenKey);
    auto resultIndex = std::get<1>(tblgenKey);
    auto resultTupleIndices = std::get<2>(tblgenKey);
    auto isMustAlias = std::get<3>(tblgenKey);
    argTupleIndices = allocator.copyInto(argTupleIndices);
    resultTupleIndices = allocator.copyInto(resultTupleIndices);
    return new (allocator.allocate<ArgResultAliasAttrStorage>()) ArgResultAliasAttrStorage(argTupleIndices, resultIndex, resultTupleIndices, isMustAlias);
  }

  ::llvm::ArrayRef<int64_t> argTupleIndices;
  int64_t resultIndex;
  ::llvm::ArrayRef<int64_t> resultTupleIndices;
  bool isMustAlias;
};
} // namespace detail
ArgResultAliasAttr ArgResultAliasAttr::get(::mlir::MLIRContext *context, ::llvm::ArrayRef<int64_t> argTupleIndices, int64_t resultIndex, ::llvm::ArrayRef<int64_t> resultTupleIndices, bool isMustAlias) {
  return Base::get(context, argTupleIndices, resultIndex, resultTupleIndices, isMustAlias);
}

::llvm::ArrayRef<int64_t> ArgResultAliasAttr::getArgTupleIndices() const {
  return getImpl()->argTupleIndices;
}

int64_t ArgResultAliasAttr::getResultIndex() const {
  return getImpl()->resultIndex;
}

::llvm::ArrayRef<int64_t> ArgResultAliasAttr::getResultTupleIndices() const {
  return getImpl()->resultTupleIndices;
}

bool ArgResultAliasAttr::getIsMustAlias() const {
  return getImpl()->isMustAlias;
}

} // namespace stablehlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::stablehlo::ArgResultAliasAttr)
namespace mlir {
namespace stablehlo {
namespace detail {
struct ChannelHandleAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<int64_t, int64_t>;
  ChannelHandleAttrStorage(int64_t handle, int64_t type) : handle(handle), type(type) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (handle == std::get<0>(tblgenKey)) && (type == std::get<1>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey));
  }

  static ChannelHandleAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto handle = std::get<0>(tblgenKey);
    auto type = std::get<1>(tblgenKey);
    return new (allocator.allocate<ChannelHandleAttrStorage>()) ChannelHandleAttrStorage(handle, type);
  }

  int64_t handle;
  int64_t type;
};
} // namespace detail
ChannelHandleAttr ChannelHandleAttr::get(::mlir::MLIRContext *context, int64_t handle, int64_t type) {
  return Base::get(context, handle, type);
}

::mlir::Attribute ChannelHandleAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<int64_t> _result_handle;
  ::mlir::FailureOr<int64_t> _result_type;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse parameter struct
  bool _seen_handle = false;
  bool _seen_type = false;
  {
    const auto _loop_body = [&](::llvm::StringRef _paramKey) -> bool {
      // Parse literal '='
      if (odsParser.parseEqual()) return {};
      if (!_seen_handle && _paramKey == "handle") {
        _seen_handle = true;

        // Parse variable 'handle'
        _result_handle = ::mlir::FieldParser<int64_t>::parse(odsParser);
        if (::mlir::failed(_result_handle)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse StableHLO_ChannelHandle parameter 'handle' which is to be a `int64_t`");
          return {};
        }
      } else if (!_seen_type && _paramKey == "type") {
        _seen_type = true;

        // Parse variable 'type'
        _result_type = ::mlir::FieldParser<int64_t>::parse(odsParser);
        if (::mlir::failed(_result_type)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse StableHLO_ChannelHandle parameter 'type' which is to be a `int64_t`");
          return {};
        }
      } else {
        odsParser.emitError(odsParser.getCurrentLocation(), "duplicate or unknown struct parameter name: ") << _paramKey;
        return {};
      }
      return true;
    };
    for (unsigned odsStructIndex = 0; odsStructIndex < 2; ++odsStructIndex) {
      ::llvm::StringRef _paramKey;
      if (odsParser.parseKeyword(&_paramKey)) {
        odsParser.emitError(odsParser.getCurrentLocation(),
                           "expected a parameter name in struct");
        return {};
      }
      if (!_loop_body(_paramKey)) return {};
      if ((odsStructIndex != 2 - 1) && odsParser.parseComma())
        return {};
    }
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_handle));
  assert(::mlir::succeeded(_result_type));
  return ChannelHandleAttr::get(odsParser.getContext(),
      int64_t((*_result_handle)),
      int64_t((*_result_type)));
}

void ChannelHandleAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  {
    bool _firstPrinted = true;
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << "handle = ";
    odsPrinter.printStrippedAttrOrType(getHandle());
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << "type = ";
    odsPrinter.printStrippedAttrOrType(getType());
  }
  odsPrinter << ">";
}

int64_t ChannelHandleAttr::getHandle() const {
  return getImpl()->handle;
}

int64_t ChannelHandleAttr::getType() const {
  return getImpl()->type;
}

} // namespace stablehlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::stablehlo::ChannelHandleAttr)
namespace mlir {
namespace stablehlo {
namespace detail {
struct ComparisonDirectionAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::stablehlo::ComparisonDirection>;
  ComparisonDirectionAttrStorage(::mlir::stablehlo::ComparisonDirection value) : value(value) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static ComparisonDirectionAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto value = std::get<0>(tblgenKey);
    return new (allocator.allocate<ComparisonDirectionAttrStorage>()) ComparisonDirectionAttrStorage(value);
  }

  ::mlir::stablehlo::ComparisonDirection value;
};
} // namespace detail
ComparisonDirectionAttr ComparisonDirectionAttr::get(::mlir::MLIRContext *context, ::mlir::stablehlo::ComparisonDirection value) {
  return Base::get(context, value);
}

::mlir::Attribute ComparisonDirectionAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::stablehlo::ComparisonDirection> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::stablehlo::ComparisonDirection> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::stablehlo::symbolizeComparisonDirection(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::mlir::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::stablehlo::ComparisonDirection" << " to be one of: " << "EQ" << ", " << "NE" << ", " << "GE" << ", " << "GT" << ", " << "LE" << ", " << "LT")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse StableHLO_ComparisonDirectionAttr parameter 'value' which is to be a `::mlir::stablehlo::ComparisonDirection`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return ComparisonDirectionAttr::get(odsParser.getContext(),
      ::mlir::stablehlo::ComparisonDirection((*_result_value)));
}

void ComparisonDirectionAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyComparisonDirection(getValue());
}

::mlir::stablehlo::ComparisonDirection ComparisonDirectionAttr::getValue() const {
  return getImpl()->value;
}

} // namespace stablehlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::stablehlo::ComparisonDirectionAttr)
namespace mlir {
namespace stablehlo {
namespace detail {
struct ComparisonTypeAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::stablehlo::ComparisonType>;
  ComparisonTypeAttrStorage(::mlir::stablehlo::ComparisonType value) : value(value) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static ComparisonTypeAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto value = std::get<0>(tblgenKey);
    return new (allocator.allocate<ComparisonTypeAttrStorage>()) ComparisonTypeAttrStorage(value);
  }

  ::mlir::stablehlo::ComparisonType value;
};
} // namespace detail
ComparisonTypeAttr ComparisonTypeAttr::get(::mlir::MLIRContext *context, ::mlir::stablehlo::ComparisonType value) {
  return Base::get(context, value);
}

::mlir::Attribute ComparisonTypeAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::stablehlo::ComparisonType> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::stablehlo::ComparisonType> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::stablehlo::symbolizeComparisonType(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::mlir::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::stablehlo::ComparisonType" << " to be one of: " << "NOTYPE" << ", " << "FLOAT" << ", " << "TOTALORDER" << ", " << "SIGNED" << ", " << "UNSIGNED")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse StableHLO_ComparisonTypeAttr parameter 'value' which is to be a `::mlir::stablehlo::ComparisonType`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return ComparisonTypeAttr::get(odsParser.getContext(),
      ::mlir::stablehlo::ComparisonType((*_result_value)));
}

void ComparisonTypeAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyComparisonType(getValue());
}

::mlir::stablehlo::ComparisonType ComparisonTypeAttr::getValue() const {
  return getImpl()->value;
}

} // namespace stablehlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::stablehlo::ComparisonTypeAttr)
namespace mlir {
namespace stablehlo {
namespace detail {
struct ConvDimensionNumbersAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<int64_t, int64_t, ::llvm::ArrayRef<int64_t>, int64_t, int64_t, ::llvm::ArrayRef<int64_t>, int64_t, int64_t, ::llvm::ArrayRef<int64_t>>;
  ConvDimensionNumbersAttrStorage(int64_t inputBatchDimension, int64_t inputFeatureDimension, ::llvm::ArrayRef<int64_t> inputSpatialDimensions, int64_t kernelInputFeatureDimension, int64_t kernelOutputFeatureDimension, ::llvm::ArrayRef<int64_t> kernelSpatialDimensions, int64_t outputBatchDimension, int64_t outputFeatureDimension, ::llvm::ArrayRef<int64_t> outputSpatialDimensions) : inputBatchDimension(inputBatchDimension), inputFeatureDimension(inputFeatureDimension), inputSpatialDimensions(inputSpatialDimensions), kernelInputFeatureDimension(kernelInputFeatureDimension), kernelOutputFeatureDimension(kernelOutputFeatureDimension), kernelSpatialDimensions(kernelSpatialDimensions), outputBatchDimension(outputBatchDimension), outputFeatureDimension(outputFeatureDimension), outputSpatialDimensions(outputSpatialDimensions) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (inputBatchDimension == std::get<0>(tblgenKey)) && (inputFeatureDimension == std::get<1>(tblgenKey)) && (inputSpatialDimensions == std::get<2>(tblgenKey)) && (kernelInputFeatureDimension == std::get<3>(tblgenKey)) && (kernelOutputFeatureDimension == std::get<4>(tblgenKey)) && (kernelSpatialDimensions == std::get<5>(tblgenKey)) && (outputBatchDimension == std::get<6>(tblgenKey)) && (outputFeatureDimension == std::get<7>(tblgenKey)) && (outputSpatialDimensions == std::get<8>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey), std::get<3>(tblgenKey), std::get<4>(tblgenKey), std::get<5>(tblgenKey), std::get<6>(tblgenKey), std::get<7>(tblgenKey), std::get<8>(tblgenKey));
  }

  static ConvDimensionNumbersAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto inputBatchDimension = std::get<0>(tblgenKey);
    auto inputFeatureDimension = std::get<1>(tblgenKey);
    auto inputSpatialDimensions = std::get<2>(tblgenKey);
    auto kernelInputFeatureDimension = std::get<3>(tblgenKey);
    auto kernelOutputFeatureDimension = std::get<4>(tblgenKey);
    auto kernelSpatialDimensions = std::get<5>(tblgenKey);
    auto outputBatchDimension = std::get<6>(tblgenKey);
    auto outputFeatureDimension = std::get<7>(tblgenKey);
    auto outputSpatialDimensions = std::get<8>(tblgenKey);
    inputSpatialDimensions = allocator.copyInto(inputSpatialDimensions);
    kernelSpatialDimensions = allocator.copyInto(kernelSpatialDimensions);
    outputSpatialDimensions = allocator.copyInto(outputSpatialDimensions);
    return new (allocator.allocate<ConvDimensionNumbersAttrStorage>()) ConvDimensionNumbersAttrStorage(inputBatchDimension, inputFeatureDimension, inputSpatialDimensions, kernelInputFeatureDimension, kernelOutputFeatureDimension, kernelSpatialDimensions, outputBatchDimension, outputFeatureDimension, outputSpatialDimensions);
  }

  int64_t inputBatchDimension;
  int64_t inputFeatureDimension;
  ::llvm::ArrayRef<int64_t> inputSpatialDimensions;
  int64_t kernelInputFeatureDimension;
  int64_t kernelOutputFeatureDimension;
  ::llvm::ArrayRef<int64_t> kernelSpatialDimensions;
  int64_t outputBatchDimension;
  int64_t outputFeatureDimension;
  ::llvm::ArrayRef<int64_t> outputSpatialDimensions;
};
} // namespace detail
ConvDimensionNumbersAttr ConvDimensionNumbersAttr::get(::mlir::MLIRContext *context, int64_t inputBatchDimension, int64_t inputFeatureDimension, ::llvm::ArrayRef<int64_t> inputSpatialDimensions, int64_t kernelInputFeatureDimension, int64_t kernelOutputFeatureDimension, ::llvm::ArrayRef<int64_t> kernelSpatialDimensions, int64_t outputBatchDimension, int64_t outputFeatureDimension, ::llvm::ArrayRef<int64_t> outputSpatialDimensions) {
  return Base::get(context, inputBatchDimension, inputFeatureDimension, inputSpatialDimensions, kernelInputFeatureDimension, kernelOutputFeatureDimension, kernelSpatialDimensions, outputBatchDimension, outputFeatureDimension, outputSpatialDimensions);
}

int64_t ConvDimensionNumbersAttr::getInputBatchDimension() const {
  return getImpl()->inputBatchDimension;
}

int64_t ConvDimensionNumbersAttr::getInputFeatureDimension() const {
  return getImpl()->inputFeatureDimension;
}

::llvm::ArrayRef<int64_t> ConvDimensionNumbersAttr::getInputSpatialDimensions() const {
  return getImpl()->inputSpatialDimensions;
}

int64_t ConvDimensionNumbersAttr::getKernelInputFeatureDimension() const {
  return getImpl()->kernelInputFeatureDimension;
}

int64_t ConvDimensionNumbersAttr::getKernelOutputFeatureDimension() const {
  return getImpl()->kernelOutputFeatureDimension;
}

::llvm::ArrayRef<int64_t> ConvDimensionNumbersAttr::getKernelSpatialDimensions() const {
  return getImpl()->kernelSpatialDimensions;
}

int64_t ConvDimensionNumbersAttr::getOutputBatchDimension() const {
  return getImpl()->outputBatchDimension;
}

int64_t ConvDimensionNumbersAttr::getOutputFeatureDimension() const {
  return getImpl()->outputFeatureDimension;
}

::llvm::ArrayRef<int64_t> ConvDimensionNumbersAttr::getOutputSpatialDimensions() const {
  return getImpl()->outputSpatialDimensions;
}

} // namespace stablehlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::stablehlo::ConvDimensionNumbersAttr)
namespace mlir {
namespace stablehlo {
namespace detail {
struct DotDimensionNumbersAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::llvm::ArrayRef<int64_t>, ::llvm::ArrayRef<int64_t>, ::llvm::ArrayRef<int64_t>, ::llvm::ArrayRef<int64_t>>;
  DotDimensionNumbersAttrStorage(::llvm::ArrayRef<int64_t> lhsBatchingDimensions, ::llvm::ArrayRef<int64_t> rhsBatchingDimensions, ::llvm::ArrayRef<int64_t> lhsContractingDimensions, ::llvm::ArrayRef<int64_t> rhsContractingDimensions) : lhsBatchingDimensions(lhsBatchingDimensions), rhsBatchingDimensions(rhsBatchingDimensions), lhsContractingDimensions(lhsContractingDimensions), rhsContractingDimensions(rhsContractingDimensions) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (lhsBatchingDimensions == std::get<0>(tblgenKey)) && (rhsBatchingDimensions == std::get<1>(tblgenKey)) && (lhsContractingDimensions == std::get<2>(tblgenKey)) && (rhsContractingDimensions == std::get<3>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey), std::get<3>(tblgenKey));
  }

  static DotDimensionNumbersAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto lhsBatchingDimensions = std::get<0>(tblgenKey);
    auto rhsBatchingDimensions = std::get<1>(tblgenKey);
    auto lhsContractingDimensions = std::get<2>(tblgenKey);
    auto rhsContractingDimensions = std::get<3>(tblgenKey);
    lhsBatchingDimensions = allocator.copyInto(lhsBatchingDimensions);
    rhsBatchingDimensions = allocator.copyInto(rhsBatchingDimensions);
    lhsContractingDimensions = allocator.copyInto(lhsContractingDimensions);
    rhsContractingDimensions = allocator.copyInto(rhsContractingDimensions);
    return new (allocator.allocate<DotDimensionNumbersAttrStorage>()) DotDimensionNumbersAttrStorage(lhsBatchingDimensions, rhsBatchingDimensions, lhsContractingDimensions, rhsContractingDimensions);
  }

  ::llvm::ArrayRef<int64_t> lhsBatchingDimensions;
  ::llvm::ArrayRef<int64_t> rhsBatchingDimensions;
  ::llvm::ArrayRef<int64_t> lhsContractingDimensions;
  ::llvm::ArrayRef<int64_t> rhsContractingDimensions;
};
} // namespace detail
DotDimensionNumbersAttr DotDimensionNumbersAttr::get(::mlir::MLIRContext *context, ::llvm::ArrayRef<int64_t> lhsBatchingDimensions, ::llvm::ArrayRef<int64_t> rhsBatchingDimensions, ::llvm::ArrayRef<int64_t> lhsContractingDimensions, ::llvm::ArrayRef<int64_t> rhsContractingDimensions) {
  return Base::get(context, lhsBatchingDimensions, rhsBatchingDimensions, lhsContractingDimensions, rhsContractingDimensions);
}

::llvm::ArrayRef<int64_t> DotDimensionNumbersAttr::getLhsBatchingDimensions() const {
  return getImpl()->lhsBatchingDimensions;
}

::llvm::ArrayRef<int64_t> DotDimensionNumbersAttr::getRhsBatchingDimensions() const {
  return getImpl()->rhsBatchingDimensions;
}

::llvm::ArrayRef<int64_t> DotDimensionNumbersAttr::getLhsContractingDimensions() const {
  return getImpl()->lhsContractingDimensions;
}

::llvm::ArrayRef<int64_t> DotDimensionNumbersAttr::getRhsContractingDimensions() const {
  return getImpl()->rhsContractingDimensions;
}

} // namespace stablehlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::stablehlo::DotDimensionNumbersAttr)
namespace mlir {
namespace stablehlo {
namespace detail {
struct FftTypeAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::stablehlo::FftType>;
  FftTypeAttrStorage(::mlir::stablehlo::FftType value) : value(value) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static FftTypeAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto value = std::get<0>(tblgenKey);
    return new (allocator.allocate<FftTypeAttrStorage>()) FftTypeAttrStorage(value);
  }

  ::mlir::stablehlo::FftType value;
};
} // namespace detail
FftTypeAttr FftTypeAttr::get(::mlir::MLIRContext *context, ::mlir::stablehlo::FftType value) {
  return Base::get(context, value);
}

::mlir::Attribute FftTypeAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::stablehlo::FftType> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::stablehlo::FftType> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::stablehlo::symbolizeFftType(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::mlir::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::stablehlo::FftType" << " to be one of: " << "FFT" << ", " << "IFFT" << ", " << "RFFT" << ", " << "IRFFT")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse StableHLO_FftTypeAttr parameter 'value' which is to be a `::mlir::stablehlo::FftType`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return FftTypeAttr::get(odsParser.getContext(),
      ::mlir::stablehlo::FftType((*_result_value)));
}

void FftTypeAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyFftType(getValue());
}

::mlir::stablehlo::FftType FftTypeAttr::getValue() const {
  return getImpl()->value;
}

} // namespace stablehlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::stablehlo::FftTypeAttr)
namespace mlir {
namespace stablehlo {
namespace detail {
struct GatherDimensionNumbersAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::llvm::ArrayRef<int64_t>, ::llvm::ArrayRef<int64_t>, ::llvm::ArrayRef<int64_t>, int64_t>;
  GatherDimensionNumbersAttrStorage(::llvm::ArrayRef<int64_t> offsetDims, ::llvm::ArrayRef<int64_t> collapsedSliceDims, ::llvm::ArrayRef<int64_t> startIndexMap, int64_t indexVectorDim) : offsetDims(offsetDims), collapsedSliceDims(collapsedSliceDims), startIndexMap(startIndexMap), indexVectorDim(indexVectorDim) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (offsetDims == std::get<0>(tblgenKey)) && (collapsedSliceDims == std::get<1>(tblgenKey)) && (startIndexMap == std::get<2>(tblgenKey)) && (indexVectorDim == std::get<3>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey), std::get<3>(tblgenKey));
  }

  static GatherDimensionNumbersAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto offsetDims = std::get<0>(tblgenKey);
    auto collapsedSliceDims = std::get<1>(tblgenKey);
    auto startIndexMap = std::get<2>(tblgenKey);
    auto indexVectorDim = std::get<3>(tblgenKey);
    offsetDims = allocator.copyInto(offsetDims);
    collapsedSliceDims = allocator.copyInto(collapsedSliceDims);
    startIndexMap = allocator.copyInto(startIndexMap);
    return new (allocator.allocate<GatherDimensionNumbersAttrStorage>()) GatherDimensionNumbersAttrStorage(offsetDims, collapsedSliceDims, startIndexMap, indexVectorDim);
  }

  ::llvm::ArrayRef<int64_t> offsetDims;
  ::llvm::ArrayRef<int64_t> collapsedSliceDims;
  ::llvm::ArrayRef<int64_t> startIndexMap;
  int64_t indexVectorDim;
};
} // namespace detail
GatherDimensionNumbersAttr GatherDimensionNumbersAttr::get(::mlir::MLIRContext *context, ::llvm::ArrayRef<int64_t> offsetDims, ::llvm::ArrayRef<int64_t> collapsedSliceDims, ::llvm::ArrayRef<int64_t> startIndexMap, int64_t indexVectorDim) {
  return Base::get(context, offsetDims, collapsedSliceDims, startIndexMap, indexVectorDim);
}

::llvm::ArrayRef<int64_t> GatherDimensionNumbersAttr::getOffsetDims() const {
  return getImpl()->offsetDims;
}

::llvm::ArrayRef<int64_t> GatherDimensionNumbersAttr::getCollapsedSliceDims() const {
  return getImpl()->collapsedSliceDims;
}

::llvm::ArrayRef<int64_t> GatherDimensionNumbersAttr::getStartIndexMap() const {
  return getImpl()->startIndexMap;
}

int64_t GatherDimensionNumbersAttr::getIndexVectorDim() const {
  return getImpl()->indexVectorDim;
}

} // namespace stablehlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::stablehlo::GatherDimensionNumbersAttr)
namespace mlir {
namespace stablehlo {
namespace detail {
struct PrecisionAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::stablehlo::Precision>;
  PrecisionAttrStorage(::mlir::stablehlo::Precision value) : value(value) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static PrecisionAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto value = std::get<0>(tblgenKey);
    return new (allocator.allocate<PrecisionAttrStorage>()) PrecisionAttrStorage(value);
  }

  ::mlir::stablehlo::Precision value;
};
} // namespace detail
PrecisionAttr PrecisionAttr::get(::mlir::MLIRContext *context, ::mlir::stablehlo::Precision value) {
  return Base::get(context, value);
}

::mlir::Attribute PrecisionAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::stablehlo::Precision> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::stablehlo::Precision> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::stablehlo::symbolizePrecision(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::mlir::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::stablehlo::Precision" << " to be one of: " << "DEFAULT" << ", " << "HIGH" << ", " << "HIGHEST")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse StableHLO_PrecisionAttr parameter 'value' which is to be a `::mlir::stablehlo::Precision`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return PrecisionAttr::get(odsParser.getContext(),
      ::mlir::stablehlo::Precision((*_result_value)));
}

void PrecisionAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyPrecision(getValue());
}

::mlir::stablehlo::Precision PrecisionAttr::getValue() const {
  return getImpl()->value;
}

} // namespace stablehlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::stablehlo::PrecisionAttr)
namespace mlir {
namespace stablehlo {
namespace detail {
struct RngAlgorithmAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::stablehlo::RngAlgorithm>;
  RngAlgorithmAttrStorage(::mlir::stablehlo::RngAlgorithm value) : value(value) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static RngAlgorithmAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto value = std::get<0>(tblgenKey);
    return new (allocator.allocate<RngAlgorithmAttrStorage>()) RngAlgorithmAttrStorage(value);
  }

  ::mlir::stablehlo::RngAlgorithm value;
};
} // namespace detail
RngAlgorithmAttr RngAlgorithmAttr::get(::mlir::MLIRContext *context, ::mlir::stablehlo::RngAlgorithm value) {
  return Base::get(context, value);
}

::mlir::Attribute RngAlgorithmAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::stablehlo::RngAlgorithm> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::stablehlo::RngAlgorithm> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::stablehlo::symbolizeRngAlgorithm(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::mlir::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::stablehlo::RngAlgorithm" << " to be one of: " << "DEFAULT" << ", " << "THREE_FRY" << ", " << "PHILOX")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse StableHLO_RngAlgorithmAttr parameter 'value' which is to be a `::mlir::stablehlo::RngAlgorithm`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return RngAlgorithmAttr::get(odsParser.getContext(),
      ::mlir::stablehlo::RngAlgorithm((*_result_value)));
}

void RngAlgorithmAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyRngAlgorithm(getValue());
}

::mlir::stablehlo::RngAlgorithm RngAlgorithmAttr::getValue() const {
  return getImpl()->value;
}

} // namespace stablehlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::stablehlo::RngAlgorithmAttr)
namespace mlir {
namespace stablehlo {
namespace detail {
struct RngDistributionAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::stablehlo::RngDistribution>;
  RngDistributionAttrStorage(::mlir::stablehlo::RngDistribution value) : value(value) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static RngDistributionAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto value = std::get<0>(tblgenKey);
    return new (allocator.allocate<RngDistributionAttrStorage>()) RngDistributionAttrStorage(value);
  }

  ::mlir::stablehlo::RngDistribution value;
};
} // namespace detail
RngDistributionAttr RngDistributionAttr::get(::mlir::MLIRContext *context, ::mlir::stablehlo::RngDistribution value) {
  return Base::get(context, value);
}

::mlir::Attribute RngDistributionAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::stablehlo::RngDistribution> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::stablehlo::RngDistribution> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::stablehlo::symbolizeRngDistribution(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::mlir::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::stablehlo::RngDistribution" << " to be one of: " << "UNIFORM" << ", " << "NORMAL")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse StableHLO_RngDistributionAttr parameter 'value' which is to be a `::mlir::stablehlo::RngDistribution`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return RngDistributionAttr::get(odsParser.getContext(),
      ::mlir::stablehlo::RngDistribution((*_result_value)));
}

void RngDistributionAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyRngDistribution(getValue());
}

::mlir::stablehlo::RngDistribution RngDistributionAttr::getValue() const {
  return getImpl()->value;
}

} // namespace stablehlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::stablehlo::RngDistributionAttr)
namespace mlir {
namespace stablehlo {
namespace detail {
struct ScatterDimensionNumbersAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::llvm::ArrayRef<int64_t>, ::llvm::ArrayRef<int64_t>, ::llvm::ArrayRef<int64_t>, int64_t>;
  ScatterDimensionNumbersAttrStorage(::llvm::ArrayRef<int64_t> updateWindowDims, ::llvm::ArrayRef<int64_t> insertedWindowDims, ::llvm::ArrayRef<int64_t> scatterDimsToOperandDims, int64_t indexVectorDim) : updateWindowDims(updateWindowDims), insertedWindowDims(insertedWindowDims), scatterDimsToOperandDims(scatterDimsToOperandDims), indexVectorDim(indexVectorDim) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (updateWindowDims == std::get<0>(tblgenKey)) && (insertedWindowDims == std::get<1>(tblgenKey)) && (scatterDimsToOperandDims == std::get<2>(tblgenKey)) && (indexVectorDim == std::get<3>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey), std::get<3>(tblgenKey));
  }

  static ScatterDimensionNumbersAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto updateWindowDims = std::get<0>(tblgenKey);
    auto insertedWindowDims = std::get<1>(tblgenKey);
    auto scatterDimsToOperandDims = std::get<2>(tblgenKey);
    auto indexVectorDim = std::get<3>(tblgenKey);
    updateWindowDims = allocator.copyInto(updateWindowDims);
    insertedWindowDims = allocator.copyInto(insertedWindowDims);
    scatterDimsToOperandDims = allocator.copyInto(scatterDimsToOperandDims);
    return new (allocator.allocate<ScatterDimensionNumbersAttrStorage>()) ScatterDimensionNumbersAttrStorage(updateWindowDims, insertedWindowDims, scatterDimsToOperandDims, indexVectorDim);
  }

  ::llvm::ArrayRef<int64_t> updateWindowDims;
  ::llvm::ArrayRef<int64_t> insertedWindowDims;
  ::llvm::ArrayRef<int64_t> scatterDimsToOperandDims;
  int64_t indexVectorDim;
};
} // namespace detail
ScatterDimensionNumbersAttr ScatterDimensionNumbersAttr::get(::mlir::MLIRContext *context, ::llvm::ArrayRef<int64_t> updateWindowDims, ::llvm::ArrayRef<int64_t> insertedWindowDims, ::llvm::ArrayRef<int64_t> scatterDimsToOperandDims, int64_t indexVectorDim) {
  return Base::get(context, updateWindowDims, insertedWindowDims, scatterDimsToOperandDims, indexVectorDim);
}

::llvm::ArrayRef<int64_t> ScatterDimensionNumbersAttr::getUpdateWindowDims() const {
  return getImpl()->updateWindowDims;
}

::llvm::ArrayRef<int64_t> ScatterDimensionNumbersAttr::getInsertedWindowDims() const {
  return getImpl()->insertedWindowDims;
}

::llvm::ArrayRef<int64_t> ScatterDimensionNumbersAttr::getScatterDimsToOperandDims() const {
  return getImpl()->scatterDimsToOperandDims;
}

int64_t ScatterDimensionNumbersAttr::getIndexVectorDim() const {
  return getImpl()->indexVectorDim;
}

} // namespace stablehlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::stablehlo::ScatterDimensionNumbersAttr)
namespace mlir {
namespace stablehlo {
namespace detail {
struct TransposeAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::stablehlo::Transpose>;
  TransposeAttrStorage(::mlir::stablehlo::Transpose value) : value(value) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static TransposeAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto value = std::get<0>(tblgenKey);
    return new (allocator.allocate<TransposeAttrStorage>()) TransposeAttrStorage(value);
  }

  ::mlir::stablehlo::Transpose value;
};
} // namespace detail
TransposeAttr TransposeAttr::get(::mlir::MLIRContext *context, ::mlir::stablehlo::Transpose value) {
  return Base::get(context, value);
}

::mlir::Attribute TransposeAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::stablehlo::Transpose> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::stablehlo::Transpose> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::stablehlo::symbolizeTranspose(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::mlir::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::stablehlo::Transpose" << " to be one of: " << "TRANSPOSE_INVALID" << ", " << "NO_TRANSPOSE" << ", " << "TRANSPOSE" << ", " << "ADJOINT")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse StableHLO_TransposeAttr parameter 'value' which is to be a `::mlir::stablehlo::Transpose`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return TransposeAttr::get(odsParser.getContext(),
      ::mlir::stablehlo::Transpose((*_result_value)));
}

void TransposeAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyTranspose(getValue());
}

::mlir::stablehlo::Transpose TransposeAttr::getValue() const {
  return getImpl()->value;
}

} // namespace stablehlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::stablehlo::TransposeAttr)
namespace mlir {
namespace stablehlo {
namespace detail {
struct TypeExtensionsAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::llvm::ArrayRef<int64_t>>;
  TypeExtensionsAttrStorage(::llvm::ArrayRef<int64_t> bounds) : bounds(bounds) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (bounds == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static TypeExtensionsAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto bounds = std::get<0>(tblgenKey);
    bounds = allocator.copyInto(bounds);
    return new (allocator.allocate<TypeExtensionsAttrStorage>()) TypeExtensionsAttrStorage(bounds);
  }

  ::llvm::ArrayRef<int64_t> bounds;
};
} // namespace detail
TypeExtensionsAttr TypeExtensionsAttr::get(::mlir::MLIRContext *context, ::llvm::ArrayRef<int64_t> bounds) {
  return Base::get(context, bounds);
}

::mlir::Attribute TypeExtensionsAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::llvm::SmallVector<int64_t>> _result_bounds;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse literal 'bounds'
  if (odsParser.parseKeyword("bounds")) return {};
  // Parse literal '='
  if (odsParser.parseEqual()) return {};
  // Parse literal '['
  if (odsParser.parseLSquare()) return {};

  // Parse variable 'bounds'
  _result_bounds = ::mlir::FieldParser<::llvm::SmallVector<int64_t>>::parse(odsParser);
  if (::mlir::failed(_result_bounds)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse StableHLO_TypeExtensions parameter 'bounds' which is to be a `::llvm::ArrayRef<int64_t>`");
    return {};
  }
  // Parse literal ']'
  if (odsParser.parseRSquare()) return {};
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_bounds));
  return TypeExtensionsAttr::get(odsParser.getContext(),
      ::llvm::ArrayRef<int64_t>((*_result_bounds)));
}

void TypeExtensionsAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter << "bounds";
  odsPrinter << ' ' << "=";
  odsPrinter << ' ' << "[";
  odsPrinter.printStrippedAttrOrType(getBounds());
  odsPrinter << "]";
  odsPrinter << ">";
}

::llvm::ArrayRef<int64_t> TypeExtensionsAttr::getBounds() const {
  return getImpl()->bounds;
}

} // namespace stablehlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::stablehlo::TypeExtensionsAttr)

#endif  // GET_ATTRDEF_CLASSES

