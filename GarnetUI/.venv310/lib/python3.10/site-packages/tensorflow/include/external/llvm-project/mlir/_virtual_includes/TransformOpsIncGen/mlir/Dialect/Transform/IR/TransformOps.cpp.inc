/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::transform::AlternativesOp,
::mlir::transform::CastOp,
::mlir::transform::ForeachOp,
::mlir::transform::GetClosestIsolatedParentOp,
::mlir::transform::GetProducerOfOperand,
::mlir::transform::MergeHandlesOp,
::mlir::transform::PDLMatchOp,
::mlir::transform::ReplicateOp,
::mlir::transform::SequenceOp,
::mlir::transform::SplitHandlesOp,
::mlir::transform::WithPDLPatternsOp,
::mlir::transform::YieldOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace transform {

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TransformOps0(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isa<::mlir::transform::TransformTypeInterface>()))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be TransformTypeInterface instance, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TransformOps0(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 64-bit signless integer attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TransformOps1(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::UnitAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: unit attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TransformOps2(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::SymbolRefAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: symbol reference attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TransformOps3(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::transform::FailurePropagationModeAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: Silenceable error propagation policy";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_region_constraint_TransformOps0(
    ::mlir::Operation *op, ::mlir::Region &region, ::llvm::StringRef regionName,
    unsigned regionIndex) {
  if (!((::llvm::hasNItems(region, 1)))) {
    return op->emitOpError("region #") << regionIndex
        << (regionName.empty() ? " " : " ('" + regionName + "') ")
        << "failed to verify constraint: region with 1 blocks";
  }
  return ::mlir::success();
}
} // namespace transform
} // namespace mlir
namespace mlir {
namespace transform {

//===----------------------------------------------------------------------===//
// ::mlir::transform::AlternativesOp definitions
//===----------------------------------------------------------------------===//

AlternativesOpAdaptor::AlternativesOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("transform.alternatives", odsAttrs.getContext());
}

AlternativesOpAdaptor::AlternativesOpAdaptor(AlternativesOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange AlternativesOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> AlternativesOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange AlternativesOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AlternativesOpAdaptor::getScope() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::DictionaryAttr AlternativesOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange AlternativesOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::RegionRange AlternativesOpAdaptor::getAlternatives() {
  return odsRegions.drop_front(0);
}

::mlir::LogicalResult AlternativesOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AlternativesOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range AlternativesOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::transform::TransformTypeInterface> AlternativesOp::getScope() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::MutableOperandRange AlternativesOp::getScopeMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> AlternativesOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::result_range AlternativesOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range AlternativesOp::getResults() {
  return getODSResults(0);
}

::mlir::MutableArrayRef<::mlir::Region> AlternativesOp::getAlternatives() {
  return (*this)->getRegions().drop_front(0);
}

void AlternativesOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, /*optional*/::mlir::Value scope, unsigned alternativesCount) {
  if (scope)
    odsState.addOperands(scope);
  for (unsigned i = 0; i < alternativesCount; ++i)
    (void)odsState.addRegion();
  odsState.addTypes(results);
}

void AlternativesOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes, unsigned numRegions) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != numRegions; ++i)
    (void)odsState.addRegion();
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult AlternativesOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    if (valueGroup0.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup0.size();
    }

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TransformOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TransformOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : getAlternatives())
      if (::mlir::failed(__mlir_ods_local_region_constraint_TransformOps0(*this, region, "alternatives", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult AlternativesOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult AlternativesOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> scopeOperands;
  ::llvm::SMLoc scopeOperandsLoc;
  (void)scopeOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> scopeTypes;
  ::llvm::SmallVector<::mlir::Type, 1> resultsTypes;
  ::llvm::SmallVector<std::unique_ptr<::mlir::Region>, 2> fullRegions;

  {
    scopeOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      scopeOperands.push_back(operand);
    }
  }
  if (!scopeOperands.empty()) {
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type optionalType;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalType(optionalType);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      scopeTypes.push_back(optionalType);
    }
  }
  }
  if (::mlir::succeeded(parser.parseOptionalArrow())) {

  if (parser.parseTypeList(resultsTypes))
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return ::mlir::failure();

  {
    std::unique_ptr<::mlir::Region> region;
    auto firstRegionResult = parser.parseOptionalRegion(region);
    if (firstRegionResult.has_value()) {
      if (failed(*firstRegionResult))
        return ::mlir::failure();
      fullRegions.emplace_back(std::move(region));

      // Parse any trailing regions.
      while (succeeded(parser.parseOptionalComma())) {
        region = std::make_unique<::mlir::Region>();
        if (parser.parseRegion(*region))
          return ::mlir::failure();
        fullRegions.emplace_back(std::move(region));
      }
    }
  }

  for (auto &region : fullRegions)
    ensureTerminator(*region, parser.getBuilder(), result.location);
  result.addRegions(fullRegions);
  result.addTypes(resultsTypes);
  if (parser.resolveOperands(scopeOperands, scopeTypes, scopeOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void AlternativesOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  if (getScope()) {
    _odsPrinter << ' ';
    if (::mlir::Value value = getScope())
      _odsPrinter << value;
    _odsPrinter << ' ' << ":";
    _odsPrinter << ' ';
    _odsPrinter << (getScope() ? ::llvm::ArrayRef<::mlir::Type>(getScope().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  }
  if (!getResults().empty()) {
    _odsPrinter << ' ' << "->";
    _odsPrinter << ' ';
    _odsPrinter << getResults().getTypes();
  }
  _odsPrinter.printOptionalAttrDictWithKeyword((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ';
    llvm::interleaveComma(getOperation()->getRegions(), _odsPrinter, [&](::mlir::Region &region) {

  {
    bool printTerminator = true;
    if (auto *term = region.empty() ? nullptr : region.begin()->getTerminator()) {
      printTerminator = !term->getAttrDictionary().empty() ||
                        term->getNumOperands() != 0 ||
                        term->getNumResults() != 0;
    }
    _odsPrinter.printRegion(region, /*printEntryBlockArgs=*/true,
      /*printBlockTerminators=*/printTerminator);
  }
    });
}

} // namespace transform
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::transform::AlternativesOp)

namespace mlir {
namespace transform {

//===----------------------------------------------------------------------===//
// ::mlir::transform::CastOp definitions
//===----------------------------------------------------------------------===//

CastOpAdaptor::CastOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("transform.cast", odsAttrs.getContext());
}

CastOpAdaptor::CastOpAdaptor(CastOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange CastOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CastOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CastOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CastOpAdaptor::getInput() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr CastOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult CastOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CastOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CastOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::transform::TransformTypeInterface> CastOp::getInput() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange CastOp::getInputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> CastOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CastOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::transform::TransformTypeInterface> CastOp::getOutput() {
  return *getODSResults(0).begin();
}

void CastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input) {
  odsState.addOperands(input);
  odsState.addTypes(output);
}

void CastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input) {
  odsState.addOperands(input);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CastOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CastOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TransformOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TransformOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult CastOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult CastOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand inputRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> inputOperands(inputRawOperands);  ::llvm::SMLoc inputOperandsLoc;
  (void)inputOperandsLoc;
  ::mlir::Type inputRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> inputTypes(inputRawTypes);
  ::mlir::Type outputRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> outputTypes(outputRawTypes);

  inputOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::transform::TransformTypeInterface type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    inputRawTypes[0] = type;
  }
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  {
    ::mlir::transform::TransformTypeInterface type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    outputRawTypes[0] = type;
  }
  result.addTypes(outputTypes);
  if (parser.resolveOperands(inputOperands, inputTypes, inputOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CastOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getInput();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getInput().getType();
    if (auto validType = type.dyn_cast<::mlir::transform::TransformTypeInterface>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  {
    auto type = getOutput().getType();
    if (auto validType = type.dyn_cast<::mlir::transform::TransformTypeInterface>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

} // namespace transform
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::transform::CastOp)

namespace mlir {
namespace transform {

//===----------------------------------------------------------------------===//
// ::mlir::transform::ForeachOp definitions
//===----------------------------------------------------------------------===//

ForeachOpAdaptor::ForeachOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("transform.foreach", odsAttrs.getContext());
}

ForeachOpAdaptor::ForeachOpAdaptor(ForeachOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange ForeachOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ForeachOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ForeachOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ForeachOpAdaptor::getTarget() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr ForeachOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange ForeachOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &ForeachOpAdaptor::getBody() {
  return *odsRegions[0];
}

::mlir::LogicalResult ForeachOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ForeachOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ForeachOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::transform::TransformTypeInterface> ForeachOp::getTarget() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange ForeachOp::getTargetMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ForeachOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::result_range ForeachOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range ForeachOp::getResults() {
  return getODSResults(0);
}

::mlir::Region &ForeachOp::getBody() {
  return (*this)->getRegion(0);
}

void ForeachOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::mlir::Value target) {
  odsState.addOperands(target);
  (void)odsState.addRegion();
  odsState.addTypes(results);
}

void ForeachOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ForeachOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TransformOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TransformOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_TransformOps0(*this, region, "body", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult ForeachOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult ForeachOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand targetRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> targetOperands(targetRawOperands);  ::llvm::SMLoc targetOperandsLoc;
  (void)targetOperandsLoc;
  ::mlir::Type targetRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> targetTypes(targetRawTypes);
  ::llvm::SmallVector<::mlir::Type, 1> resultsTypes;
  std::unique_ptr<::mlir::Region> bodyRegion = std::make_unique<::mlir::Region>();

  targetOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(targetRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::transform::TransformTypeInterface type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    targetRawTypes[0] = type;
  }
  if (::mlir::succeeded(parser.parseOptionalArrow())) {

  if (parser.parseTypeList(resultsTypes))
    return ::mlir::failure();
  }

  if (parser.parseRegion(*bodyRegion))
    return ::mlir::failure();

  ensureTerminator(*bodyRegion, parser.getBuilder(), result.location);
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addRegion(std::move(bodyRegion));
  result.addTypes(resultsTypes);
  if (parser.resolveOperands(targetOperands, targetTypes, targetOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ForeachOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getTarget();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getTarget().getType();
    if (auto validType = type.dyn_cast<::mlir::transform::TransformTypeInterface>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  if (!getResults().empty()) {
    _odsPrinter << ' ' << "->";
    _odsPrinter << ' ';
    _odsPrinter << getResults().getTypes();
  }
  _odsPrinter << ' ';

  {
    bool printTerminator = true;
    if (auto *term = getBody().empty() ? nullptr : getBody().begin()->getTerminator()) {
      printTerminator = !term->getAttrDictionary().empty() ||
                        term->getNumOperands() != 0 ||
                        term->getNumResults() != 0;
    }
    _odsPrinter.printRegion(getBody(), /*printEntryBlockArgs=*/true,
      /*printBlockTerminators=*/printTerminator);
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

} // namespace transform
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::transform::ForeachOp)

namespace mlir {
namespace transform {

//===----------------------------------------------------------------------===//
// ::mlir::transform::GetClosestIsolatedParentOp definitions
//===----------------------------------------------------------------------===//

GetClosestIsolatedParentOpAdaptor::GetClosestIsolatedParentOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("transform.get_closest_isolated_parent", odsAttrs.getContext());
}

GetClosestIsolatedParentOpAdaptor::GetClosestIsolatedParentOpAdaptor(GetClosestIsolatedParentOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange GetClosestIsolatedParentOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> GetClosestIsolatedParentOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange GetClosestIsolatedParentOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GetClosestIsolatedParentOpAdaptor::getTarget() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr GetClosestIsolatedParentOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult GetClosestIsolatedParentOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> GetClosestIsolatedParentOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range GetClosestIsolatedParentOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::transform::TransformTypeInterface> GetClosestIsolatedParentOp::getTarget() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange GetClosestIsolatedParentOp::getTargetMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> GetClosestIsolatedParentOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GetClosestIsolatedParentOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::transform::TransformTypeInterface> GetClosestIsolatedParentOp::getParent() {
  return *getODSResults(0).begin();
}

void GetClosestIsolatedParentOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type parent, ::mlir::Value target) {
  odsState.addOperands(target);
  odsState.addTypes(parent);
}

void GetClosestIsolatedParentOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target) {
  odsState.addOperands(target);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GetClosestIsolatedParentOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult GetClosestIsolatedParentOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TransformOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TransformOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult GetClosestIsolatedParentOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult GetClosestIsolatedParentOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand targetRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> targetOperands(targetRawOperands);  ::llvm::SMLoc targetOperandsLoc;
  (void)targetOperandsLoc;
  ::llvm::ArrayRef<::mlir::Type> allOperandTypes;
  ::llvm::ArrayRef<::mlir::Type> allResultTypes;

  targetOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(targetRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  ::mlir::FunctionType allOperand__allResult_functionType;
  if (parser.parseType(allOperand__allResult_functionType))
    return ::mlir::failure();
  allOperandTypes = allOperand__allResult_functionType.getInputs();
  allResultTypes = allOperand__allResult_functionType.getResults();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(targetOperands, allOperandTypes, parser.getNameLoc(), result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void GetClosestIsolatedParentOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getTarget();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter.printFunctionalType(getOperation()->getOperandTypes(), getOperation()->getResultTypes());
}

} // namespace transform
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::transform::GetClosestIsolatedParentOp)

namespace mlir {
namespace transform {

//===----------------------------------------------------------------------===//
// ::mlir::transform::GetProducerOfOperand definitions
//===----------------------------------------------------------------------===//

GetProducerOfOperandAdaptor::GetProducerOfOperandAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("transform.get_producer_of_operand", odsAttrs.getContext());
}

GetProducerOfOperandAdaptor::GetProducerOfOperandAdaptor(GetProducerOfOperand op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange GetProducerOfOperandAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> GetProducerOfOperandAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange GetProducerOfOperandAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GetProducerOfOperandAdaptor::getTarget() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr GetProducerOfOperandAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr GetProducerOfOperandAdaptor::getOperandNumberAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, GetProducerOfOperand::getOperandNumberAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
  return attr;
}

uint64_t GetProducerOfOperandAdaptor::getOperandNumber() {
  auto attr = getOperandNumberAttr();
  return attr.getValue().getZExtValue();
}

::mlir::LogicalResult GetProducerOfOperandAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_number;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'transform.get_producer_of_operand' op ""requires attribute 'operand_number'");
    if (namedAttrIt->getName() == GetProducerOfOperand::getOperandNumberAttrName(*odsOpName)) {
      tblgen_operand_number = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_operand_number && !(((tblgen_operand_number.isa<::mlir::IntegerAttr>())) && ((tblgen_operand_number.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
    return emitError(loc, "'transform.get_producer_of_operand' op ""attribute 'operand_number' failed to satisfy constraint: 64-bit signless integer attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> GetProducerOfOperand::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range GetProducerOfOperand::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::transform::TransformTypeInterface> GetProducerOfOperand::getTarget() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange GetProducerOfOperand::getTargetMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> GetProducerOfOperand::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GetProducerOfOperand::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::transform::TransformTypeInterface> GetProducerOfOperand::getParent() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr GetProducerOfOperand::getOperandNumberAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandNumberAttrName()).cast<::mlir::IntegerAttr>();
}

uint64_t GetProducerOfOperand::getOperandNumber() {
  auto attr = getOperandNumberAttr();
  return attr.getValue().getZExtValue();
}

void GetProducerOfOperand::setOperandNumberAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getOperandNumberAttrName(), attr);
}

void GetProducerOfOperand::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type parent, ::mlir::Value target, ::mlir::IntegerAttr operand_number) {
  odsState.addOperands(target);
  odsState.addAttribute(getOperandNumberAttrName(odsState.name), operand_number);
  odsState.addTypes(parent);
}

void GetProducerOfOperand::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target, ::mlir::IntegerAttr operand_number) {
  odsState.addOperands(target);
  odsState.addAttribute(getOperandNumberAttrName(odsState.name), operand_number);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GetProducerOfOperand::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type parent, ::mlir::Value target, uint64_t operand_number) {
  odsState.addOperands(target);
  odsState.addAttribute(getOperandNumberAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), operand_number));
  odsState.addTypes(parent);
}

void GetProducerOfOperand::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target, uint64_t operand_number) {
  odsState.addOperands(target);
  odsState.addAttribute(getOperandNumberAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), operand_number));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GetProducerOfOperand::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult GetProducerOfOperand::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_number;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_number'");
    if (namedAttrIt->getName() == getOperandNumberAttrName()) {
      tblgen_operand_number = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TransformOps0(*this, tblgen_operand_number, "operand_number")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TransformOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TransformOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult GetProducerOfOperand::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult GetProducerOfOperand::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand targetRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> targetOperands(targetRawOperands);  ::llvm::SMLoc targetOperandsLoc;
  (void)targetOperandsLoc;
  ::mlir::IntegerAttr operand_numberAttr;
  ::llvm::ArrayRef<::mlir::Type> allOperandTypes;
  ::llvm::ArrayRef<::mlir::Type> allResultTypes;

  targetOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(targetRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseLSquare())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(operand_numberAttr, parser.getBuilder().getIntegerType(64), "operand_number",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseRSquare())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  ::mlir::FunctionType allOperand__allResult_functionType;
  if (parser.parseType(allOperand__allResult_functionType))
    return ::mlir::failure();
  allOperandTypes = allOperand__allResult_functionType.getInputs();
  allResultTypes = allOperand__allResult_functionType.getResults();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(targetOperands, allOperandTypes, parser.getNameLoc(), result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void GetProducerOfOperand::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getTarget();
  _odsPrinter << "[";
  _odsPrinter.printAttributeWithoutType(getOperandNumberAttr());
  _odsPrinter << "]";
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"operand_number"});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter.printFunctionalType(getOperation()->getOperandTypes(), getOperation()->getResultTypes());
}

} // namespace transform
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::transform::GetProducerOfOperand)

namespace mlir {
namespace transform {

//===----------------------------------------------------------------------===//
// ::mlir::transform::MergeHandlesOp definitions
//===----------------------------------------------------------------------===//

MergeHandlesOpAdaptor::MergeHandlesOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("transform.merge_handles", odsAttrs.getContext());
}

MergeHandlesOpAdaptor::MergeHandlesOpAdaptor(MergeHandlesOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange MergeHandlesOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MergeHandlesOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange MergeHandlesOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange MergeHandlesOpAdaptor::getHandles() {
  return getODSOperands(0);
}

::mlir::DictionaryAttr MergeHandlesOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::UnitAttr MergeHandlesOpAdaptor::getDeduplicateAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, MergeHandlesOp::getDeduplicateAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool MergeHandlesOpAdaptor::getDeduplicate() {
  auto attr = getDeduplicateAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::LogicalResult MergeHandlesOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_deduplicate;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == MergeHandlesOp::getDeduplicateAttrName(*odsOpName)) {
      tblgen_deduplicate = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_deduplicate && !((tblgen_deduplicate.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'transform.merge_handles' op ""attribute 'deduplicate' failed to satisfy constraint: unit attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> MergeHandlesOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range MergeHandlesOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range MergeHandlesOp::getHandles() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange MergeHandlesOp::getHandlesMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> MergeHandlesOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MergeHandlesOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::transform::TransformTypeInterface> MergeHandlesOp::getResult() {
  return *getODSResults(0).begin();
}

::mlir::UnitAttr MergeHandlesOp::getDeduplicateAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getDeduplicateAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool MergeHandlesOp::getDeduplicate() {
  auto attr = getDeduplicateAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

void MergeHandlesOp::setDeduplicateAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getDeduplicateAttrName(), attr);
}

::mlir::Attribute MergeHandlesOp::removeDeduplicateAttr() {
  return (*this)->removeAttr(getDeduplicateAttrName());
}

void MergeHandlesOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::ValueRange handles, /*optional*/::mlir::UnitAttr deduplicate) {
  odsState.addOperands(handles);
  if (deduplicate) {
    odsState.addAttribute(getDeduplicateAttrName(odsState.name), deduplicate);
  }
  odsState.addTypes(result);
}

void MergeHandlesOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange handles, /*optional*/::mlir::UnitAttr deduplicate) {
  odsState.addOperands(handles);
  if (deduplicate) {
    odsState.addAttribute(getDeduplicateAttrName(odsState.name), deduplicate);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MergeHandlesOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::ValueRange handles, /*optional*/bool deduplicate) {
  odsState.addOperands(handles);
  if (deduplicate) {
    odsState.addAttribute(getDeduplicateAttrName(odsState.name), ((deduplicate) ? odsBuilder.getUnitAttr() : nullptr));
  }
  odsState.addTypes(result);
}

void MergeHandlesOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange handles, /*optional*/bool deduplicate) {
  odsState.addOperands(handles);
  if (deduplicate) {
    odsState.addAttribute(getDeduplicateAttrName(odsState.name), ((deduplicate) ? odsBuilder.getUnitAttr() : nullptr));
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MergeHandlesOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void MergeHandlesOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange handles, /*optional*/::mlir::UnitAttr deduplicate) {
  odsState.addOperands(handles);
  if (deduplicate) {
    odsState.addAttribute(getDeduplicateAttrName(odsState.name), deduplicate);
  }
  odsState.addTypes({handles.front().getType()});

}

void MergeHandlesOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange handles, /*optional*/bool deduplicate) {
  odsState.addOperands(handles);
  if (deduplicate) {
    odsState.addAttribute(getDeduplicateAttrName(odsState.name), ((deduplicate) ? odsBuilder.getUnitAttr() : nullptr));
  }
  odsState.addTypes({handles.front().getType()});

}

void MergeHandlesOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::LogicalResult MergeHandlesOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_deduplicate;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getDeduplicateAttrName()) {
      tblgen_deduplicate = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TransformOps1(*this, tblgen_deduplicate, "deduplicate")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TransformOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TransformOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult MergeHandlesOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult MergeHandlesOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::UnitAttr deduplicateAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> handlesOperands;
  ::llvm::SMLoc handlesOperandsLoc;
  (void)handlesOperandsLoc;
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);

  {
    ::mlir::OptionalParseResult parseResult =
      parser.parseOptionalAttribute(deduplicateAttr, parser.getBuilder().getType<::mlir::NoneType>(), "deduplicate", result.attributes);
    if (parseResult.has_value() && failed(*parseResult))
      return ::mlir::failure();
  }
  if (deduplicateAttr) {
  }

  handlesOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(handlesOperands))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::transform::TransformTypeInterface type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawTypes[0] = type;
  }
  result.addTypes(resultTypes);
  if (parser.resolveOperands(handlesOperands, resultTypes[0], handlesOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void MergeHandlesOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  if ((*this)->getAttr("deduplicate")) {
    _odsPrinter << ' ';
    _odsPrinter.printAttributeWithoutType(getDeduplicateAttr());
  }
  _odsPrinter << ' ';
  _odsPrinter << getHandles();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"deduplicate"});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = type.dyn_cast<::mlir::transform::TransformTypeInterface>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

} // namespace transform
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::transform::MergeHandlesOp)

namespace mlir {
namespace transform {

//===----------------------------------------------------------------------===//
// ::mlir::transform::PDLMatchOp definitions
//===----------------------------------------------------------------------===//

PDLMatchOpAdaptor::PDLMatchOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("transform.pdl_match", odsAttrs.getContext());
}

PDLMatchOpAdaptor::PDLMatchOpAdaptor(PDLMatchOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange PDLMatchOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> PDLMatchOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange PDLMatchOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value PDLMatchOpAdaptor::getRoot() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr PDLMatchOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::SymbolRefAttr PDLMatchOpAdaptor::getPatternNameAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, PDLMatchOp::getPatternNameAttrName(*odsOpName)).cast<::mlir::SymbolRefAttr>();
  return attr;
}

::mlir::SymbolRefAttr PDLMatchOpAdaptor::getPatternName() {
  auto attr = getPatternNameAttr();
  return attr;
}

::mlir::LogicalResult PDLMatchOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_pattern_name;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'transform.pdl_match' op ""requires attribute 'pattern_name'");
    if (namedAttrIt->getName() == PDLMatchOp::getPatternNameAttrName(*odsOpName)) {
      tblgen_pattern_name = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_pattern_name && !((tblgen_pattern_name.isa<::mlir::SymbolRefAttr>())))
    return emitError(loc, "'transform.pdl_match' op ""attribute 'pattern_name' failed to satisfy constraint: symbol reference attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> PDLMatchOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range PDLMatchOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::transform::TransformTypeInterface> PDLMatchOp::getRoot() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange PDLMatchOp::getRootMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> PDLMatchOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range PDLMatchOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::transform::TransformTypeInterface> PDLMatchOp::getMatched() {
  return *getODSResults(0).begin();
}

::mlir::SymbolRefAttr PDLMatchOp::getPatternNameAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getPatternNameAttrName()).cast<::mlir::SymbolRefAttr>();
}

::mlir::SymbolRefAttr PDLMatchOp::getPatternName() {
  auto attr = getPatternNameAttr();
  return attr;
}

void PDLMatchOp::setPatternNameAttr(::mlir::SymbolRefAttr attr) {
  (*this)->setAttr(getPatternNameAttrName(), attr);
}

void PDLMatchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type matched, ::mlir::Value root, ::mlir::SymbolRefAttr pattern_name) {
  odsState.addOperands(root);
  odsState.addAttribute(getPatternNameAttrName(odsState.name), pattern_name);
  odsState.addTypes(matched);
}

void PDLMatchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value root, ::mlir::SymbolRefAttr pattern_name) {
  odsState.addOperands(root);
  odsState.addAttribute(getPatternNameAttrName(odsState.name), pattern_name);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void PDLMatchOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult PDLMatchOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_pattern_name;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'pattern_name'");
    if (namedAttrIt->getName() == getPatternNameAttrName()) {
      tblgen_pattern_name = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TransformOps2(*this, tblgen_pattern_name, "pattern_name")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TransformOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TransformOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult PDLMatchOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult PDLMatchOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SymbolRefAttr pattern_nameAttr;
  ::mlir::OpAsmParser::UnresolvedOperand rootRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> rootOperands(rootRawOperands);  ::llvm::SMLoc rootOperandsLoc;
  (void)rootOperandsLoc;
  ::llvm::ArrayRef<::mlir::Type> allOperandTypes;
  ::llvm::ArrayRef<::mlir::Type> allResultTypes;

  if (parser.parseCustomAttributeWithFallback(pattern_nameAttr, parser.getBuilder().getType<::mlir::NoneType>(), "pattern_name",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseKeyword("in"))
    return ::mlir::failure();

  rootOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rootRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  ::mlir::FunctionType allOperand__allResult_functionType;
  if (parser.parseType(allOperand__allResult_functionType))
    return ::mlir::failure();
  allOperandTypes = allOperand__allResult_functionType.getInputs();
  allResultTypes = allOperand__allResult_functionType.getResults();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(rootOperands, allOperandTypes, parser.getNameLoc(), result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void PDLMatchOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getPatternNameAttr());
  _odsPrinter << ' ' << "in";
  _odsPrinter << ' ';
  _odsPrinter << getRoot();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"pattern_name"});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter.printFunctionalType(getOperation()->getOperandTypes(), getOperation()->getResultTypes());
}

} // namespace transform
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::transform::PDLMatchOp)

namespace mlir {
namespace transform {

//===----------------------------------------------------------------------===//
// ::mlir::transform::ReplicateOp definitions
//===----------------------------------------------------------------------===//

ReplicateOpAdaptor::ReplicateOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("transform.replicate", odsAttrs.getContext());
}

ReplicateOpAdaptor::ReplicateOpAdaptor(ReplicateOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange ReplicateOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ReplicateOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange ReplicateOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReplicateOpAdaptor::getPattern() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange ReplicateOpAdaptor::getHandles() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr ReplicateOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ReplicateOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ReplicateOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range ReplicateOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::transform::TransformTypeInterface> ReplicateOp::getPattern() {
  return *getODSOperands(0).begin();
}

::mlir::Operation::operand_range ReplicateOp::getHandles() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange ReplicateOp::getPatternMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ReplicateOp::getHandlesMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ReplicateOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::result_range ReplicateOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range ReplicateOp::getReplicated() {
  return getODSResults(0);
}

void ReplicateOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange replicated, ::mlir::Value pattern, ::mlir::ValueRange handles) {
  odsState.addOperands(pattern);
  odsState.addOperands(handles);
  odsState.addTypes(replicated);
}

void ReplicateOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ReplicateOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TransformOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TransformOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TransformOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  if (!(((this->getODSOperands(1).getType()) == (this->getODSResults(0).getType()) && (this->getODSResults(0).getType()) == (this->getODSOperands(1).getType()))))
    return emitOpError("failed to verify that all of {handles, replicated} have same type");
  return ::mlir::success();
}

::mlir::LogicalResult ReplicateOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult ReplicateOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand patternRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> patternOperands(patternRawOperands);  ::llvm::SMLoc patternOperandsLoc;
  (void)patternOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> handlesOperands;
  ::llvm::SMLoc handlesOperandsLoc;
  (void)handlesOperandsLoc;
  ::mlir::Type patternRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> patternTypes(patternRawTypes);
  ::llvm::SmallVector<::mlir::Type, 1> handlesTypes;
  if (parser.parseKeyword("num"))
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  patternOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(patternRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();

  handlesOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(handlesOperands))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::transform::TransformTypeInterface type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    patternRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseTypeList(handlesTypes))
    return ::mlir::failure();
  result.addTypes(handlesTypes);
  if (parser.resolveOperands(patternOperands, patternTypes, patternOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(handlesOperands, handlesTypes, handlesOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ReplicateOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << "num";
  _odsPrinter << "(";
  _odsPrinter << getPattern();
  _odsPrinter << ")";
  _odsPrinter << ' ';
  _odsPrinter << getHandles();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getPattern().getType();
    if (auto validType = type.dyn_cast<::mlir::transform::TransformTypeInterface>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getHandles().getTypes();
}

} // namespace transform
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::transform::ReplicateOp)

namespace mlir {
namespace transform {

//===----------------------------------------------------------------------===//
// ::mlir::transform::SequenceOp definitions
//===----------------------------------------------------------------------===//

SequenceOpAdaptor::SequenceOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("transform.sequence", odsAttrs.getContext());
}

SequenceOpAdaptor::SequenceOpAdaptor(SequenceOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange SequenceOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> SequenceOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange SequenceOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SequenceOpAdaptor::getRoot() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::DictionaryAttr SequenceOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::transform::FailurePropagationModeAttr SequenceOpAdaptor::getFailurePropagationModeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, SequenceOp::getFailurePropagationModeAttrName(*odsOpName)).cast<::mlir::transform::FailurePropagationModeAttr>();
  return attr;
}

::mlir::transform::FailurePropagationMode SequenceOpAdaptor::getFailurePropagationMode() {
  auto attr = getFailurePropagationModeAttr();
  return attr.getValue();
}

::mlir::RegionRange SequenceOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &SequenceOpAdaptor::getBody() {
  return *odsRegions[0];
}

::mlir::LogicalResult SequenceOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_failure_propagation_mode;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'transform.sequence' op ""requires attribute 'failure_propagation_mode'");
    if (namedAttrIt->getName() == SequenceOp::getFailurePropagationModeAttrName(*odsOpName)) {
      tblgen_failure_propagation_mode = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_failure_propagation_mode && !((tblgen_failure_propagation_mode.isa<::mlir::transform::FailurePropagationModeAttr>())))
    return emitError(loc, "'transform.sequence' op ""attribute 'failure_propagation_mode' failed to satisfy constraint: Silenceable error propagation policy");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> SequenceOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range SequenceOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::transform::TransformTypeInterface> SequenceOp::getRoot() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::MutableOperandRange SequenceOp::getRootMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> SequenceOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::result_range SequenceOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range SequenceOp::getResults() {
  return getODSResults(0);
}

::mlir::Region &SequenceOp::getBody() {
  return (*this)->getRegion(0);
}

::mlir::transform::FailurePropagationModeAttr SequenceOp::getFailurePropagationModeAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getFailurePropagationModeAttrName()).cast<::mlir::transform::FailurePropagationModeAttr>();
}

::mlir::transform::FailurePropagationMode SequenceOp::getFailurePropagationMode() {
  auto attr = getFailurePropagationModeAttr();
  return attr.getValue();
}

void SequenceOp::setFailurePropagationModeAttr(::mlir::transform::FailurePropagationModeAttr attr) {
  (*this)->setAttr(getFailurePropagationModeAttrName(), attr);
}

void SequenceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::mlir::transform::FailurePropagationModeAttr failure_propagation_mode, /*optional*/::mlir::Value root) {
  if (root)
    odsState.addOperands(root);
  odsState.addAttribute(getFailurePropagationModeAttrName(odsState.name), failure_propagation_mode);
  (void)odsState.addRegion();
  odsState.addTypes(results);
}

void SequenceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::mlir::transform::FailurePropagationMode failure_propagation_mode, /*optional*/::mlir::Value root) {
  if (root)
    odsState.addOperands(root);
  odsState.addAttribute(getFailurePropagationModeAttrName(odsState.name), ::mlir::transform::FailurePropagationModeAttr::get(odsBuilder.getContext(), failure_propagation_mode));
  (void)odsState.addRegion();
  odsState.addTypes(results);
}

void SequenceOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult SequenceOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_failure_propagation_mode;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'failure_propagation_mode'");
    if (namedAttrIt->getName() == getFailurePropagationModeAttrName()) {
      tblgen_failure_propagation_mode = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TransformOps3(*this, tblgen_failure_propagation_mode, "failure_propagation_mode")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    if (valueGroup0.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup0.size();
    }

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TransformOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TransformOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_TransformOps0(*this, region, "body", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult SequenceOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult SequenceOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> rootOperands;
  ::llvm::SMLoc rootOperandsLoc;
  (void)rootOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> rootTypes;
  ::llvm::SmallVector<::mlir::Type, 1> resultsTypes;
  ::mlir::transform::FailurePropagationModeAttr failure_propagation_modeAttr;
  ::llvm::SmallVector<std::unique_ptr<::mlir::Region>, 2> fullRegions;

  {
    rootOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      rootOperands.push_back(operand);
    }
  }
  if (!rootOperands.empty()) {
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type optionalType;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalType(optionalType);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      rootTypes.push_back(optionalType);
    }
  }
  }
  if (::mlir::succeeded(parser.parseOptionalArrow())) {

  if (parser.parseTypeList(resultsTypes))
    return ::mlir::failure();
  }
  if (parser.parseKeyword("failures"))
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ::llvm::StringRef attrStr;
    ::mlir::NamedAttrList attrStorage;
    auto loc = parser.getCurrentLocation();
    if (parser.parseOptionalKeyword(&attrStr, {"propagate","suppress"})) {
      ::mlir::StringAttr attrVal;
      ::mlir::OptionalParseResult parseResult =
        parser.parseOptionalAttribute(attrVal,
                                      parser.getBuilder().getNoneType(),
                                      "failure_propagation_mode", attrStorage);
      if (parseResult.has_value()) {
        if (failed(*parseResult))
          return ::mlir::failure();
        attrStr = attrVal.getValue();
      } else {
        return parser.emitError(loc, "expected string or keyword containing one of the following enum values for attribute 'failure_propagation_mode' [propagate, suppress]");
      }
    }
    if (!attrStr.empty()) {
      auto attrOptional = ::mlir::transform::symbolizeFailurePropagationMode(attrStr);
      if (!attrOptional)
        return parser.emitError(loc, "invalid ")
               << "failure_propagation_mode attribute specification: \"" << attrStr << '"';;

      failure_propagation_modeAttr = ::mlir::transform::FailurePropagationModeAttr::get(parser.getBuilder().getContext(), attrOptional.value());
      result.addAttribute("failure_propagation_mode", failure_propagation_modeAttr);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return ::mlir::failure();

  {
    std::unique_ptr<::mlir::Region> region;
    auto firstRegionResult = parser.parseOptionalRegion(region);
    if (firstRegionResult.has_value()) {
      if (failed(*firstRegionResult))
        return ::mlir::failure();
      fullRegions.emplace_back(std::move(region));

      // Parse any trailing regions.
      while (succeeded(parser.parseOptionalComma())) {
        region = std::make_unique<::mlir::Region>();
        if (parser.parseRegion(*region))
          return ::mlir::failure();
        fullRegions.emplace_back(std::move(region));
      }
    }
  }

  for (auto &region : fullRegions)
    ensureTerminator(*region, parser.getBuilder(), result.location);
  result.addRegions(fullRegions);
  result.addTypes(resultsTypes);
  if (parser.resolveOperands(rootOperands, rootTypes, rootOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void SequenceOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  if (getRoot()) {
    _odsPrinter << ' ';
    if (::mlir::Value value = getRoot())
      _odsPrinter << value;
    _odsPrinter << ' ' << ":";
    _odsPrinter << ' ';
    _odsPrinter << (getRoot() ? ::llvm::ArrayRef<::mlir::Type>(getRoot().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  }
  if (!getResults().empty()) {
    _odsPrinter << ' ' << "->";
    _odsPrinter << ' ';
    _odsPrinter << getResults().getTypes();
  }
  _odsPrinter << ' ' << "failures";
  _odsPrinter << "(";

  {
    auto caseValue = getFailurePropagationMode();
    auto caseValueStr = stringifyFailurePropagationMode(caseValue);
    _odsPrinter << caseValueStr;
  }
  _odsPrinter << ")";
  _odsPrinter.printOptionalAttrDictWithKeyword((*this)->getAttrs(), /*elidedAttrs=*/{"failure_propagation_mode"});
  _odsPrinter << ' ';
    llvm::interleaveComma(getOperation()->getRegions(), _odsPrinter, [&](::mlir::Region &region) {

  {
    bool printTerminator = true;
    if (auto *term = region.empty() ? nullptr : region.begin()->getTerminator()) {
      printTerminator = !term->getAttrDictionary().empty() ||
                        term->getNumOperands() != 0 ||
                        term->getNumResults() != 0;
    }
    _odsPrinter.printRegion(region, /*printEntryBlockArgs=*/true,
      /*printBlockTerminators=*/printTerminator);
  }
    });
}

} // namespace transform
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::transform::SequenceOp)

namespace mlir {
namespace transform {

//===----------------------------------------------------------------------===//
// ::mlir::transform::SplitHandlesOp definitions
//===----------------------------------------------------------------------===//

SplitHandlesOpAdaptor::SplitHandlesOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("transform.split_handles", odsAttrs.getContext());
}

SplitHandlesOpAdaptor::SplitHandlesOpAdaptor(SplitHandlesOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange SplitHandlesOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> SplitHandlesOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SplitHandlesOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SplitHandlesOpAdaptor::getHandle() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr SplitHandlesOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr SplitHandlesOpAdaptor::getNumResultHandlesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, SplitHandlesOp::getNumResultHandlesAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
  return attr;
}

uint64_t SplitHandlesOpAdaptor::getNumResultHandles() {
  auto attr = getNumResultHandlesAttr();
  return attr.getValue().getZExtValue();
}

::mlir::LogicalResult SplitHandlesOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_num_result_handles;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'transform.split_handles' op ""requires attribute 'num_result_handles'");
    if (namedAttrIt->getName() == SplitHandlesOp::getNumResultHandlesAttrName(*odsOpName)) {
      tblgen_num_result_handles = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_num_result_handles && !(((tblgen_num_result_handles.isa<::mlir::IntegerAttr>())) && ((tblgen_num_result_handles.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
    return emitError(loc, "'transform.split_handles' op ""attribute 'num_result_handles' failed to satisfy constraint: 64-bit signless integer attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> SplitHandlesOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SplitHandlesOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::transform::TransformTypeInterface> SplitHandlesOp::getHandle() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange SplitHandlesOp::getHandleMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> SplitHandlesOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::result_range SplitHandlesOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range SplitHandlesOp::getResults() {
  return getODSResults(0);
}

::mlir::IntegerAttr SplitHandlesOp::getNumResultHandlesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getNumResultHandlesAttrName()).cast<::mlir::IntegerAttr>();
}

uint64_t SplitHandlesOp::getNumResultHandles() {
  auto attr = getNumResultHandlesAttr();
  return attr.getValue().getZExtValue();
}

void SplitHandlesOp::setNumResultHandlesAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getNumResultHandlesAttrName(), attr);
}

void SplitHandlesOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::mlir::Value handle, ::mlir::IntegerAttr num_result_handles) {
  odsState.addOperands(handle);
  odsState.addAttribute(getNumResultHandlesAttrName(odsState.name), num_result_handles);
  odsState.addTypes(results);
}

void SplitHandlesOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::mlir::Value handle, uint64_t num_result_handles) {
  odsState.addOperands(handle);
  odsState.addAttribute(getNumResultHandlesAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), num_result_handles));
  odsState.addTypes(results);
}

void SplitHandlesOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult SplitHandlesOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_num_result_handles;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'num_result_handles'");
    if (namedAttrIt->getName() == getNumResultHandlesAttrName()) {
      tblgen_num_result_handles = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TransformOps0(*this, tblgen_num_result_handles, "num_result_handles")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TransformOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TransformOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult SplitHandlesOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult SplitHandlesOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand handleRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> handleOperands(handleRawOperands);  ::llvm::SMLoc handleOperandsLoc;
  (void)handleOperandsLoc;
  ::mlir::IntegerAttr num_result_handlesAttr;
  ::llvm::ArrayRef<::mlir::Type> allOperandTypes;
  ::llvm::ArrayRef<::mlir::Type> allResultTypes;

  handleOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(handleRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("in"))
    return ::mlir::failure();
  if (parser.parseLSquare())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(num_result_handlesAttr, parser.getBuilder().getIntegerType(64), "num_result_handles",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseRSquare())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  ::mlir::FunctionType allOperand__allResult_functionType;
  if (parser.parseType(allOperand__allResult_functionType))
    return ::mlir::failure();
  allOperandTypes = allOperand__allResult_functionType.getInputs();
  allResultTypes = allOperand__allResult_functionType.getResults();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(handleOperands, allOperandTypes, parser.getNameLoc(), result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void SplitHandlesOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getHandle();
  _odsPrinter << ' ' << "in";
  _odsPrinter << "[";
  _odsPrinter.printAttributeWithoutType(getNumResultHandlesAttr());
  _odsPrinter << "]";
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"num_result_handles"});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter.printFunctionalType(getOperation()->getOperandTypes(), getOperation()->getResultTypes());
}

} // namespace transform
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::transform::SplitHandlesOp)

namespace mlir {
namespace transform {

//===----------------------------------------------------------------------===//
// ::mlir::transform::WithPDLPatternsOp definitions
//===----------------------------------------------------------------------===//

WithPDLPatternsOpAdaptor::WithPDLPatternsOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("transform.with_pdl_patterns", odsAttrs.getContext());
}

WithPDLPatternsOpAdaptor::WithPDLPatternsOpAdaptor(WithPDLPatternsOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange WithPDLPatternsOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> WithPDLPatternsOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange WithPDLPatternsOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value WithPDLPatternsOpAdaptor::getRoot() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::DictionaryAttr WithPDLPatternsOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange WithPDLPatternsOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &WithPDLPatternsOpAdaptor::getBody() {
  return *odsRegions[0];
}

::mlir::LogicalResult WithPDLPatternsOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> WithPDLPatternsOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range WithPDLPatternsOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::transform::TransformTypeInterface> WithPDLPatternsOp::getRoot() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::MutableOperandRange WithPDLPatternsOp::getRootMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> WithPDLPatternsOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range WithPDLPatternsOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &WithPDLPatternsOp::getBody() {
  return (*this)->getRegion(0);
}

void WithPDLPatternsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value root) {
  if (root)
    odsState.addOperands(root);
  (void)odsState.addRegion();
}

void WithPDLPatternsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value root) {
  if (root)
    odsState.addOperands(root);
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void WithPDLPatternsOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult WithPDLPatternsOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    if (valueGroup0.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup0.size();
    }

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TransformOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_TransformOps0(*this, region, "body", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult WithPDLPatternsOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult WithPDLPatternsOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> rootOperands;
  ::llvm::SMLoc rootOperandsLoc;
  (void)rootOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> rootTypes;
  ::llvm::SmallVector<std::unique_ptr<::mlir::Region>, 2> fullRegions;

  {
    rootOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      rootOperands.push_back(operand);
    }
  }
  if (!rootOperands.empty()) {
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type optionalType;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalType(optionalType);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      rootTypes.push_back(optionalType);
    }
  }
  }
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return ::mlir::failure();

  {
    std::unique_ptr<::mlir::Region> region;
    auto firstRegionResult = parser.parseOptionalRegion(region);
    if (firstRegionResult.has_value()) {
      if (failed(*firstRegionResult))
        return ::mlir::failure();
      fullRegions.emplace_back(std::move(region));

      // Parse any trailing regions.
      while (succeeded(parser.parseOptionalComma())) {
        region = std::make_unique<::mlir::Region>();
        if (parser.parseRegion(*region))
          return ::mlir::failure();
        fullRegions.emplace_back(std::move(region));
      }
    }
  }
  result.addRegions(fullRegions);
  if (parser.resolveOperands(rootOperands, rootTypes, rootOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void WithPDLPatternsOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  if (getRoot()) {
    _odsPrinter << ' ';
    if (::mlir::Value value = getRoot())
      _odsPrinter << value;
    _odsPrinter << ' ' << ":";
    _odsPrinter << ' ';
    _odsPrinter << (getRoot() ? ::llvm::ArrayRef<::mlir::Type>(getRoot().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  }
  _odsPrinter.printOptionalAttrDictWithKeyword((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ';
    llvm::interleaveComma(getOperation()->getRegions(), _odsPrinter, [&](::mlir::Region &region) {
        _odsPrinter.printRegion(region);
    });
}

void WithPDLPatternsOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  for (::mlir::Value value : getODSOperands(0))
    effects.emplace_back(::mlir::MemoryEffects::Read::get(), value, ::mlir::transform::TransformMappingResource::get());
}

} // namespace transform
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::transform::WithPDLPatternsOp)

namespace mlir {
namespace transform {

//===----------------------------------------------------------------------===//
// ::mlir::transform::YieldOp definitions
//===----------------------------------------------------------------------===//

YieldOpAdaptor::YieldOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("transform.yield", odsAttrs.getContext());
}

YieldOpAdaptor::YieldOpAdaptor(YieldOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange YieldOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> YieldOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange YieldOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange YieldOpAdaptor::operands() {
  return getODSOperands(0);
}

::mlir::DictionaryAttr YieldOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult YieldOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> YieldOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range YieldOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range YieldOp::operands() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange YieldOp::operandsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> YieldOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range YieldOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void YieldOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {
      return build(odsBuilder, odsState, ::mlir::ValueRange());
    
}

void YieldOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands) {
  odsState.addOperands(operands);
}

void YieldOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult YieldOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TransformOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult YieldOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult YieldOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> allOperands;
  ::llvm::SmallVector<::mlir::Type, 1> operandsTypes;
  ::llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(allOperands))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalColon())) {

  if (parser.parseTypeList(operandsTypes))
    return ::mlir::failure();
  }
  if (parser.resolveOperands(allOperands, operandsTypes, allOperandLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void YieldOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getOperation()->getOperands();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  if (!operands().empty()) {
    _odsPrinter << ' ' << ":";
    _odsPrinter << ' ';
    _odsPrinter << operands().getTypes();
  }
}

void YieldOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  for (::mlir::Value value : getODSOperands(0))
    effects.emplace_back(::mlir::MemoryEffects::Read::get(), value, ::mlir::transform::TransformMappingResource::get());
}

} // namespace transform
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::transform::YieldOp)


#endif  // GET_OP_CLASSES

