/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::transform::MemRefMultiBufferOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace transform {

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_MemRefTransformOps0(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isa<::mlir::pdl::OperationType>()))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be PDL handle to an `mlir::Operation *`, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_MemRefTransformOps0(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))) && ((attr.cast<::mlir::IntegerAttr>().getValue().isStrictlyPositive())))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 64-bit signless integer attribute whose value is positive";
  }
  return ::mlir::success();
}
} // namespace transform
} // namespace mlir
namespace mlir {
namespace transform {

//===----------------------------------------------------------------------===//
// ::mlir::transform::MemRefMultiBufferOp definitions
//===----------------------------------------------------------------------===//

MemRefMultiBufferOpAdaptor::MemRefMultiBufferOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("transform.memref.multibuffer", odsAttrs.getContext());
}

MemRefMultiBufferOpAdaptor::MemRefMultiBufferOpAdaptor(MemRefMultiBufferOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange MemRefMultiBufferOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MemRefMultiBufferOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange MemRefMultiBufferOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MemRefMultiBufferOpAdaptor::getTarget() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr MemRefMultiBufferOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr MemRefMultiBufferOpAdaptor::getFactorAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, MemRefMultiBufferOp::getFactorAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
  return attr;
}

uint64_t MemRefMultiBufferOpAdaptor::getFactor() {
  auto attr = getFactorAttr();
  return attr.getValue().getZExtValue();
}

::mlir::LogicalResult MemRefMultiBufferOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_factor;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'transform.memref.multibuffer' op ""requires attribute 'factor'");
    if (namedAttrIt->getName() == MemRefMultiBufferOp::getFactorAttrName(*odsOpName)) {
      tblgen_factor = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_factor && !((((tblgen_factor.isa<::mlir::IntegerAttr>())) && ((tblgen_factor.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))) && ((tblgen_factor.cast<::mlir::IntegerAttr>().getValue().isStrictlyPositive()))))
    return emitError(loc, "'transform.memref.multibuffer' op ""attribute 'factor' failed to satisfy constraint: 64-bit signless integer attribute whose value is positive");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> MemRefMultiBufferOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range MemRefMultiBufferOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::OperationType> MemRefMultiBufferOp::getTarget() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange MemRefMultiBufferOp::getTargetMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> MemRefMultiBufferOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MemRefMultiBufferOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::OperationType> MemRefMultiBufferOp::getTransformed() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr MemRefMultiBufferOp::getFactorAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getFactorAttrName()).cast<::mlir::IntegerAttr>();
}

uint64_t MemRefMultiBufferOp::getFactor() {
  auto attr = getFactorAttr();
  return attr.getValue().getZExtValue();
}

void MemRefMultiBufferOp::setFactorAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getFactorAttrName(), attr);
}

void MemRefMultiBufferOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type transformed, ::mlir::Value target, ::mlir::IntegerAttr factor) {
  odsState.addOperands(target);
  odsState.addAttribute(getFactorAttrName(odsState.name), factor);
  odsState.addTypes(transformed);
}

void MemRefMultiBufferOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target, ::mlir::IntegerAttr factor) {
  odsState.addOperands(target);
  odsState.addAttribute(getFactorAttrName(odsState.name), factor);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MemRefMultiBufferOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type transformed, ::mlir::Value target, uint64_t factor) {
  odsState.addOperands(target);
  odsState.addAttribute(getFactorAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), factor));
  odsState.addTypes(transformed);
}

void MemRefMultiBufferOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target, uint64_t factor) {
  odsState.addOperands(target);
  odsState.addAttribute(getFactorAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), factor));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MemRefMultiBufferOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult MemRefMultiBufferOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_factor;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'factor'");
    if (namedAttrIt->getName() == getFactorAttrName()) {
      tblgen_factor = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_MemRefTransformOps0(*this, tblgen_factor, "factor")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefTransformOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefTransformOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult MemRefMultiBufferOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult MemRefMultiBufferOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand targetRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> targetOperands(targetRawOperands);  ::llvm::SMLoc targetOperandsLoc;
  (void)targetOperandsLoc;

  targetOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(targetRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(targetOperands, odsBuildableType0, targetOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void MemRefMultiBufferOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getTarget();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

} // namespace transform
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::transform::MemRefMultiBufferOp)


#endif  // GET_OP_CLASSES

