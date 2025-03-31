/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::transform::MapForeachToBlocks,
::mlir::transform::MapNestedForeachToThreads
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace transform {

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_GPUTransformOps0(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isa<::mlir::pdl::OperationType>()))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be PDL handle to an `mlir::Operation *`, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_GPUTransformOps0(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(attr.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))); })))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 64-bit integer array attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_GPUTransformOps1(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::UnitAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: unit attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_GPUTransformOps2(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::BoolAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: bool attribute";
  }
  return ::mlir::success();
}
} // namespace transform
} // namespace mlir
namespace mlir {
namespace transform {

//===----------------------------------------------------------------------===//
// ::mlir::transform::MapForeachToBlocks definitions
//===----------------------------------------------------------------------===//

MapForeachToBlocksAdaptor::MapForeachToBlocksAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("transform.gpu.map_foreach_to_blocks", odsAttrs.getContext());
}

MapForeachToBlocksAdaptor::MapForeachToBlocksAdaptor(MapForeachToBlocks op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange MapForeachToBlocksAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MapForeachToBlocksAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange MapForeachToBlocksAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MapForeachToBlocksAdaptor::getTarget() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr MapForeachToBlocksAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr MapForeachToBlocksAdaptor::getGridDimAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, MapForeachToBlocks::getGridDimAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr MapForeachToBlocksAdaptor::getGridDim() {
  auto attr = getGridDimAttr();
  return attr;
}

::mlir::UnitAttr MapForeachToBlocksAdaptor::getGenerateGpuLaunchAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, MapForeachToBlocks::getGenerateGpuLaunchAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool MapForeachToBlocksAdaptor::getGenerateGpuLaunch() {
  auto attr = getGenerateGpuLaunchAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::LogicalResult MapForeachToBlocksAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_generate_gpu_launch;
  ::mlir::Attribute tblgen_gridDim;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == MapForeachToBlocks::getGenerateGpuLaunchAttrName(*odsOpName)) {
      tblgen_generate_gpu_launch = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == MapForeachToBlocks::getGridDimAttrName(*odsOpName)) {
      tblgen_gridDim = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_gridDim && !(((tblgen_gridDim.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_gridDim.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))); }))))
    return emitError(loc, "'transform.gpu.map_foreach_to_blocks' op ""attribute 'gridDim' failed to satisfy constraint: 64-bit integer array attribute");

  if (tblgen_generate_gpu_launch && !((tblgen_generate_gpu_launch.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'transform.gpu.map_foreach_to_blocks' op ""attribute 'generate_gpu_launch' failed to satisfy constraint: unit attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> MapForeachToBlocks::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range MapForeachToBlocks::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::OperationType> MapForeachToBlocks::getTarget() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange MapForeachToBlocks::getTargetMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> MapForeachToBlocks::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MapForeachToBlocks::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::OperationType> MapForeachToBlocks::getResult() {
  return *getODSResults(0).begin();
}

::mlir::ArrayAttr MapForeachToBlocks::getGridDimAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getGridDimAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr MapForeachToBlocks::getGridDim() {
  auto attr = getGridDimAttr();
  return attr;
}

::mlir::UnitAttr MapForeachToBlocks::getGenerateGpuLaunchAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getGenerateGpuLaunchAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool MapForeachToBlocks::getGenerateGpuLaunch() {
  auto attr = getGenerateGpuLaunchAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

void MapForeachToBlocks::setGridDimAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getGridDimAttrName(), attr);
}

void MapForeachToBlocks::setGenerateGpuLaunchAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getGenerateGpuLaunchAttrName(), attr);
}

::mlir::Attribute MapForeachToBlocks::removeGenerate_gpu_launchAttr() {
  return (*this)->removeAttr(getGenerateGpuLaunchAttrName());
}

void MapForeachToBlocks::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value target, ::mlir::ArrayAttr gridDim, /*optional*/::mlir::UnitAttr generate_gpu_launch) {
  odsState.addOperands(target);
  if (gridDim) {
    odsState.addAttribute(getGridDimAttrName(odsState.name), gridDim);
  }
  if (generate_gpu_launch) {
    odsState.addAttribute(getGenerateGpuLaunchAttrName(odsState.name), generate_gpu_launch);
  }
  odsState.addTypes(result);
}

void MapForeachToBlocks::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target, ::mlir::ArrayAttr gridDim, /*optional*/::mlir::UnitAttr generate_gpu_launch) {
  odsState.addOperands(target);
  if (gridDim) {
    odsState.addAttribute(getGridDimAttrName(odsState.name), gridDim);
  }
  if (generate_gpu_launch) {
    odsState.addAttribute(getGenerateGpuLaunchAttrName(odsState.name), generate_gpu_launch);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MapForeachToBlocks::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value target, ::mlir::ArrayAttr gridDim, /*optional*/bool generate_gpu_launch) {
  odsState.addOperands(target);
  odsState.addAttribute(getGridDimAttrName(odsState.name), gridDim);
  if (generate_gpu_launch) {
    odsState.addAttribute(getGenerateGpuLaunchAttrName(odsState.name), ((generate_gpu_launch) ? odsBuilder.getUnitAttr() : nullptr));
  }
  odsState.addTypes(result);
}

void MapForeachToBlocks::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target, ::mlir::ArrayAttr gridDim, /*optional*/bool generate_gpu_launch) {
  odsState.addOperands(target);
  odsState.addAttribute(getGridDimAttrName(odsState.name), gridDim);
  if (generate_gpu_launch) {
    odsState.addAttribute(getGenerateGpuLaunchAttrName(odsState.name), ((generate_gpu_launch) ? odsBuilder.getUnitAttr() : nullptr));
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MapForeachToBlocks::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void MapForeachToBlocks::populateDefaultAttrs(const ::mlir::RegisteredOperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
  if (!attributes.get(attrNames[1])) {
    attributes.append(attrNames[1], odsBuilder.getI64ArrayAttr({}));
  }
}

::mlir::LogicalResult MapForeachToBlocks::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_generate_gpu_launch;
  ::mlir::Attribute tblgen_gridDim;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getGenerateGpuLaunchAttrName()) {
      tblgen_generate_gpu_launch = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getGridDimAttrName()) {
      tblgen_gridDim = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_GPUTransformOps0(*this, tblgen_gridDim, "gridDim")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_GPUTransformOps1(*this, tblgen_generate_gpu_launch, "generate_gpu_launch")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_GPUTransformOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_GPUTransformOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult MapForeachToBlocks::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult MapForeachToBlocks::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
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

void MapForeachToBlocks::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getTarget();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

} // namespace transform
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::transform::MapForeachToBlocks)

namespace mlir {
namespace transform {

//===----------------------------------------------------------------------===//
// ::mlir::transform::MapNestedForeachToThreads definitions
//===----------------------------------------------------------------------===//

MapNestedForeachToThreadsAdaptor::MapNestedForeachToThreadsAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("transform.gpu.map_nested_foreach_to_threads", odsAttrs.getContext());
}

MapNestedForeachToThreadsAdaptor::MapNestedForeachToThreadsAdaptor(MapNestedForeachToThreads op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange MapNestedForeachToThreadsAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MapNestedForeachToThreadsAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange MapNestedForeachToThreadsAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MapNestedForeachToThreadsAdaptor::getTarget() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr MapNestedForeachToThreadsAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr MapNestedForeachToThreadsAdaptor::getBlockDimAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, MapNestedForeachToThreads::getBlockDimAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr MapNestedForeachToThreadsAdaptor::getBlockDim() {
  auto attr = getBlockDimAttr();
  return attr;
}

::mlir::BoolAttr MapNestedForeachToThreadsAdaptor::getSyncAfterDistributeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, MapNestedForeachToThreads::getSyncAfterDistributeAttrName(*odsOpName)).dyn_cast_or_null<::mlir::BoolAttr>();
  return attr;
}

bool MapNestedForeachToThreadsAdaptor::getSyncAfterDistribute() {
  auto attr = getSyncAfterDistributeAttr();
  return attr.getValue();
}

::mlir::LogicalResult MapNestedForeachToThreadsAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_blockDim;
  ::mlir::Attribute tblgen_syncAfterDistribute;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == MapNestedForeachToThreads::getBlockDimAttrName(*odsOpName)) {
      tblgen_blockDim = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == MapNestedForeachToThreads::getSyncAfterDistributeAttrName(*odsOpName)) {
      tblgen_syncAfterDistribute = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_blockDim && !(((tblgen_blockDim.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_blockDim.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))); }))))
    return emitError(loc, "'transform.gpu.map_nested_foreach_to_threads' op ""attribute 'blockDim' failed to satisfy constraint: 64-bit integer array attribute");

  if (tblgen_syncAfterDistribute && !((tblgen_syncAfterDistribute.isa<::mlir::BoolAttr>())))
    return emitError(loc, "'transform.gpu.map_nested_foreach_to_threads' op ""attribute 'syncAfterDistribute' failed to satisfy constraint: bool attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> MapNestedForeachToThreads::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range MapNestedForeachToThreads::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::OperationType> MapNestedForeachToThreads::getTarget() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange MapNestedForeachToThreads::getTargetMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> MapNestedForeachToThreads::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MapNestedForeachToThreads::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::OperationType> MapNestedForeachToThreads::getResult() {
  return *getODSResults(0).begin();
}

::mlir::ArrayAttr MapNestedForeachToThreads::getBlockDimAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getBlockDimAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr MapNestedForeachToThreads::getBlockDim() {
  auto attr = getBlockDimAttr();
  return attr;
}

::mlir::BoolAttr MapNestedForeachToThreads::getSyncAfterDistributeAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getSyncAfterDistributeAttrName()).dyn_cast_or_null<::mlir::BoolAttr>();
}

bool MapNestedForeachToThreads::getSyncAfterDistribute() {
  auto attr = getSyncAfterDistributeAttr();
  return attr.getValue();
}

void MapNestedForeachToThreads::setBlockDimAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getBlockDimAttrName(), attr);
}

void MapNestedForeachToThreads::setSyncAfterDistributeAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr(getSyncAfterDistributeAttrName(), attr);
}

void MapNestedForeachToThreads::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value target, ::mlir::ArrayAttr blockDim, ::mlir::BoolAttr syncAfterDistribute) {
  odsState.addOperands(target);
  if (blockDim) {
    odsState.addAttribute(getBlockDimAttrName(odsState.name), blockDim);
  }
  if (syncAfterDistribute) {
    odsState.addAttribute(getSyncAfterDistributeAttrName(odsState.name), syncAfterDistribute);
  }
  odsState.addTypes(result);
}

void MapNestedForeachToThreads::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target, ::mlir::ArrayAttr blockDim, ::mlir::BoolAttr syncAfterDistribute) {
  odsState.addOperands(target);
  if (blockDim) {
    odsState.addAttribute(getBlockDimAttrName(odsState.name), blockDim);
  }
  if (syncAfterDistribute) {
    odsState.addAttribute(getSyncAfterDistributeAttrName(odsState.name), syncAfterDistribute);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MapNestedForeachToThreads::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value target, ::mlir::ArrayAttr blockDim, bool syncAfterDistribute) {
  odsState.addOperands(target);
  odsState.addAttribute(getBlockDimAttrName(odsState.name), blockDim);
  odsState.addAttribute(getSyncAfterDistributeAttrName(odsState.name), odsBuilder.getBoolAttr(syncAfterDistribute));
  odsState.addTypes(result);
}

void MapNestedForeachToThreads::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target, ::mlir::ArrayAttr blockDim, bool syncAfterDistribute) {
  odsState.addOperands(target);
  odsState.addAttribute(getBlockDimAttrName(odsState.name), blockDim);
  odsState.addAttribute(getSyncAfterDistributeAttrName(odsState.name), odsBuilder.getBoolAttr(syncAfterDistribute));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MapNestedForeachToThreads::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void MapNestedForeachToThreads::populateDefaultAttrs(const ::mlir::RegisteredOperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
  if (!attributes.get(attrNames[0])) {
    attributes.append(attrNames[0], odsBuilder.getI64ArrayAttr({}));
  }
  if (!attributes.get(attrNames[1])) {
    attributes.append(attrNames[1], odsBuilder.getBoolAttr(true));
  }
}

::mlir::LogicalResult MapNestedForeachToThreads::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_blockDim;
  ::mlir::Attribute tblgen_syncAfterDistribute;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getBlockDimAttrName()) {
      tblgen_blockDim = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getSyncAfterDistributeAttrName()) {
      tblgen_syncAfterDistribute = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_GPUTransformOps0(*this, tblgen_blockDim, "blockDim")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_GPUTransformOps2(*this, tblgen_syncAfterDistribute, "syncAfterDistribute")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_GPUTransformOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_GPUTransformOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult MapNestedForeachToThreads::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult MapNestedForeachToThreads::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
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

void MapNestedForeachToThreads::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getTarget();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

} // namespace transform
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::transform::MapNestedForeachToThreads)


#endif  // GET_OP_CLASSES

