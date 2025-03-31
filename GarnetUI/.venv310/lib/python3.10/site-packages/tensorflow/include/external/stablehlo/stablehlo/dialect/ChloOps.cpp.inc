/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::chlo::AcosOp,
::mlir::chlo::AcoshOp,
::mlir::chlo::AsinOp,
::mlir::chlo::AsinhOp,
::mlir::chlo::AtanOp,
::mlir::chlo::AtanhOp,
::mlir::chlo::BesselI1eOp,
::mlir::chlo::BroadcastAddOp,
::mlir::chlo::BroadcastAndOp,
::mlir::chlo::BroadcastAtan2Op,
::mlir::chlo::BroadcastCompareOp,
::mlir::chlo::BroadcastComplexOp,
::mlir::chlo::BroadcastDivOp,
::mlir::chlo::BroadcastMaxOp,
::mlir::chlo::BroadcastMinOp,
::mlir::chlo::BroadcastMulOp,
::mlir::chlo::BroadcastNextAfterOp,
::mlir::chlo::BroadcastOrOp,
::mlir::chlo::BroadcastPolygammaOp,
::mlir::chlo::BroadcastPowOp,
::mlir::chlo::BroadcastRemOp,
::mlir::chlo::BroadcastSelectOp,
::mlir::chlo::BroadcastShiftLeftOp,
::mlir::chlo::BroadcastShiftRightArithmeticOp,
::mlir::chlo::BroadcastShiftRightLogicalOp,
::mlir::chlo::BroadcastSubOp,
::mlir::chlo::BroadcastXorOp,
::mlir::chlo::BroadcastZetaOp,
::mlir::chlo::ConjOp,
::mlir::chlo::ConstantLikeOp,
::mlir::chlo::ConstantOp,
::mlir::chlo::CoshOp,
::mlir::chlo::DigammaOp,
::mlir::chlo::DynamicReshapeOp,
::mlir::chlo::ErfOp,
::mlir::chlo::ErfcOp,
::mlir::chlo::IsInfOp,
::mlir::chlo::IsNegInfOp,
::mlir::chlo::IsPosInfOp,
::mlir::chlo::LgammaOp,
::mlir::chlo::MinimumBroadcastShapesOp,
::mlir::chlo::NextAfterOp,
::mlir::chlo::PolygammaOp,
::mlir::chlo::RankSpecializationClusterOp,
::mlir::chlo::RankSpecializationClusterYieldOp,
::mlir::chlo::SinhOp,
::mlir::chlo::TanOp,
::mlir::chlo::TopKOp,
::mlir::chlo::ZetaOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace chlo {

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_ChloOps0(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isa<::mlir::TensorType>())) && ([](::mlir::Type elementType) { return (((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())) || ((elementType.isBF16()))) || (((elementType.isa<::mlir::ComplexType>())) && (((elementType.cast<::mlir::ComplexType>().getElementType().isF32())) || ((elementType.cast<::mlir::ComplexType>().getElementType().isF64())))); }(type.cast<::mlir::ShapedType>().getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be tensor of 16-bit float or 32-bit float or 64-bit float or bfloat16 type or complex type with 32-bit float or 64-bit float elements values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_ChloOps1(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isa<::mlir::TensorType>())) && ([](::mlir::Type elementType) { return (((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())) || ((elementType.isBF16()))) || ((elementType.isSignlessInteger(1))) || ((((elementType.isSignlessInteger(4))) || ((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(64)))) || (((elementType.isUnsignedInteger(4))) || ((elementType.isUnsignedInteger(8))) || ((elementType.isUnsignedInteger(16))) || ((elementType.isUnsignedInteger(32))) || ((elementType.isUnsignedInteger(64))))) || (((elementType.isa<::mlir::ComplexType>())) && (((elementType.cast<::mlir::ComplexType>().getElementType().isF32())) || ((elementType.cast<::mlir::ComplexType>().getElementType().isF64())))) || ((((((elementType.isa<mlir::quant::UniformQuantizedType>())) && ((elementType.cast<mlir::quant::UniformQuantizedType>().getStorageTypeIntegralWidth() == 4)) && ((elementType.cast<mlir::quant::UniformQuantizedType>().isSigned()))) || (((elementType.isa<mlir::quant::UniformQuantizedPerAxisType>())) && ((elementType.cast<mlir::quant::UniformQuantizedPerAxisType>().getStorageTypeIntegralWidth() == 4)) && ((elementType.cast<mlir::quant::UniformQuantizedPerAxisType>().isSigned())))) || ((((elementType.isa<mlir::quant::UniformQuantizedType>())) && ((elementType.cast<mlir::quant::UniformQuantizedType>().getStorageTypeIntegralWidth() == 8)) && ((elementType.cast<mlir::quant::UniformQuantizedType>().isSigned()))) || (((elementType.isa<mlir::quant::UniformQuantizedPerAxisType>())) && ((elementType.cast<mlir::quant::UniformQuantizedPerAxisType>().getStorageTypeIntegralWidth() == 8)) && ((elementType.cast<mlir::quant::UniformQuantizedPerAxisType>().isSigned())))) || ((((elementType.isa<mlir::quant::UniformQuantizedType>())) && ((elementType.cast<mlir::quant::UniformQuantizedType>().getStorageTypeIntegralWidth() == 16)) && ((elementType.cast<mlir::quant::UniformQuantizedType>().isSigned()))) || (((elementType.isa<mlir::quant::UniformQuantizedPerAxisType>())) && ((elementType.cast<mlir::quant::UniformQuantizedPerAxisType>().getStorageTypeIntegralWidth() == 16)) && ((elementType.cast<mlir::quant::UniformQuantizedPerAxisType>().isSigned())))) || ((((elementType.isa<mlir::quant::UniformQuantizedType>())) && ((elementType.cast<mlir::quant::UniformQuantizedType>().getStorageTypeIntegralWidth() == 32)) && ((elementType.cast<mlir::quant::UniformQuantizedType>().isSigned()))) || (((elementType.isa<mlir::quant::UniformQuantizedPerAxisType>())) && ((elementType.cast<mlir::quant::UniformQuantizedPerAxisType>().getStorageTypeIntegralWidth() == 32)) && ((elementType.cast<mlir::quant::UniformQuantizedPerAxisType>().isSigned()))))) || (((((elementType.isa<mlir::quant::UniformQuantizedType>())) && ((elementType.cast<mlir::quant::UniformQuantizedType>().getStorageTypeIntegralWidth() == 4)) && ((!elementType.cast<mlir::quant::UniformQuantizedType>().isSigned()))) || (((elementType.isa<mlir::quant::UniformQuantizedPerAxisType>())) && ((elementType.cast<mlir::quant::UniformQuantizedPerAxisType>().getStorageTypeIntegralWidth() == 4)) && ((!elementType.cast<mlir::quant::UniformQuantizedPerAxisType>().isSigned())))) || ((((elementType.isa<mlir::quant::UniformQuantizedType>())) && ((elementType.cast<mlir::quant::UniformQuantizedType>().getStorageTypeIntegralWidth() == 8)) && ((!elementType.cast<mlir::quant::UniformQuantizedType>().isSigned()))) || (((elementType.isa<mlir::quant::UniformQuantizedPerAxisType>())) && ((elementType.cast<mlir::quant::UniformQuantizedPerAxisType>().getStorageTypeIntegralWidth() == 8)) && ((!elementType.cast<mlir::quant::UniformQuantizedPerAxisType>().isSigned())))) || ((((elementType.isa<mlir::quant::UniformQuantizedType>())) && ((elementType.cast<mlir::quant::UniformQuantizedType>().getStorageTypeIntegralWidth() == 16)) && ((!elementType.cast<mlir::quant::UniformQuantizedType>().isSigned()))) || (((elementType.isa<mlir::quant::UniformQuantizedPerAxisType>())) && ((elementType.cast<mlir::quant::UniformQuantizedPerAxisType>().getStorageTypeIntegralWidth() == 16)) && ((!elementType.cast<mlir::quant::UniformQuantizedPerAxisType>().isSigned())))) || ((((elementType.isa<mlir::quant::UniformQuantizedType>())) && ((elementType.cast<mlir::quant::UniformQuantizedType>().getStorageTypeIntegralWidth() == 32)) && ((!elementType.cast<mlir::quant::UniformQuantizedType>().isSigned()))) || (((elementType.isa<mlir::quant::UniformQuantizedPerAxisType>())) && ((elementType.cast<mlir::quant::UniformQuantizedPerAxisType>().getStorageTypeIntegralWidth() == 32)) && ((!elementType.cast<mlir::quant::UniformQuantizedPerAxisType>().isSigned())))))); }(type.cast<::mlir::ShapedType>().getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be tensor of 16-bit float or 32-bit float or 64-bit float or bfloat16 type or pred (AKA boolean or 1-bit integer) or 4/8/16/32/64-bit signless integer or 4/8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements or 4/8/16/32-bit uniform quantized signed integer or 4/8/16/32-bit uniform quantized unsigned integer values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_ChloOps2(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isa<::mlir::TensorType>())) && ([](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || ((((elementType.isSignlessInteger(4))) || ((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(64)))) || (((elementType.isUnsignedInteger(4))) || ((elementType.isUnsignedInteger(8))) || ((elementType.isUnsignedInteger(16))) || ((elementType.isUnsignedInteger(32))) || ((elementType.isUnsignedInteger(64))))); }(type.cast<::mlir::ShapedType>().getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be tensor of pred (AKA boolean or 1-bit integer) or 4/8/16/32/64-bit signless integer or 4/8/16/32/64-bit unsigned integer values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_ChloOps3(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isa<::mlir::TensorType>())) && ([](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(type.cast<::mlir::ShapedType>().getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be tensor of pred (AKA boolean or 1-bit integer) values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_ChloOps4(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isa<::mlir::TensorType>())) && ([](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())) || ((elementType.isBF16())); }(type.cast<::mlir::ShapedType>().getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be tensor of 16-bit float or 32-bit float or 64-bit float or bfloat16 type values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_ChloOps5(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isa<::mlir::TensorType>())) && ([](::mlir::Type elementType) { return ((elementType.isa<::mlir::ComplexType>())) && (((elementType.cast<::mlir::ComplexType>().getElementType().isF32())) || ((elementType.cast<::mlir::ComplexType>().getElementType().isF64()))); }(type.cast<::mlir::ShapedType>().getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be tensor of complex type with 32-bit float or 64-bit float elements values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_ChloOps6(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((type.isa<::mlir::TensorType>())) && ((type.cast<::mlir::ShapedType>().hasStaticShape()))) && ([](::mlir::Type elementType) { return (((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())) || ((elementType.isBF16()))) || ((elementType.isSignlessInteger(1))) || ((((elementType.isSignlessInteger(4))) || ((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(64)))) || (((elementType.isUnsignedInteger(4))) || ((elementType.isUnsignedInteger(8))) || ((elementType.isUnsignedInteger(16))) || ((elementType.isUnsignedInteger(32))) || ((elementType.isUnsignedInteger(64))))) || (((elementType.isa<::mlir::ComplexType>())) && (((elementType.cast<::mlir::ComplexType>().getElementType().isF32())) || ((elementType.cast<::mlir::ComplexType>().getElementType().isF64())))) || ((((((elementType.isa<mlir::quant::UniformQuantizedType>())) && ((elementType.cast<mlir::quant::UniformQuantizedType>().getStorageTypeIntegralWidth() == 4)) && ((elementType.cast<mlir::quant::UniformQuantizedType>().isSigned()))) || (((elementType.isa<mlir::quant::UniformQuantizedPerAxisType>())) && ((elementType.cast<mlir::quant::UniformQuantizedPerAxisType>().getStorageTypeIntegralWidth() == 4)) && ((elementType.cast<mlir::quant::UniformQuantizedPerAxisType>().isSigned())))) || ((((elementType.isa<mlir::quant::UniformQuantizedType>())) && ((elementType.cast<mlir::quant::UniformQuantizedType>().getStorageTypeIntegralWidth() == 8)) && ((elementType.cast<mlir::quant::UniformQuantizedType>().isSigned()))) || (((elementType.isa<mlir::quant::UniformQuantizedPerAxisType>())) && ((elementType.cast<mlir::quant::UniformQuantizedPerAxisType>().getStorageTypeIntegralWidth() == 8)) && ((elementType.cast<mlir::quant::UniformQuantizedPerAxisType>().isSigned())))) || ((((elementType.isa<mlir::quant::UniformQuantizedType>())) && ((elementType.cast<mlir::quant::UniformQuantizedType>().getStorageTypeIntegralWidth() == 16)) && ((elementType.cast<mlir::quant::UniformQuantizedType>().isSigned()))) || (((elementType.isa<mlir::quant::UniformQuantizedPerAxisType>())) && ((elementType.cast<mlir::quant::UniformQuantizedPerAxisType>().getStorageTypeIntegralWidth() == 16)) && ((elementType.cast<mlir::quant::UniformQuantizedPerAxisType>().isSigned())))) || ((((elementType.isa<mlir::quant::UniformQuantizedType>())) && ((elementType.cast<mlir::quant::UniformQuantizedType>().getStorageTypeIntegralWidth() == 32)) && ((elementType.cast<mlir::quant::UniformQuantizedType>().isSigned()))) || (((elementType.isa<mlir::quant::UniformQuantizedPerAxisType>())) && ((elementType.cast<mlir::quant::UniformQuantizedPerAxisType>().getStorageTypeIntegralWidth() == 32)) && ((elementType.cast<mlir::quant::UniformQuantizedPerAxisType>().isSigned()))))) || (((((elementType.isa<mlir::quant::UniformQuantizedType>())) && ((elementType.cast<mlir::quant::UniformQuantizedType>().getStorageTypeIntegralWidth() == 4)) && ((!elementType.cast<mlir::quant::UniformQuantizedType>().isSigned()))) || (((elementType.isa<mlir::quant::UniformQuantizedPerAxisType>())) && ((elementType.cast<mlir::quant::UniformQuantizedPerAxisType>().getStorageTypeIntegralWidth() == 4)) && ((!elementType.cast<mlir::quant::UniformQuantizedPerAxisType>().isSigned())))) || ((((elementType.isa<mlir::quant::UniformQuantizedType>())) && ((elementType.cast<mlir::quant::UniformQuantizedType>().getStorageTypeIntegralWidth() == 8)) && ((!elementType.cast<mlir::quant::UniformQuantizedType>().isSigned()))) || (((elementType.isa<mlir::quant::UniformQuantizedPerAxisType>())) && ((elementType.cast<mlir::quant::UniformQuantizedPerAxisType>().getStorageTypeIntegralWidth() == 8)) && ((!elementType.cast<mlir::quant::UniformQuantizedPerAxisType>().isSigned())))) || ((((elementType.isa<mlir::quant::UniformQuantizedType>())) && ((elementType.cast<mlir::quant::UniformQuantizedType>().getStorageTypeIntegralWidth() == 16)) && ((!elementType.cast<mlir::quant::UniformQuantizedType>().isSigned()))) || (((elementType.isa<mlir::quant::UniformQuantizedPerAxisType>())) && ((elementType.cast<mlir::quant::UniformQuantizedPerAxisType>().getStorageTypeIntegralWidth() == 16)) && ((!elementType.cast<mlir::quant::UniformQuantizedPerAxisType>().isSigned())))) || ((((elementType.isa<mlir::quant::UniformQuantizedType>())) && ((elementType.cast<mlir::quant::UniformQuantizedType>().getStorageTypeIntegralWidth() == 32)) && ((!elementType.cast<mlir::quant::UniformQuantizedType>().isSigned()))) || (((elementType.isa<mlir::quant::UniformQuantizedPerAxisType>())) && ((elementType.cast<mlir::quant::UniformQuantizedPerAxisType>().getStorageTypeIntegralWidth() == 32)) && ((!elementType.cast<mlir::quant::UniformQuantizedPerAxisType>().isSigned())))))); }(type.cast<::mlir::ShapedType>().getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be statically shaped tensor of 16-bit float or 32-bit float or 64-bit float or bfloat16 type or pred (AKA boolean or 1-bit integer) or 4/8/16/32/64-bit signless integer or 4/8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements or 4/8/16/32-bit uniform quantized signed integer or 4/8/16/32-bit uniform quantized unsigned integer values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_ChloOps7(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((type.isa<::mlir::TensorType>())) && (((type.cast<::mlir::ShapedType>().hasRank())) && ((type.cast<::mlir::ShapedType>().getRank()
                         == 1)))) && ([](::mlir::Type elementType) { return ((elementType.isa<::mlir::IndexType>())) || ((((elementType.isSignlessInteger(4))) || ((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(64)))) || (((elementType.isUnsignedInteger(4))) || ((elementType.isUnsignedInteger(8))) || ((elementType.isUnsignedInteger(16))) || ((elementType.isUnsignedInteger(32))) || ((elementType.isUnsignedInteger(64))))); }(type.cast<::mlir::ShapedType>().getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be 1D tensor of index or 4/8/16/32/64-bit signless integer or 4/8/16/32/64-bit unsigned integer values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_ChloOps8(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((type.isa<::mlir::TensorType>())) && (((type.cast<::mlir::ShapedType>().hasRank())) && ((type.cast<::mlir::ShapedType>().getRank()
                         == 1)))) && ([](::mlir::Type elementType) { return (elementType.isa<::mlir::IndexType>()); }(type.cast<::mlir::ShapedType>().getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be 1D tensor of index values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_ChloOps0(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::DenseIntElementsAttr>())) && ((attr.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64))))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 64-bit signless integer elements attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_ChloOps1(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::chlo::ComparisonDirectionAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: Which comparison operation to perform.";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_ChloOps2(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::chlo::ComparisonTypeAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: Which comparison type to use.";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_ChloOps3(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::TypedAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: TypedAttr instance";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_ChloOps4(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::ElementsAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: constant vector/tensor attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_ChloOps5(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 64-bit signless integer attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_region_constraint_ChloOps0(
    ::mlir::Operation *op, ::mlir::Region &region, ::llvm::StringRef regionName,
    unsigned regionIndex) {
  if (!((::llvm::hasNItems(region, 1)))) {
    return op->emitOpError("region #") << regionIndex
        << (regionName.empty() ? " " : " ('" + regionName + "') ")
        << "failed to verify constraint: region with 1 blocks";
  }
  return ::mlir::success();
}
} // namespace chlo
} // namespace mlir
namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// ::mlir::chlo::AcosOp definitions
//===----------------------------------------------------------------------===//

AcosOpAdaptor::AcosOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("chlo.acos", odsAttrs.getContext());
}

AcosOpAdaptor::AcosOpAdaptor(AcosOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange AcosOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> AcosOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange AcosOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AcosOpAdaptor::getOperand() {
  return *getODSOperands(0).begin();
}

::mlir::Value AcosOpAdaptor::operand() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr AcosOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult AcosOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AcosOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AcosOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> AcosOp::getOperand() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> AcosOp::operand() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange AcosOp::getOperandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange AcosOp::operandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> AcosOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AcosOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> AcosOp::getResult() {
  return *getODSResults(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> AcosOp::result() {
  return *getODSResults(0).begin();
}

void AcosOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes(result);
}

void AcosOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand) {
  odsState.addOperands(operand);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(AcosOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void AcosOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand) {
  odsState.addOperands(operand);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AcosOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void AcosOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(AcosOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult AcosOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult AcosOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult AcosOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand operandRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> operandOperands(operandRawOperands);  ::llvm::SMLoc operandOperandsLoc;
  (void)operandOperandsLoc;
  ::mlir::Type operandRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> operandTypes(operandRawTypes);
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);

  operandOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(operandRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    operandRawTypes[0] = type;
  }
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawTypes[0] = type;
  }
  result.addTypes(resultTypes);
  if (parser.resolveOperands(operandOperands, operandTypes, operandOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void AcosOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getOperand();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getOperand().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void AcosOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace chlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::chlo::AcosOp)

namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// ::mlir::chlo::AcoshOp definitions
//===----------------------------------------------------------------------===//

AcoshOpAdaptor::AcoshOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("chlo.acosh", odsAttrs.getContext());
}

AcoshOpAdaptor::AcoshOpAdaptor(AcoshOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange AcoshOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> AcoshOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange AcoshOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AcoshOpAdaptor::getOperand() {
  return *getODSOperands(0).begin();
}

::mlir::Value AcoshOpAdaptor::operand() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr AcoshOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult AcoshOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AcoshOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AcoshOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> AcoshOp::getOperand() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> AcoshOp::operand() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange AcoshOp::getOperandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange AcoshOp::operandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> AcoshOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AcoshOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> AcoshOp::getResult() {
  return *getODSResults(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> AcoshOp::result() {
  return *getODSResults(0).begin();
}

void AcoshOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes(result);
}

void AcoshOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand) {
  odsState.addOperands(operand);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(AcoshOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void AcoshOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand) {
  odsState.addOperands(operand);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AcoshOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void AcoshOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(AcoshOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult AcoshOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult AcoshOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult AcoshOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand operandRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> operandOperands(operandRawOperands);  ::llvm::SMLoc operandOperandsLoc;
  (void)operandOperandsLoc;
  ::mlir::Type operandRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> operandTypes(operandRawTypes);
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);

  operandOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(operandRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    operandRawTypes[0] = type;
  }
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawTypes[0] = type;
  }
  result.addTypes(resultTypes);
  if (parser.resolveOperands(operandOperands, operandTypes, operandOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void AcoshOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getOperand();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getOperand().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void AcoshOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace chlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::chlo::AcoshOp)

namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// ::mlir::chlo::AsinOp definitions
//===----------------------------------------------------------------------===//

AsinOpAdaptor::AsinOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("chlo.asin", odsAttrs.getContext());
}

AsinOpAdaptor::AsinOpAdaptor(AsinOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange AsinOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> AsinOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange AsinOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AsinOpAdaptor::getOperand() {
  return *getODSOperands(0).begin();
}

::mlir::Value AsinOpAdaptor::operand() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr AsinOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult AsinOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AsinOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AsinOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> AsinOp::getOperand() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> AsinOp::operand() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange AsinOp::getOperandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange AsinOp::operandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> AsinOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AsinOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> AsinOp::getResult() {
  return *getODSResults(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> AsinOp::result() {
  return *getODSResults(0).begin();
}

void AsinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes(result);
}

void AsinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand) {
  odsState.addOperands(operand);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(AsinOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void AsinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand) {
  odsState.addOperands(operand);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AsinOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void AsinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(AsinOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult AsinOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult AsinOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult AsinOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand operandRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> operandOperands(operandRawOperands);  ::llvm::SMLoc operandOperandsLoc;
  (void)operandOperandsLoc;
  ::mlir::Type operandRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> operandTypes(operandRawTypes);
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);

  operandOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(operandRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    operandRawTypes[0] = type;
  }
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawTypes[0] = type;
  }
  result.addTypes(resultTypes);
  if (parser.resolveOperands(operandOperands, operandTypes, operandOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void AsinOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getOperand();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getOperand().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void AsinOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace chlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::chlo::AsinOp)

namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// ::mlir::chlo::AsinhOp definitions
//===----------------------------------------------------------------------===//

AsinhOpAdaptor::AsinhOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("chlo.asinh", odsAttrs.getContext());
}

AsinhOpAdaptor::AsinhOpAdaptor(AsinhOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange AsinhOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> AsinhOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange AsinhOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AsinhOpAdaptor::getOperand() {
  return *getODSOperands(0).begin();
}

::mlir::Value AsinhOpAdaptor::operand() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr AsinhOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult AsinhOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AsinhOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AsinhOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> AsinhOp::getOperand() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> AsinhOp::operand() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange AsinhOp::getOperandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange AsinhOp::operandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> AsinhOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AsinhOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> AsinhOp::getResult() {
  return *getODSResults(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> AsinhOp::result() {
  return *getODSResults(0).begin();
}

void AsinhOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes(result);
}

void AsinhOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand) {
  odsState.addOperands(operand);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(AsinhOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void AsinhOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand) {
  odsState.addOperands(operand);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AsinhOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void AsinhOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(AsinhOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult AsinhOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult AsinhOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult AsinhOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand operandRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> operandOperands(operandRawOperands);  ::llvm::SMLoc operandOperandsLoc;
  (void)operandOperandsLoc;
  ::mlir::Type operandRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> operandTypes(operandRawTypes);
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);

  operandOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(operandRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    operandRawTypes[0] = type;
  }
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawTypes[0] = type;
  }
  result.addTypes(resultTypes);
  if (parser.resolveOperands(operandOperands, operandTypes, operandOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void AsinhOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getOperand();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getOperand().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void AsinhOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace chlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::chlo::AsinhOp)

namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// ::mlir::chlo::AtanOp definitions
//===----------------------------------------------------------------------===//

AtanOpAdaptor::AtanOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("chlo.atan", odsAttrs.getContext());
}

AtanOpAdaptor::AtanOpAdaptor(AtanOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange AtanOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> AtanOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange AtanOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AtanOpAdaptor::getOperand() {
  return *getODSOperands(0).begin();
}

::mlir::Value AtanOpAdaptor::operand() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr AtanOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult AtanOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AtanOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AtanOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> AtanOp::getOperand() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> AtanOp::operand() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange AtanOp::getOperandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange AtanOp::operandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> AtanOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AtanOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> AtanOp::getResult() {
  return *getODSResults(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> AtanOp::result() {
  return *getODSResults(0).begin();
}

void AtanOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes(result);
}

void AtanOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand) {
  odsState.addOperands(operand);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(AtanOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void AtanOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand) {
  odsState.addOperands(operand);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AtanOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void AtanOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(AtanOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult AtanOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult AtanOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult AtanOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand operandRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> operandOperands(operandRawOperands);  ::llvm::SMLoc operandOperandsLoc;
  (void)operandOperandsLoc;
  ::mlir::Type operandRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> operandTypes(operandRawTypes);
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);

  operandOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(operandRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    operandRawTypes[0] = type;
  }
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawTypes[0] = type;
  }
  result.addTypes(resultTypes);
  if (parser.resolveOperands(operandOperands, operandTypes, operandOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void AtanOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getOperand();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getOperand().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void AtanOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace chlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::chlo::AtanOp)

namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// ::mlir::chlo::AtanhOp definitions
//===----------------------------------------------------------------------===//

AtanhOpAdaptor::AtanhOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("chlo.atanh", odsAttrs.getContext());
}

AtanhOpAdaptor::AtanhOpAdaptor(AtanhOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange AtanhOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> AtanhOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange AtanhOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AtanhOpAdaptor::getOperand() {
  return *getODSOperands(0).begin();
}

::mlir::Value AtanhOpAdaptor::operand() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr AtanhOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult AtanhOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AtanhOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AtanhOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> AtanhOp::getOperand() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> AtanhOp::operand() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange AtanhOp::getOperandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange AtanhOp::operandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> AtanhOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AtanhOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> AtanhOp::getResult() {
  return *getODSResults(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> AtanhOp::result() {
  return *getODSResults(0).begin();
}

void AtanhOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes(result);
}

void AtanhOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand) {
  odsState.addOperands(operand);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(AtanhOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void AtanhOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand) {
  odsState.addOperands(operand);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AtanhOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void AtanhOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(AtanhOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult AtanhOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult AtanhOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult AtanhOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand operandRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> operandOperands(operandRawOperands);  ::llvm::SMLoc operandOperandsLoc;
  (void)operandOperandsLoc;
  ::mlir::Type operandRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> operandTypes(operandRawTypes);
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);

  operandOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(operandRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    operandRawTypes[0] = type;
  }
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawTypes[0] = type;
  }
  result.addTypes(resultTypes);
  if (parser.resolveOperands(operandOperands, operandTypes, operandOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void AtanhOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getOperand();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getOperand().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void AtanhOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace chlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::chlo::AtanhOp)

namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// ::mlir::chlo::BesselI1eOp definitions
//===----------------------------------------------------------------------===//

BesselI1eOpAdaptor::BesselI1eOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("chlo.bessel_i1e", odsAttrs.getContext());
}

BesselI1eOpAdaptor::BesselI1eOpAdaptor(BesselI1eOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange BesselI1eOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> BesselI1eOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BesselI1eOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BesselI1eOpAdaptor::getOperand() {
  return *getODSOperands(0).begin();
}

::mlir::Value BesselI1eOpAdaptor::operand() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr BesselI1eOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult BesselI1eOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> BesselI1eOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BesselI1eOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> BesselI1eOp::getOperand() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> BesselI1eOp::operand() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange BesselI1eOp::getOperandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BesselI1eOp::operandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> BesselI1eOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BesselI1eOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> BesselI1eOp::getResult() {
  return *getODSResults(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> BesselI1eOp::result() {
  return *getODSResults(0).begin();
}

void BesselI1eOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes(result);
}

void BesselI1eOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand) {
  odsState.addOperands(operand);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(BesselI1eOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void BesselI1eOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand) {
  odsState.addOperands(operand);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BesselI1eOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void BesselI1eOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(BesselI1eOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult BesselI1eOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult BesselI1eOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult BesselI1eOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand operandRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> operandOperands(operandRawOperands);  ::llvm::SMLoc operandOperandsLoc;
  (void)operandOperandsLoc;
  ::mlir::Type operandRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> operandTypes(operandRawTypes);
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);

  operandOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(operandRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    operandRawTypes[0] = type;
  }
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawTypes[0] = type;
  }
  result.addTypes(resultTypes);
  if (parser.resolveOperands(operandOperands, operandTypes, operandOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void BesselI1eOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getOperand();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getOperand().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void BesselI1eOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace chlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::chlo::BesselI1eOp)

namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// ::mlir::chlo::BroadcastAddOp definitions
//===----------------------------------------------------------------------===//

BroadcastAddOpAdaptor::BroadcastAddOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("chlo.broadcast_add", odsAttrs.getContext());
}

BroadcastAddOpAdaptor::BroadcastAddOpAdaptor(BroadcastAddOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange BroadcastAddOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> BroadcastAddOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BroadcastAddOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BroadcastAddOpAdaptor::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastAddOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastAddOpAdaptor::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value BroadcastAddOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr BroadcastAddOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr BroadcastAddOpAdaptor::getBroadcastDimensionsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, BroadcastAddOp::getBroadcastDimensionsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastAddOpAdaptor::getBroadcastDimensions() {
  auto attr = getBroadcastDimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::DenseIntElementsAttr BroadcastAddOpAdaptor::broadcast_dimensionsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, BroadcastAddOp::getBroadcastDimensionsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastAddOpAdaptor::broadcast_dimensions() {
  auto attr = broadcast_dimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::LogicalResult BroadcastAddOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_broadcast_dimensions;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == BroadcastAddOp::getBroadcastDimensionsAttrName(*odsOpName)) {
      tblgen_broadcast_dimensions = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_broadcast_dimensions && !(((tblgen_broadcast_dimensions.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_broadcast_dimensions.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))))
    return emitError(loc, "'chlo.broadcast_add' op ""attribute 'broadcast_dimensions' failed to satisfy constraint: 64-bit signless integer elements attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> BroadcastAddOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BroadcastAddOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> BroadcastAddOp::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> BroadcastAddOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> BroadcastAddOp::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::TypedValue<::mlir::TensorType> BroadcastAddOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange BroadcastAddOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BroadcastAddOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BroadcastAddOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BroadcastAddOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> BroadcastAddOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BroadcastAddOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::DenseIntElementsAttr BroadcastAddOp::getBroadcastDimensionsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getBroadcastDimensionsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastAddOp::getBroadcastDimensions() {
  auto attr = getBroadcastDimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::DenseIntElementsAttr BroadcastAddOp::broadcast_dimensionsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getBroadcastDimensionsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastAddOp::broadcast_dimensions() {
  auto attr = broadcast_dimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

void BroadcastAddOp::setBroadcastDimensionsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getBroadcastDimensionsAttrName(), attr);
}

void BroadcastAddOp::broadcast_dimensionsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(broadcast_dimensionsAttrName(), attr);
}

::mlir::Attribute BroadcastAddOp::removeBroadcast_dimensionsAttr() {
  return (*this)->removeAttr(getBroadcastDimensionsAttrName());
}

void BroadcastAddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
    odsState.addAttribute(getBroadcastDimensionsAttrName(odsState.name), broadcast_dimensions);
  }
  odsState.addTypes(resultType0);
}

void BroadcastAddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
    odsState.addAttribute(getBroadcastDimensionsAttrName(odsState.name), broadcast_dimensions);
  }

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(BroadcastAddOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void BroadcastAddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
    odsState.addAttribute(getBroadcastDimensionsAttrName(odsState.name), broadcast_dimensions);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BroadcastAddOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void BroadcastAddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(BroadcastAddOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult BroadcastAddOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_broadcast_dimensions;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getBroadcastDimensionsAttrName()) {
      tblgen_broadcast_dimensions = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ChloOps0(*this, tblgen_broadcast_dimensions, "broadcast_dimensions")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult BroadcastAddOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult BroadcastAddOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type lhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> lhsTypes(lhsRawTypes);
  ::mlir::Type rhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> rhsTypes(rhsRawTypes);
  ::llvm::SmallVector<::mlir::Type, 1> allResultTypes;

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    lhsRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    rhsRawTypes[0] = type;
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseTypeList(allResultTypes))
    return ::mlir::failure();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(lhsOperands, lhsTypes, lhsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, rhsTypes, rhsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void BroadcastAddOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getLhs();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getRhs();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ' << "(";
  {
    auto type = getLhs().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getRhs().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ")";
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  _odsPrinter << getOperation()->getResultTypes();
}

void BroadcastAddOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace chlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::chlo::BroadcastAddOp)

namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// ::mlir::chlo::BroadcastAndOp definitions
//===----------------------------------------------------------------------===//

BroadcastAndOpAdaptor::BroadcastAndOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("chlo.broadcast_and", odsAttrs.getContext());
}

BroadcastAndOpAdaptor::BroadcastAndOpAdaptor(BroadcastAndOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange BroadcastAndOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> BroadcastAndOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BroadcastAndOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BroadcastAndOpAdaptor::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastAndOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastAndOpAdaptor::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value BroadcastAndOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr BroadcastAndOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr BroadcastAndOpAdaptor::getBroadcastDimensionsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, BroadcastAndOp::getBroadcastDimensionsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastAndOpAdaptor::getBroadcastDimensions() {
  auto attr = getBroadcastDimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::DenseIntElementsAttr BroadcastAndOpAdaptor::broadcast_dimensionsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, BroadcastAndOp::getBroadcastDimensionsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastAndOpAdaptor::broadcast_dimensions() {
  auto attr = broadcast_dimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::LogicalResult BroadcastAndOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_broadcast_dimensions;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == BroadcastAndOp::getBroadcastDimensionsAttrName(*odsOpName)) {
      tblgen_broadcast_dimensions = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_broadcast_dimensions && !(((tblgen_broadcast_dimensions.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_broadcast_dimensions.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))))
    return emitError(loc, "'chlo.broadcast_and' op ""attribute 'broadcast_dimensions' failed to satisfy constraint: 64-bit signless integer elements attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> BroadcastAndOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BroadcastAndOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> BroadcastAndOp::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> BroadcastAndOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> BroadcastAndOp::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::TypedValue<::mlir::TensorType> BroadcastAndOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange BroadcastAndOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BroadcastAndOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BroadcastAndOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BroadcastAndOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> BroadcastAndOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BroadcastAndOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::DenseIntElementsAttr BroadcastAndOp::getBroadcastDimensionsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getBroadcastDimensionsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastAndOp::getBroadcastDimensions() {
  auto attr = getBroadcastDimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::DenseIntElementsAttr BroadcastAndOp::broadcast_dimensionsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getBroadcastDimensionsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastAndOp::broadcast_dimensions() {
  auto attr = broadcast_dimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

void BroadcastAndOp::setBroadcastDimensionsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getBroadcastDimensionsAttrName(), attr);
}

void BroadcastAndOp::broadcast_dimensionsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(broadcast_dimensionsAttrName(), attr);
}

::mlir::Attribute BroadcastAndOp::removeBroadcast_dimensionsAttr() {
  return (*this)->removeAttr(getBroadcastDimensionsAttrName());
}

void BroadcastAndOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
    odsState.addAttribute(getBroadcastDimensionsAttrName(odsState.name), broadcast_dimensions);
  }
  odsState.addTypes(resultType0);
}

void BroadcastAndOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
    odsState.addAttribute(getBroadcastDimensionsAttrName(odsState.name), broadcast_dimensions);
  }

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(BroadcastAndOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void BroadcastAndOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
    odsState.addAttribute(getBroadcastDimensionsAttrName(odsState.name), broadcast_dimensions);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BroadcastAndOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void BroadcastAndOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(BroadcastAndOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult BroadcastAndOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_broadcast_dimensions;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getBroadcastDimensionsAttrName()) {
      tblgen_broadcast_dimensions = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ChloOps0(*this, tblgen_broadcast_dimensions, "broadcast_dimensions")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult BroadcastAndOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult BroadcastAndOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type lhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> lhsTypes(lhsRawTypes);
  ::mlir::Type rhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> rhsTypes(rhsRawTypes);
  ::llvm::SmallVector<::mlir::Type, 1> allResultTypes;

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    lhsRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    rhsRawTypes[0] = type;
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseTypeList(allResultTypes))
    return ::mlir::failure();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(lhsOperands, lhsTypes, lhsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, rhsTypes, rhsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void BroadcastAndOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getLhs();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getRhs();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ' << "(";
  {
    auto type = getLhs().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getRhs().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ")";
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  _odsPrinter << getOperation()->getResultTypes();
}

void BroadcastAndOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace chlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::chlo::BroadcastAndOp)

namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// ::mlir::chlo::BroadcastAtan2Op definitions
//===----------------------------------------------------------------------===//

BroadcastAtan2OpAdaptor::BroadcastAtan2OpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("chlo.broadcast_atan2", odsAttrs.getContext());
}

BroadcastAtan2OpAdaptor::BroadcastAtan2OpAdaptor(BroadcastAtan2Op op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange BroadcastAtan2OpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> BroadcastAtan2OpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BroadcastAtan2OpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BroadcastAtan2OpAdaptor::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastAtan2OpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastAtan2OpAdaptor::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value BroadcastAtan2OpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr BroadcastAtan2OpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr BroadcastAtan2OpAdaptor::getBroadcastDimensionsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, BroadcastAtan2Op::getBroadcastDimensionsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastAtan2OpAdaptor::getBroadcastDimensions() {
  auto attr = getBroadcastDimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::DenseIntElementsAttr BroadcastAtan2OpAdaptor::broadcast_dimensionsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, BroadcastAtan2Op::getBroadcastDimensionsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastAtan2OpAdaptor::broadcast_dimensions() {
  auto attr = broadcast_dimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::LogicalResult BroadcastAtan2OpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_broadcast_dimensions;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == BroadcastAtan2Op::getBroadcastDimensionsAttrName(*odsOpName)) {
      tblgen_broadcast_dimensions = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_broadcast_dimensions && !(((tblgen_broadcast_dimensions.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_broadcast_dimensions.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))))
    return emitError(loc, "'chlo.broadcast_atan2' op ""attribute 'broadcast_dimensions' failed to satisfy constraint: 64-bit signless integer elements attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> BroadcastAtan2Op::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BroadcastAtan2Op::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> BroadcastAtan2Op::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> BroadcastAtan2Op::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> BroadcastAtan2Op::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::TypedValue<::mlir::TensorType> BroadcastAtan2Op::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange BroadcastAtan2Op::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BroadcastAtan2Op::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BroadcastAtan2Op::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BroadcastAtan2Op::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> BroadcastAtan2Op::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BroadcastAtan2Op::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::DenseIntElementsAttr BroadcastAtan2Op::getBroadcastDimensionsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getBroadcastDimensionsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastAtan2Op::getBroadcastDimensions() {
  auto attr = getBroadcastDimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::DenseIntElementsAttr BroadcastAtan2Op::broadcast_dimensionsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getBroadcastDimensionsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastAtan2Op::broadcast_dimensions() {
  auto attr = broadcast_dimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

void BroadcastAtan2Op::setBroadcastDimensionsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getBroadcastDimensionsAttrName(), attr);
}

void BroadcastAtan2Op::broadcast_dimensionsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(broadcast_dimensionsAttrName(), attr);
}

::mlir::Attribute BroadcastAtan2Op::removeBroadcast_dimensionsAttr() {
  return (*this)->removeAttr(getBroadcastDimensionsAttrName());
}

void BroadcastAtan2Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
    odsState.addAttribute(getBroadcastDimensionsAttrName(odsState.name), broadcast_dimensions);
  }
  odsState.addTypes(resultType0);
}

void BroadcastAtan2Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
    odsState.addAttribute(getBroadcastDimensionsAttrName(odsState.name), broadcast_dimensions);
  }

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(BroadcastAtan2Op::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void BroadcastAtan2Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
    odsState.addAttribute(getBroadcastDimensionsAttrName(odsState.name), broadcast_dimensions);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BroadcastAtan2Op::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void BroadcastAtan2Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(BroadcastAtan2Op::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult BroadcastAtan2Op::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_broadcast_dimensions;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getBroadcastDimensionsAttrName()) {
      tblgen_broadcast_dimensions = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ChloOps0(*this, tblgen_broadcast_dimensions, "broadcast_dimensions")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult BroadcastAtan2Op::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult BroadcastAtan2Op::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type lhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> lhsTypes(lhsRawTypes);
  ::mlir::Type rhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> rhsTypes(rhsRawTypes);
  ::llvm::SmallVector<::mlir::Type, 1> allResultTypes;

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    lhsRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    rhsRawTypes[0] = type;
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseTypeList(allResultTypes))
    return ::mlir::failure();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(lhsOperands, lhsTypes, lhsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, rhsTypes, rhsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void BroadcastAtan2Op::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getLhs();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getRhs();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ' << "(";
  {
    auto type = getLhs().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getRhs().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ")";
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  _odsPrinter << getOperation()->getResultTypes();
}

void BroadcastAtan2Op::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace chlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::chlo::BroadcastAtan2Op)

namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// ::mlir::chlo::BroadcastCompareOp definitions
//===----------------------------------------------------------------------===//

BroadcastCompareOpAdaptor::BroadcastCompareOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("chlo.broadcast_compare", odsAttrs.getContext());
}

BroadcastCompareOpAdaptor::BroadcastCompareOpAdaptor(BroadcastCompareOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange BroadcastCompareOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> BroadcastCompareOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BroadcastCompareOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BroadcastCompareOpAdaptor::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastCompareOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastCompareOpAdaptor::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value BroadcastCompareOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr BroadcastCompareOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr BroadcastCompareOpAdaptor::getBroadcastDimensionsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, BroadcastCompareOp::getBroadcastDimensionsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastCompareOpAdaptor::getBroadcastDimensions() {
  auto attr = getBroadcastDimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::DenseIntElementsAttr BroadcastCompareOpAdaptor::broadcast_dimensionsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, BroadcastCompareOp::getBroadcastDimensionsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastCompareOpAdaptor::broadcast_dimensions() {
  auto attr = broadcast_dimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::chlo::ComparisonDirectionAttr BroadcastCompareOpAdaptor::getComparisonDirectionAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, BroadcastCompareOp::getComparisonDirectionAttrName(*odsOpName)).cast<::mlir::chlo::ComparisonDirectionAttr>();
  return attr;
}

::mlir::chlo::ComparisonDirection BroadcastCompareOpAdaptor::getComparisonDirection() {
  auto attr = getComparisonDirectionAttr();
  return attr.getValue();
}

::mlir::chlo::ComparisonDirectionAttr BroadcastCompareOpAdaptor::comparison_directionAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, BroadcastCompareOp::getComparisonDirectionAttrName(*odsOpName)).cast<::mlir::chlo::ComparisonDirectionAttr>();
  return attr;
}

::mlir::chlo::ComparisonDirection BroadcastCompareOpAdaptor::comparison_direction() {
  auto attr = comparison_directionAttr();
  return attr.getValue();
}

::mlir::chlo::ComparisonTypeAttr BroadcastCompareOpAdaptor::getCompareTypeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, BroadcastCompareOp::getCompareTypeAttrName(*odsOpName)).dyn_cast_or_null<::mlir::chlo::ComparisonTypeAttr>();
  return attr;
}

::llvm::Optional<::mlir::chlo::ComparisonType> BroadcastCompareOpAdaptor::getCompareType() {
  auto attr = getCompareTypeAttr();
  return attr ? ::llvm::Optional<::mlir::chlo::ComparisonType>(attr.getValue()) : (::llvm::None);
}

::mlir::chlo::ComparisonTypeAttr BroadcastCompareOpAdaptor::compare_typeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, BroadcastCompareOp::getCompareTypeAttrName(*odsOpName)).dyn_cast_or_null<::mlir::chlo::ComparisonTypeAttr>();
  return attr;
}

::llvm::Optional<::mlir::chlo::ComparisonType> BroadcastCompareOpAdaptor::compare_type() {
  auto attr = compare_typeAttr();
  return attr ? ::llvm::Optional<::mlir::chlo::ComparisonType>(attr.getValue()) : (::llvm::None);
}

::mlir::LogicalResult BroadcastCompareOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_comparison_direction;
  ::mlir::Attribute tblgen_broadcast_dimensions;
  ::mlir::Attribute tblgen_compare_type;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'chlo.broadcast_compare' op ""requires attribute 'comparison_direction'");
    if (namedAttrIt->getName() == BroadcastCompareOp::getComparisonDirectionAttrName(*odsOpName)) {
      tblgen_comparison_direction = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == BroadcastCompareOp::getBroadcastDimensionsAttrName(*odsOpName)) {
      tblgen_broadcast_dimensions = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == BroadcastCompareOp::getCompareTypeAttrName(*odsOpName)) {
      tblgen_compare_type = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_broadcast_dimensions && !(((tblgen_broadcast_dimensions.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_broadcast_dimensions.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))))
    return emitError(loc, "'chlo.broadcast_compare' op ""attribute 'broadcast_dimensions' failed to satisfy constraint: 64-bit signless integer elements attribute");

  if (tblgen_comparison_direction && !((tblgen_comparison_direction.isa<::mlir::chlo::ComparisonDirectionAttr>())))
    return emitError(loc, "'chlo.broadcast_compare' op ""attribute 'comparison_direction' failed to satisfy constraint: Which comparison operation to perform.");

  if (tblgen_compare_type && !((tblgen_compare_type.isa<::mlir::chlo::ComparisonTypeAttr>())))
    return emitError(loc, "'chlo.broadcast_compare' op ""attribute 'compare_type' failed to satisfy constraint: Which comparison type to use.");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> BroadcastCompareOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BroadcastCompareOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> BroadcastCompareOp::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> BroadcastCompareOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> BroadcastCompareOp::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::TypedValue<::mlir::TensorType> BroadcastCompareOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange BroadcastCompareOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BroadcastCompareOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BroadcastCompareOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BroadcastCompareOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> BroadcastCompareOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BroadcastCompareOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::DenseIntElementsAttr BroadcastCompareOp::getBroadcastDimensionsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getBroadcastDimensionsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastCompareOp::getBroadcastDimensions() {
  auto attr = getBroadcastDimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::DenseIntElementsAttr BroadcastCompareOp::broadcast_dimensionsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getBroadcastDimensionsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastCompareOp::broadcast_dimensions() {
  auto attr = broadcast_dimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::chlo::ComparisonDirectionAttr BroadcastCompareOp::getComparisonDirectionAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getComparisonDirectionAttrName()).cast<::mlir::chlo::ComparisonDirectionAttr>();
}

::mlir::chlo::ComparisonDirection BroadcastCompareOp::getComparisonDirection() {
  auto attr = getComparisonDirectionAttr();
  return attr.getValue();
}

::mlir::chlo::ComparisonDirectionAttr BroadcastCompareOp::comparison_directionAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getComparisonDirectionAttrName()).cast<::mlir::chlo::ComparisonDirectionAttr>();
}

::mlir::chlo::ComparisonDirection BroadcastCompareOp::comparison_direction() {
  auto attr = comparison_directionAttr();
  return attr.getValue();
}

::mlir::chlo::ComparisonTypeAttr BroadcastCompareOp::getCompareTypeAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getCompareTypeAttrName()).dyn_cast_or_null<::mlir::chlo::ComparisonTypeAttr>();
}

::llvm::Optional<::mlir::chlo::ComparisonType> BroadcastCompareOp::getCompareType() {
  auto attr = getCompareTypeAttr();
  return attr ? ::llvm::Optional<::mlir::chlo::ComparisonType>(attr.getValue()) : (::llvm::None);
}

::mlir::chlo::ComparisonTypeAttr BroadcastCompareOp::compare_typeAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getCompareTypeAttrName()).dyn_cast_or_null<::mlir::chlo::ComparisonTypeAttr>();
}

::llvm::Optional<::mlir::chlo::ComparisonType> BroadcastCompareOp::compare_type() {
  auto attr = compare_typeAttr();
  return attr ? ::llvm::Optional<::mlir::chlo::ComparisonType>(attr.getValue()) : (::llvm::None);
}

void BroadcastCompareOp::setBroadcastDimensionsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getBroadcastDimensionsAttrName(), attr);
}

void BroadcastCompareOp::broadcast_dimensionsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(broadcast_dimensionsAttrName(), attr);
}

void BroadcastCompareOp::setComparisonDirectionAttr(::mlir::chlo::ComparisonDirectionAttr attr) {
  (*this)->setAttr(getComparisonDirectionAttrName(), attr);
}

void BroadcastCompareOp::comparison_directionAttr(::mlir::chlo::ComparisonDirectionAttr attr) {
  (*this)->setAttr(comparison_directionAttrName(), attr);
}

void BroadcastCompareOp::setCompareTypeAttr(::mlir::chlo::ComparisonTypeAttr attr) {
  (*this)->setAttr(getCompareTypeAttrName(), attr);
}

void BroadcastCompareOp::compare_typeAttr(::mlir::chlo::ComparisonTypeAttr attr) {
  (*this)->setAttr(compare_typeAttrName(), attr);
}

::mlir::Attribute BroadcastCompareOp::removeBroadcast_dimensionsAttr() {
  return (*this)->removeAttr(getBroadcastDimensionsAttrName());
}

::mlir::Attribute BroadcastCompareOp::removeCompare_typeAttr() {
  return (*this)->removeAttr(getCompareTypeAttrName());
}

void BroadcastCompareOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions, ::mlir::chlo::ComparisonDirectionAttr comparison_direction, /*optional*/::mlir::chlo::ComparisonTypeAttr compare_type) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
    odsState.addAttribute(getBroadcastDimensionsAttrName(odsState.name), broadcast_dimensions);
  }
  odsState.addAttribute(getComparisonDirectionAttrName(odsState.name), comparison_direction);
  if (compare_type) {
    odsState.addAttribute(getCompareTypeAttrName(odsState.name), compare_type);
  }
  odsState.addTypes(resultType0);
}

void BroadcastCompareOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions, ::mlir::chlo::ComparisonDirectionAttr comparison_direction, /*optional*/::mlir::chlo::ComparisonTypeAttr compare_type) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
    odsState.addAttribute(getBroadcastDimensionsAttrName(odsState.name), broadcast_dimensions);
  }
  odsState.addAttribute(getComparisonDirectionAttrName(odsState.name), comparison_direction);
  if (compare_type) {
    odsState.addAttribute(getCompareTypeAttrName(odsState.name), compare_type);
  }

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(BroadcastCompareOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void BroadcastCompareOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions, ::mlir::chlo::ComparisonDirectionAttr comparison_direction, /*optional*/::mlir::chlo::ComparisonTypeAttr compare_type) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
    odsState.addAttribute(getBroadcastDimensionsAttrName(odsState.name), broadcast_dimensions);
  }
  odsState.addAttribute(getComparisonDirectionAttrName(odsState.name), comparison_direction);
  if (compare_type) {
    odsState.addAttribute(getCompareTypeAttrName(odsState.name), compare_type);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BroadcastCompareOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions, ::mlir::chlo::ComparisonDirection comparison_direction, /*optional*/::mlir::chlo::ComparisonTypeAttr compare_type) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
    odsState.addAttribute(getBroadcastDimensionsAttrName(odsState.name), broadcast_dimensions);
  }
  odsState.addAttribute(getComparisonDirectionAttrName(odsState.name), ::mlir::chlo::ComparisonDirectionAttr::get(odsBuilder.getContext(), comparison_direction));
  if (compare_type) {
    odsState.addAttribute(getCompareTypeAttrName(odsState.name), compare_type);
  }
  odsState.addTypes(resultType0);
}

void BroadcastCompareOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions, ::mlir::chlo::ComparisonDirection comparison_direction, /*optional*/::mlir::chlo::ComparisonTypeAttr compare_type) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
    odsState.addAttribute(getBroadcastDimensionsAttrName(odsState.name), broadcast_dimensions);
  }
  odsState.addAttribute(getComparisonDirectionAttrName(odsState.name), ::mlir::chlo::ComparisonDirectionAttr::get(odsBuilder.getContext(), comparison_direction));
  if (compare_type) {
    odsState.addAttribute(getCompareTypeAttrName(odsState.name), compare_type);
  }

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(BroadcastCompareOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void BroadcastCompareOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions, ::mlir::chlo::ComparisonDirection comparison_direction, /*optional*/::mlir::chlo::ComparisonTypeAttr compare_type) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
    odsState.addAttribute(getBroadcastDimensionsAttrName(odsState.name), broadcast_dimensions);
  }
  odsState.addAttribute(getComparisonDirectionAttrName(odsState.name), ::mlir::chlo::ComparisonDirectionAttr::get(odsBuilder.getContext(), comparison_direction));
  if (compare_type) {
    odsState.addAttribute(getCompareTypeAttrName(odsState.name), compare_type);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BroadcastCompareOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void BroadcastCompareOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(BroadcastCompareOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult BroadcastCompareOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_comparison_direction;
  ::mlir::Attribute tblgen_broadcast_dimensions;
  ::mlir::Attribute tblgen_compare_type;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'comparison_direction'");
    if (namedAttrIt->getName() == getComparisonDirectionAttrName()) {
      tblgen_comparison_direction = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getBroadcastDimensionsAttrName()) {
      tblgen_broadcast_dimensions = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getCompareTypeAttrName()) {
      tblgen_compare_type = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ChloOps0(*this, tblgen_broadcast_dimensions, "broadcast_dimensions")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ChloOps1(*this, tblgen_comparison_direction, "comparison_direction")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ChloOps2(*this, tblgen_compare_type, "compare_type")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult BroadcastCompareOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult BroadcastCompareOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type lhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> lhsTypes(lhsRawTypes);
  ::mlir::Type rhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> rhsTypes(rhsRawTypes);
  ::llvm::SmallVector<::mlir::Type, 1> allResultTypes;

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    lhsRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    rhsRawTypes[0] = type;
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseTypeList(allResultTypes))
    return ::mlir::failure();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(lhsOperands, lhsTypes, lhsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, rhsTypes, rhsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void BroadcastCompareOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getLhs();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getRhs();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ' << "(";
  {
    auto type = getLhs().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getRhs().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ")";
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  _odsPrinter << getOperation()->getResultTypes();
}

void BroadcastCompareOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace chlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::chlo::BroadcastCompareOp)

namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// ::mlir::chlo::BroadcastComplexOp definitions
//===----------------------------------------------------------------------===//

BroadcastComplexOpAdaptor::BroadcastComplexOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("chlo.broadcast_complex", odsAttrs.getContext());
}

BroadcastComplexOpAdaptor::BroadcastComplexOpAdaptor(BroadcastComplexOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange BroadcastComplexOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> BroadcastComplexOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BroadcastComplexOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BroadcastComplexOpAdaptor::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastComplexOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastComplexOpAdaptor::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value BroadcastComplexOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr BroadcastComplexOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr BroadcastComplexOpAdaptor::getBroadcastDimensionsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, BroadcastComplexOp::getBroadcastDimensionsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastComplexOpAdaptor::getBroadcastDimensions() {
  auto attr = getBroadcastDimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::DenseIntElementsAttr BroadcastComplexOpAdaptor::broadcast_dimensionsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, BroadcastComplexOp::getBroadcastDimensionsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastComplexOpAdaptor::broadcast_dimensions() {
  auto attr = broadcast_dimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::LogicalResult BroadcastComplexOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_broadcast_dimensions;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == BroadcastComplexOp::getBroadcastDimensionsAttrName(*odsOpName)) {
      tblgen_broadcast_dimensions = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_broadcast_dimensions && !(((tblgen_broadcast_dimensions.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_broadcast_dimensions.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))))
    return emitError(loc, "'chlo.broadcast_complex' op ""attribute 'broadcast_dimensions' failed to satisfy constraint: 64-bit signless integer elements attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> BroadcastComplexOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BroadcastComplexOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> BroadcastComplexOp::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> BroadcastComplexOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> BroadcastComplexOp::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::TypedValue<::mlir::TensorType> BroadcastComplexOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange BroadcastComplexOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BroadcastComplexOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BroadcastComplexOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BroadcastComplexOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> BroadcastComplexOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BroadcastComplexOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::DenseIntElementsAttr BroadcastComplexOp::getBroadcastDimensionsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getBroadcastDimensionsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastComplexOp::getBroadcastDimensions() {
  auto attr = getBroadcastDimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::DenseIntElementsAttr BroadcastComplexOp::broadcast_dimensionsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getBroadcastDimensionsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastComplexOp::broadcast_dimensions() {
  auto attr = broadcast_dimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

void BroadcastComplexOp::setBroadcastDimensionsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getBroadcastDimensionsAttrName(), attr);
}

void BroadcastComplexOp::broadcast_dimensionsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(broadcast_dimensionsAttrName(), attr);
}

::mlir::Attribute BroadcastComplexOp::removeBroadcast_dimensionsAttr() {
  return (*this)->removeAttr(getBroadcastDimensionsAttrName());
}

void BroadcastComplexOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
    odsState.addAttribute(getBroadcastDimensionsAttrName(odsState.name), broadcast_dimensions);
  }
  odsState.addTypes(resultType0);
}

void BroadcastComplexOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
    odsState.addAttribute(getBroadcastDimensionsAttrName(odsState.name), broadcast_dimensions);
  }

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(BroadcastComplexOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void BroadcastComplexOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
    odsState.addAttribute(getBroadcastDimensionsAttrName(odsState.name), broadcast_dimensions);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BroadcastComplexOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void BroadcastComplexOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(BroadcastComplexOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult BroadcastComplexOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_broadcast_dimensions;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getBroadcastDimensionsAttrName()) {
      tblgen_broadcast_dimensions = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ChloOps0(*this, tblgen_broadcast_dimensions, "broadcast_dimensions")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps5(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult BroadcastComplexOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult BroadcastComplexOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type lhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> lhsTypes(lhsRawTypes);
  ::mlir::Type rhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> rhsTypes(rhsRawTypes);
  ::llvm::SmallVector<::mlir::Type, 1> allResultTypes;

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    lhsRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    rhsRawTypes[0] = type;
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseTypeList(allResultTypes))
    return ::mlir::failure();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(lhsOperands, lhsTypes, lhsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, rhsTypes, rhsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void BroadcastComplexOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getLhs();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getRhs();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ' << "(";
  {
    auto type = getLhs().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getRhs().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ")";
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  _odsPrinter << getOperation()->getResultTypes();
}

void BroadcastComplexOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace chlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::chlo::BroadcastComplexOp)

namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// ::mlir::chlo::BroadcastDivOp definitions
//===----------------------------------------------------------------------===//

BroadcastDivOpAdaptor::BroadcastDivOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("chlo.broadcast_divide", odsAttrs.getContext());
}

BroadcastDivOpAdaptor::BroadcastDivOpAdaptor(BroadcastDivOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange BroadcastDivOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> BroadcastDivOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BroadcastDivOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BroadcastDivOpAdaptor::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastDivOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastDivOpAdaptor::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value BroadcastDivOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr BroadcastDivOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr BroadcastDivOpAdaptor::getBroadcastDimensionsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, BroadcastDivOp::getBroadcastDimensionsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastDivOpAdaptor::getBroadcastDimensions() {
  auto attr = getBroadcastDimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::DenseIntElementsAttr BroadcastDivOpAdaptor::broadcast_dimensionsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, BroadcastDivOp::getBroadcastDimensionsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastDivOpAdaptor::broadcast_dimensions() {
  auto attr = broadcast_dimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::LogicalResult BroadcastDivOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_broadcast_dimensions;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == BroadcastDivOp::getBroadcastDimensionsAttrName(*odsOpName)) {
      tblgen_broadcast_dimensions = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_broadcast_dimensions && !(((tblgen_broadcast_dimensions.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_broadcast_dimensions.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))))
    return emitError(loc, "'chlo.broadcast_divide' op ""attribute 'broadcast_dimensions' failed to satisfy constraint: 64-bit signless integer elements attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> BroadcastDivOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BroadcastDivOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> BroadcastDivOp::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> BroadcastDivOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> BroadcastDivOp::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::TypedValue<::mlir::TensorType> BroadcastDivOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange BroadcastDivOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BroadcastDivOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BroadcastDivOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BroadcastDivOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> BroadcastDivOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BroadcastDivOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::DenseIntElementsAttr BroadcastDivOp::getBroadcastDimensionsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getBroadcastDimensionsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastDivOp::getBroadcastDimensions() {
  auto attr = getBroadcastDimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::DenseIntElementsAttr BroadcastDivOp::broadcast_dimensionsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getBroadcastDimensionsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastDivOp::broadcast_dimensions() {
  auto attr = broadcast_dimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

void BroadcastDivOp::setBroadcastDimensionsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getBroadcastDimensionsAttrName(), attr);
}

void BroadcastDivOp::broadcast_dimensionsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(broadcast_dimensionsAttrName(), attr);
}

::mlir::Attribute BroadcastDivOp::removeBroadcast_dimensionsAttr() {
  return (*this)->removeAttr(getBroadcastDimensionsAttrName());
}

void BroadcastDivOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
    odsState.addAttribute(getBroadcastDimensionsAttrName(odsState.name), broadcast_dimensions);
  }
  odsState.addTypes(resultType0);
}

void BroadcastDivOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
    odsState.addAttribute(getBroadcastDimensionsAttrName(odsState.name), broadcast_dimensions);
  }

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(BroadcastDivOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void BroadcastDivOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
    odsState.addAttribute(getBroadcastDimensionsAttrName(odsState.name), broadcast_dimensions);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BroadcastDivOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void BroadcastDivOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(BroadcastDivOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult BroadcastDivOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_broadcast_dimensions;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getBroadcastDimensionsAttrName()) {
      tblgen_broadcast_dimensions = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ChloOps0(*this, tblgen_broadcast_dimensions, "broadcast_dimensions")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult BroadcastDivOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult BroadcastDivOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type lhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> lhsTypes(lhsRawTypes);
  ::mlir::Type rhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> rhsTypes(rhsRawTypes);
  ::llvm::SmallVector<::mlir::Type, 1> allResultTypes;

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    lhsRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    rhsRawTypes[0] = type;
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseTypeList(allResultTypes))
    return ::mlir::failure();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(lhsOperands, lhsTypes, lhsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, rhsTypes, rhsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void BroadcastDivOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getLhs();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getRhs();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ' << "(";
  {
    auto type = getLhs().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getRhs().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ")";
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  _odsPrinter << getOperation()->getResultTypes();
}

void BroadcastDivOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace chlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::chlo::BroadcastDivOp)

namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// ::mlir::chlo::BroadcastMaxOp definitions
//===----------------------------------------------------------------------===//

BroadcastMaxOpAdaptor::BroadcastMaxOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("chlo.broadcast_maximum", odsAttrs.getContext());
}

BroadcastMaxOpAdaptor::BroadcastMaxOpAdaptor(BroadcastMaxOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange BroadcastMaxOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> BroadcastMaxOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BroadcastMaxOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BroadcastMaxOpAdaptor::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastMaxOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastMaxOpAdaptor::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value BroadcastMaxOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr BroadcastMaxOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr BroadcastMaxOpAdaptor::getBroadcastDimensionsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, BroadcastMaxOp::getBroadcastDimensionsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastMaxOpAdaptor::getBroadcastDimensions() {
  auto attr = getBroadcastDimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::DenseIntElementsAttr BroadcastMaxOpAdaptor::broadcast_dimensionsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, BroadcastMaxOp::getBroadcastDimensionsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastMaxOpAdaptor::broadcast_dimensions() {
  auto attr = broadcast_dimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::LogicalResult BroadcastMaxOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_broadcast_dimensions;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == BroadcastMaxOp::getBroadcastDimensionsAttrName(*odsOpName)) {
      tblgen_broadcast_dimensions = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_broadcast_dimensions && !(((tblgen_broadcast_dimensions.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_broadcast_dimensions.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))))
    return emitError(loc, "'chlo.broadcast_maximum' op ""attribute 'broadcast_dimensions' failed to satisfy constraint: 64-bit signless integer elements attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> BroadcastMaxOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BroadcastMaxOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> BroadcastMaxOp::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> BroadcastMaxOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> BroadcastMaxOp::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::TypedValue<::mlir::TensorType> BroadcastMaxOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange BroadcastMaxOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BroadcastMaxOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BroadcastMaxOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BroadcastMaxOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> BroadcastMaxOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BroadcastMaxOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::DenseIntElementsAttr BroadcastMaxOp::getBroadcastDimensionsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getBroadcastDimensionsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastMaxOp::getBroadcastDimensions() {
  auto attr = getBroadcastDimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::DenseIntElementsAttr BroadcastMaxOp::broadcast_dimensionsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getBroadcastDimensionsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastMaxOp::broadcast_dimensions() {
  auto attr = broadcast_dimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

void BroadcastMaxOp::setBroadcastDimensionsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getBroadcastDimensionsAttrName(), attr);
}

void BroadcastMaxOp::broadcast_dimensionsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(broadcast_dimensionsAttrName(), attr);
}

::mlir::Attribute BroadcastMaxOp::removeBroadcast_dimensionsAttr() {
  return (*this)->removeAttr(getBroadcastDimensionsAttrName());
}

void BroadcastMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
    odsState.addAttribute(getBroadcastDimensionsAttrName(odsState.name), broadcast_dimensions);
  }
  odsState.addTypes(resultType0);
}

void BroadcastMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
    odsState.addAttribute(getBroadcastDimensionsAttrName(odsState.name), broadcast_dimensions);
  }

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(BroadcastMaxOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void BroadcastMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
    odsState.addAttribute(getBroadcastDimensionsAttrName(odsState.name), broadcast_dimensions);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BroadcastMaxOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void BroadcastMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(BroadcastMaxOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult BroadcastMaxOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_broadcast_dimensions;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getBroadcastDimensionsAttrName()) {
      tblgen_broadcast_dimensions = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ChloOps0(*this, tblgen_broadcast_dimensions, "broadcast_dimensions")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult BroadcastMaxOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult BroadcastMaxOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type lhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> lhsTypes(lhsRawTypes);
  ::mlir::Type rhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> rhsTypes(rhsRawTypes);
  ::llvm::SmallVector<::mlir::Type, 1> allResultTypes;

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    lhsRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    rhsRawTypes[0] = type;
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseTypeList(allResultTypes))
    return ::mlir::failure();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(lhsOperands, lhsTypes, lhsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, rhsTypes, rhsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void BroadcastMaxOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getLhs();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getRhs();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ' << "(";
  {
    auto type = getLhs().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getRhs().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ")";
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  _odsPrinter << getOperation()->getResultTypes();
}

void BroadcastMaxOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace chlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::chlo::BroadcastMaxOp)

namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// ::mlir::chlo::BroadcastMinOp definitions
//===----------------------------------------------------------------------===//

BroadcastMinOpAdaptor::BroadcastMinOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("chlo.broadcast_minimum", odsAttrs.getContext());
}

BroadcastMinOpAdaptor::BroadcastMinOpAdaptor(BroadcastMinOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange BroadcastMinOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> BroadcastMinOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BroadcastMinOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BroadcastMinOpAdaptor::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastMinOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastMinOpAdaptor::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value BroadcastMinOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr BroadcastMinOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr BroadcastMinOpAdaptor::getBroadcastDimensionsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, BroadcastMinOp::getBroadcastDimensionsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastMinOpAdaptor::getBroadcastDimensions() {
  auto attr = getBroadcastDimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::DenseIntElementsAttr BroadcastMinOpAdaptor::broadcast_dimensionsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, BroadcastMinOp::getBroadcastDimensionsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastMinOpAdaptor::broadcast_dimensions() {
  auto attr = broadcast_dimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::LogicalResult BroadcastMinOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_broadcast_dimensions;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == BroadcastMinOp::getBroadcastDimensionsAttrName(*odsOpName)) {
      tblgen_broadcast_dimensions = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_broadcast_dimensions && !(((tblgen_broadcast_dimensions.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_broadcast_dimensions.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))))
    return emitError(loc, "'chlo.broadcast_minimum' op ""attribute 'broadcast_dimensions' failed to satisfy constraint: 64-bit signless integer elements attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> BroadcastMinOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BroadcastMinOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> BroadcastMinOp::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> BroadcastMinOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> BroadcastMinOp::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::TypedValue<::mlir::TensorType> BroadcastMinOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange BroadcastMinOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BroadcastMinOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BroadcastMinOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BroadcastMinOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> BroadcastMinOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BroadcastMinOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::DenseIntElementsAttr BroadcastMinOp::getBroadcastDimensionsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getBroadcastDimensionsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastMinOp::getBroadcastDimensions() {
  auto attr = getBroadcastDimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::DenseIntElementsAttr BroadcastMinOp::broadcast_dimensionsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getBroadcastDimensionsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastMinOp::broadcast_dimensions() {
  auto attr = broadcast_dimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

void BroadcastMinOp::setBroadcastDimensionsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getBroadcastDimensionsAttrName(), attr);
}

void BroadcastMinOp::broadcast_dimensionsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(broadcast_dimensionsAttrName(), attr);
}

::mlir::Attribute BroadcastMinOp::removeBroadcast_dimensionsAttr() {
  return (*this)->removeAttr(getBroadcastDimensionsAttrName());
}

void BroadcastMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
    odsState.addAttribute(getBroadcastDimensionsAttrName(odsState.name), broadcast_dimensions);
  }
  odsState.addTypes(resultType0);
}

void BroadcastMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
    odsState.addAttribute(getBroadcastDimensionsAttrName(odsState.name), broadcast_dimensions);
  }

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(BroadcastMinOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void BroadcastMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
    odsState.addAttribute(getBroadcastDimensionsAttrName(odsState.name), broadcast_dimensions);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BroadcastMinOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void BroadcastMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(BroadcastMinOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult BroadcastMinOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_broadcast_dimensions;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getBroadcastDimensionsAttrName()) {
      tblgen_broadcast_dimensions = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ChloOps0(*this, tblgen_broadcast_dimensions, "broadcast_dimensions")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult BroadcastMinOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult BroadcastMinOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type lhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> lhsTypes(lhsRawTypes);
  ::mlir::Type rhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> rhsTypes(rhsRawTypes);
  ::llvm::SmallVector<::mlir::Type, 1> allResultTypes;

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    lhsRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    rhsRawTypes[0] = type;
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseTypeList(allResultTypes))
    return ::mlir::failure();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(lhsOperands, lhsTypes, lhsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, rhsTypes, rhsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void BroadcastMinOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getLhs();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getRhs();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ' << "(";
  {
    auto type = getLhs().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getRhs().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ")";
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  _odsPrinter << getOperation()->getResultTypes();
}

void BroadcastMinOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace chlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::chlo::BroadcastMinOp)

namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// ::mlir::chlo::BroadcastMulOp definitions
//===----------------------------------------------------------------------===//

BroadcastMulOpAdaptor::BroadcastMulOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("chlo.broadcast_multiply", odsAttrs.getContext());
}

BroadcastMulOpAdaptor::BroadcastMulOpAdaptor(BroadcastMulOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange BroadcastMulOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> BroadcastMulOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BroadcastMulOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BroadcastMulOpAdaptor::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastMulOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastMulOpAdaptor::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value BroadcastMulOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr BroadcastMulOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr BroadcastMulOpAdaptor::getBroadcastDimensionsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, BroadcastMulOp::getBroadcastDimensionsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastMulOpAdaptor::getBroadcastDimensions() {
  auto attr = getBroadcastDimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::DenseIntElementsAttr BroadcastMulOpAdaptor::broadcast_dimensionsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, BroadcastMulOp::getBroadcastDimensionsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastMulOpAdaptor::broadcast_dimensions() {
  auto attr = broadcast_dimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::LogicalResult BroadcastMulOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_broadcast_dimensions;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == BroadcastMulOp::getBroadcastDimensionsAttrName(*odsOpName)) {
      tblgen_broadcast_dimensions = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_broadcast_dimensions && !(((tblgen_broadcast_dimensions.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_broadcast_dimensions.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))))
    return emitError(loc, "'chlo.broadcast_multiply' op ""attribute 'broadcast_dimensions' failed to satisfy constraint: 64-bit signless integer elements attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> BroadcastMulOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BroadcastMulOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> BroadcastMulOp::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> BroadcastMulOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> BroadcastMulOp::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::TypedValue<::mlir::TensorType> BroadcastMulOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange BroadcastMulOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BroadcastMulOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BroadcastMulOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BroadcastMulOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> BroadcastMulOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BroadcastMulOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::DenseIntElementsAttr BroadcastMulOp::getBroadcastDimensionsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getBroadcastDimensionsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastMulOp::getBroadcastDimensions() {
  auto attr = getBroadcastDimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::DenseIntElementsAttr BroadcastMulOp::broadcast_dimensionsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getBroadcastDimensionsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastMulOp::broadcast_dimensions() {
  auto attr = broadcast_dimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

void BroadcastMulOp::setBroadcastDimensionsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getBroadcastDimensionsAttrName(), attr);
}

void BroadcastMulOp::broadcast_dimensionsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(broadcast_dimensionsAttrName(), attr);
}

::mlir::Attribute BroadcastMulOp::removeBroadcast_dimensionsAttr() {
  return (*this)->removeAttr(getBroadcastDimensionsAttrName());
}

void BroadcastMulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
    odsState.addAttribute(getBroadcastDimensionsAttrName(odsState.name), broadcast_dimensions);
  }
  odsState.addTypes(resultType0);
}

void BroadcastMulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
    odsState.addAttribute(getBroadcastDimensionsAttrName(odsState.name), broadcast_dimensions);
  }

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(BroadcastMulOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void BroadcastMulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
    odsState.addAttribute(getBroadcastDimensionsAttrName(odsState.name), broadcast_dimensions);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BroadcastMulOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void BroadcastMulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(BroadcastMulOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult BroadcastMulOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_broadcast_dimensions;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getBroadcastDimensionsAttrName()) {
      tblgen_broadcast_dimensions = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ChloOps0(*this, tblgen_broadcast_dimensions, "broadcast_dimensions")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult BroadcastMulOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult BroadcastMulOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type lhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> lhsTypes(lhsRawTypes);
  ::mlir::Type rhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> rhsTypes(rhsRawTypes);
  ::llvm::SmallVector<::mlir::Type, 1> allResultTypes;

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    lhsRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    rhsRawTypes[0] = type;
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseTypeList(allResultTypes))
    return ::mlir::failure();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(lhsOperands, lhsTypes, lhsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, rhsTypes, rhsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void BroadcastMulOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getLhs();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getRhs();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ' << "(";
  {
    auto type = getLhs().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getRhs().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ")";
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  _odsPrinter << getOperation()->getResultTypes();
}

void BroadcastMulOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace chlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::chlo::BroadcastMulOp)

namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// ::mlir::chlo::BroadcastNextAfterOp definitions
//===----------------------------------------------------------------------===//

BroadcastNextAfterOpAdaptor::BroadcastNextAfterOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("chlo.broadcast_next_after", odsAttrs.getContext());
}

BroadcastNextAfterOpAdaptor::BroadcastNextAfterOpAdaptor(BroadcastNextAfterOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange BroadcastNextAfterOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> BroadcastNextAfterOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BroadcastNextAfterOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BroadcastNextAfterOpAdaptor::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastNextAfterOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastNextAfterOpAdaptor::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value BroadcastNextAfterOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr BroadcastNextAfterOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr BroadcastNextAfterOpAdaptor::getBroadcastDimensionsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, BroadcastNextAfterOp::getBroadcastDimensionsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastNextAfterOpAdaptor::getBroadcastDimensions() {
  auto attr = getBroadcastDimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::DenseIntElementsAttr BroadcastNextAfterOpAdaptor::broadcast_dimensionsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, BroadcastNextAfterOp::getBroadcastDimensionsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastNextAfterOpAdaptor::broadcast_dimensions() {
  auto attr = broadcast_dimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::LogicalResult BroadcastNextAfterOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_broadcast_dimensions;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == BroadcastNextAfterOp::getBroadcastDimensionsAttrName(*odsOpName)) {
      tblgen_broadcast_dimensions = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_broadcast_dimensions && !(((tblgen_broadcast_dimensions.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_broadcast_dimensions.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))))
    return emitError(loc, "'chlo.broadcast_next_after' op ""attribute 'broadcast_dimensions' failed to satisfy constraint: 64-bit signless integer elements attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> BroadcastNextAfterOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BroadcastNextAfterOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> BroadcastNextAfterOp::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> BroadcastNextAfterOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> BroadcastNextAfterOp::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::TypedValue<::mlir::TensorType> BroadcastNextAfterOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange BroadcastNextAfterOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BroadcastNextAfterOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BroadcastNextAfterOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BroadcastNextAfterOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> BroadcastNextAfterOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BroadcastNextAfterOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::DenseIntElementsAttr BroadcastNextAfterOp::getBroadcastDimensionsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getBroadcastDimensionsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastNextAfterOp::getBroadcastDimensions() {
  auto attr = getBroadcastDimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::DenseIntElementsAttr BroadcastNextAfterOp::broadcast_dimensionsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getBroadcastDimensionsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastNextAfterOp::broadcast_dimensions() {
  auto attr = broadcast_dimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

void BroadcastNextAfterOp::setBroadcastDimensionsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getBroadcastDimensionsAttrName(), attr);
}

void BroadcastNextAfterOp::broadcast_dimensionsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(broadcast_dimensionsAttrName(), attr);
}

::mlir::Attribute BroadcastNextAfterOp::removeBroadcast_dimensionsAttr() {
  return (*this)->removeAttr(getBroadcastDimensionsAttrName());
}

void BroadcastNextAfterOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
    odsState.addAttribute(getBroadcastDimensionsAttrName(odsState.name), broadcast_dimensions);
  }
  odsState.addTypes(resultType0);
}

void BroadcastNextAfterOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
    odsState.addAttribute(getBroadcastDimensionsAttrName(odsState.name), broadcast_dimensions);
  }

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(BroadcastNextAfterOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void BroadcastNextAfterOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
    odsState.addAttribute(getBroadcastDimensionsAttrName(odsState.name), broadcast_dimensions);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BroadcastNextAfterOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void BroadcastNextAfterOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(BroadcastNextAfterOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult BroadcastNextAfterOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_broadcast_dimensions;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getBroadcastDimensionsAttrName()) {
      tblgen_broadcast_dimensions = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ChloOps0(*this, tblgen_broadcast_dimensions, "broadcast_dimensions")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult BroadcastNextAfterOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult BroadcastNextAfterOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type lhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> lhsTypes(lhsRawTypes);
  ::mlir::Type rhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> rhsTypes(rhsRawTypes);
  ::llvm::SmallVector<::mlir::Type, 1> allResultTypes;

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    lhsRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    rhsRawTypes[0] = type;
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseTypeList(allResultTypes))
    return ::mlir::failure();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(lhsOperands, lhsTypes, lhsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, rhsTypes, rhsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void BroadcastNextAfterOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getLhs();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getRhs();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ' << "(";
  {
    auto type = getLhs().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getRhs().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ")";
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  _odsPrinter << getOperation()->getResultTypes();
}

void BroadcastNextAfterOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace chlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::chlo::BroadcastNextAfterOp)

namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// ::mlir::chlo::BroadcastOrOp definitions
//===----------------------------------------------------------------------===//

BroadcastOrOpAdaptor::BroadcastOrOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("chlo.broadcast_or", odsAttrs.getContext());
}

BroadcastOrOpAdaptor::BroadcastOrOpAdaptor(BroadcastOrOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange BroadcastOrOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> BroadcastOrOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BroadcastOrOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BroadcastOrOpAdaptor::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastOrOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastOrOpAdaptor::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value BroadcastOrOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr BroadcastOrOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr BroadcastOrOpAdaptor::getBroadcastDimensionsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, BroadcastOrOp::getBroadcastDimensionsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastOrOpAdaptor::getBroadcastDimensions() {
  auto attr = getBroadcastDimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::DenseIntElementsAttr BroadcastOrOpAdaptor::broadcast_dimensionsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, BroadcastOrOp::getBroadcastDimensionsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastOrOpAdaptor::broadcast_dimensions() {
  auto attr = broadcast_dimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::LogicalResult BroadcastOrOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_broadcast_dimensions;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == BroadcastOrOp::getBroadcastDimensionsAttrName(*odsOpName)) {
      tblgen_broadcast_dimensions = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_broadcast_dimensions && !(((tblgen_broadcast_dimensions.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_broadcast_dimensions.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))))
    return emitError(loc, "'chlo.broadcast_or' op ""attribute 'broadcast_dimensions' failed to satisfy constraint: 64-bit signless integer elements attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> BroadcastOrOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BroadcastOrOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> BroadcastOrOp::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> BroadcastOrOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> BroadcastOrOp::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::TypedValue<::mlir::TensorType> BroadcastOrOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange BroadcastOrOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BroadcastOrOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BroadcastOrOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BroadcastOrOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> BroadcastOrOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BroadcastOrOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::DenseIntElementsAttr BroadcastOrOp::getBroadcastDimensionsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getBroadcastDimensionsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastOrOp::getBroadcastDimensions() {
  auto attr = getBroadcastDimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::DenseIntElementsAttr BroadcastOrOp::broadcast_dimensionsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getBroadcastDimensionsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastOrOp::broadcast_dimensions() {
  auto attr = broadcast_dimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

void BroadcastOrOp::setBroadcastDimensionsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getBroadcastDimensionsAttrName(), attr);
}

void BroadcastOrOp::broadcast_dimensionsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(broadcast_dimensionsAttrName(), attr);
}

::mlir::Attribute BroadcastOrOp::removeBroadcast_dimensionsAttr() {
  return (*this)->removeAttr(getBroadcastDimensionsAttrName());
}

void BroadcastOrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
    odsState.addAttribute(getBroadcastDimensionsAttrName(odsState.name), broadcast_dimensions);
  }
  odsState.addTypes(resultType0);
}

void BroadcastOrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
    odsState.addAttribute(getBroadcastDimensionsAttrName(odsState.name), broadcast_dimensions);
  }

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(BroadcastOrOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void BroadcastOrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
    odsState.addAttribute(getBroadcastDimensionsAttrName(odsState.name), broadcast_dimensions);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BroadcastOrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void BroadcastOrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(BroadcastOrOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult BroadcastOrOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_broadcast_dimensions;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getBroadcastDimensionsAttrName()) {
      tblgen_broadcast_dimensions = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ChloOps0(*this, tblgen_broadcast_dimensions, "broadcast_dimensions")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult BroadcastOrOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult BroadcastOrOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type lhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> lhsTypes(lhsRawTypes);
  ::mlir::Type rhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> rhsTypes(rhsRawTypes);
  ::llvm::SmallVector<::mlir::Type, 1> allResultTypes;

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    lhsRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    rhsRawTypes[0] = type;
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseTypeList(allResultTypes))
    return ::mlir::failure();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(lhsOperands, lhsTypes, lhsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, rhsTypes, rhsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void BroadcastOrOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getLhs();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getRhs();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ' << "(";
  {
    auto type = getLhs().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getRhs().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ")";
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  _odsPrinter << getOperation()->getResultTypes();
}

void BroadcastOrOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace chlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::chlo::BroadcastOrOp)

namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// ::mlir::chlo::BroadcastPolygammaOp definitions
//===----------------------------------------------------------------------===//

BroadcastPolygammaOpAdaptor::BroadcastPolygammaOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("chlo.broadcast_polygamma", odsAttrs.getContext());
}

BroadcastPolygammaOpAdaptor::BroadcastPolygammaOpAdaptor(BroadcastPolygammaOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange BroadcastPolygammaOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> BroadcastPolygammaOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BroadcastPolygammaOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BroadcastPolygammaOpAdaptor::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastPolygammaOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastPolygammaOpAdaptor::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value BroadcastPolygammaOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr BroadcastPolygammaOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr BroadcastPolygammaOpAdaptor::getBroadcastDimensionsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, BroadcastPolygammaOp::getBroadcastDimensionsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastPolygammaOpAdaptor::getBroadcastDimensions() {
  auto attr = getBroadcastDimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::DenseIntElementsAttr BroadcastPolygammaOpAdaptor::broadcast_dimensionsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, BroadcastPolygammaOp::getBroadcastDimensionsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastPolygammaOpAdaptor::broadcast_dimensions() {
  auto attr = broadcast_dimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::LogicalResult BroadcastPolygammaOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_broadcast_dimensions;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == BroadcastPolygammaOp::getBroadcastDimensionsAttrName(*odsOpName)) {
      tblgen_broadcast_dimensions = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_broadcast_dimensions && !(((tblgen_broadcast_dimensions.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_broadcast_dimensions.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))))
    return emitError(loc, "'chlo.broadcast_polygamma' op ""attribute 'broadcast_dimensions' failed to satisfy constraint: 64-bit signless integer elements attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> BroadcastPolygammaOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BroadcastPolygammaOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> BroadcastPolygammaOp::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> BroadcastPolygammaOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> BroadcastPolygammaOp::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::TypedValue<::mlir::TensorType> BroadcastPolygammaOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange BroadcastPolygammaOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BroadcastPolygammaOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BroadcastPolygammaOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BroadcastPolygammaOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> BroadcastPolygammaOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BroadcastPolygammaOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::DenseIntElementsAttr BroadcastPolygammaOp::getBroadcastDimensionsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getBroadcastDimensionsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastPolygammaOp::getBroadcastDimensions() {
  auto attr = getBroadcastDimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::DenseIntElementsAttr BroadcastPolygammaOp::broadcast_dimensionsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getBroadcastDimensionsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastPolygammaOp::broadcast_dimensions() {
  auto attr = broadcast_dimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

void BroadcastPolygammaOp::setBroadcastDimensionsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getBroadcastDimensionsAttrName(), attr);
}

void BroadcastPolygammaOp::broadcast_dimensionsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(broadcast_dimensionsAttrName(), attr);
}

::mlir::Attribute BroadcastPolygammaOp::removeBroadcast_dimensionsAttr() {
  return (*this)->removeAttr(getBroadcastDimensionsAttrName());
}

void BroadcastPolygammaOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
    odsState.addAttribute(getBroadcastDimensionsAttrName(odsState.name), broadcast_dimensions);
  }
  odsState.addTypes(resultType0);
}

void BroadcastPolygammaOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
    odsState.addAttribute(getBroadcastDimensionsAttrName(odsState.name), broadcast_dimensions);
  }

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(BroadcastPolygammaOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void BroadcastPolygammaOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
    odsState.addAttribute(getBroadcastDimensionsAttrName(odsState.name), broadcast_dimensions);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BroadcastPolygammaOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void BroadcastPolygammaOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(BroadcastPolygammaOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult BroadcastPolygammaOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_broadcast_dimensions;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getBroadcastDimensionsAttrName()) {
      tblgen_broadcast_dimensions = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ChloOps0(*this, tblgen_broadcast_dimensions, "broadcast_dimensions")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult BroadcastPolygammaOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult BroadcastPolygammaOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type lhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> lhsTypes(lhsRawTypes);
  ::mlir::Type rhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> rhsTypes(rhsRawTypes);
  ::llvm::SmallVector<::mlir::Type, 1> allResultTypes;

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    lhsRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    rhsRawTypes[0] = type;
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseTypeList(allResultTypes))
    return ::mlir::failure();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(lhsOperands, lhsTypes, lhsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, rhsTypes, rhsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void BroadcastPolygammaOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getLhs();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getRhs();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ' << "(";
  {
    auto type = getLhs().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getRhs().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ")";
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  _odsPrinter << getOperation()->getResultTypes();
}

void BroadcastPolygammaOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace chlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::chlo::BroadcastPolygammaOp)

namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// ::mlir::chlo::BroadcastPowOp definitions
//===----------------------------------------------------------------------===//

BroadcastPowOpAdaptor::BroadcastPowOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("chlo.broadcast_power", odsAttrs.getContext());
}

BroadcastPowOpAdaptor::BroadcastPowOpAdaptor(BroadcastPowOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange BroadcastPowOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> BroadcastPowOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BroadcastPowOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BroadcastPowOpAdaptor::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastPowOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastPowOpAdaptor::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value BroadcastPowOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr BroadcastPowOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr BroadcastPowOpAdaptor::getBroadcastDimensionsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, BroadcastPowOp::getBroadcastDimensionsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastPowOpAdaptor::getBroadcastDimensions() {
  auto attr = getBroadcastDimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::DenseIntElementsAttr BroadcastPowOpAdaptor::broadcast_dimensionsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, BroadcastPowOp::getBroadcastDimensionsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastPowOpAdaptor::broadcast_dimensions() {
  auto attr = broadcast_dimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::LogicalResult BroadcastPowOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_broadcast_dimensions;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == BroadcastPowOp::getBroadcastDimensionsAttrName(*odsOpName)) {
      tblgen_broadcast_dimensions = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_broadcast_dimensions && !(((tblgen_broadcast_dimensions.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_broadcast_dimensions.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))))
    return emitError(loc, "'chlo.broadcast_power' op ""attribute 'broadcast_dimensions' failed to satisfy constraint: 64-bit signless integer elements attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> BroadcastPowOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BroadcastPowOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> BroadcastPowOp::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> BroadcastPowOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> BroadcastPowOp::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::TypedValue<::mlir::TensorType> BroadcastPowOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange BroadcastPowOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BroadcastPowOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BroadcastPowOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BroadcastPowOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> BroadcastPowOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BroadcastPowOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::DenseIntElementsAttr BroadcastPowOp::getBroadcastDimensionsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getBroadcastDimensionsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastPowOp::getBroadcastDimensions() {
  auto attr = getBroadcastDimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::DenseIntElementsAttr BroadcastPowOp::broadcast_dimensionsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getBroadcastDimensionsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastPowOp::broadcast_dimensions() {
  auto attr = broadcast_dimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

void BroadcastPowOp::setBroadcastDimensionsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getBroadcastDimensionsAttrName(), attr);
}

void BroadcastPowOp::broadcast_dimensionsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(broadcast_dimensionsAttrName(), attr);
}

::mlir::Attribute BroadcastPowOp::removeBroadcast_dimensionsAttr() {
  return (*this)->removeAttr(getBroadcastDimensionsAttrName());
}

void BroadcastPowOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
    odsState.addAttribute(getBroadcastDimensionsAttrName(odsState.name), broadcast_dimensions);
  }
  odsState.addTypes(resultType0);
}

void BroadcastPowOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
    odsState.addAttribute(getBroadcastDimensionsAttrName(odsState.name), broadcast_dimensions);
  }

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(BroadcastPowOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void BroadcastPowOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
    odsState.addAttribute(getBroadcastDimensionsAttrName(odsState.name), broadcast_dimensions);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BroadcastPowOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void BroadcastPowOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(BroadcastPowOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult BroadcastPowOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_broadcast_dimensions;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getBroadcastDimensionsAttrName()) {
      tblgen_broadcast_dimensions = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ChloOps0(*this, tblgen_broadcast_dimensions, "broadcast_dimensions")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult BroadcastPowOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult BroadcastPowOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type lhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> lhsTypes(lhsRawTypes);
  ::mlir::Type rhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> rhsTypes(rhsRawTypes);
  ::llvm::SmallVector<::mlir::Type, 1> allResultTypes;

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    lhsRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    rhsRawTypes[0] = type;
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseTypeList(allResultTypes))
    return ::mlir::failure();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(lhsOperands, lhsTypes, lhsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, rhsTypes, rhsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void BroadcastPowOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getLhs();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getRhs();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ' << "(";
  {
    auto type = getLhs().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getRhs().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ")";
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  _odsPrinter << getOperation()->getResultTypes();
}

void BroadcastPowOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace chlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::chlo::BroadcastPowOp)

namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// ::mlir::chlo::BroadcastRemOp definitions
//===----------------------------------------------------------------------===//

BroadcastRemOpAdaptor::BroadcastRemOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("chlo.broadcast_remainder", odsAttrs.getContext());
}

BroadcastRemOpAdaptor::BroadcastRemOpAdaptor(BroadcastRemOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange BroadcastRemOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> BroadcastRemOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BroadcastRemOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BroadcastRemOpAdaptor::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastRemOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastRemOpAdaptor::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value BroadcastRemOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr BroadcastRemOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr BroadcastRemOpAdaptor::getBroadcastDimensionsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, BroadcastRemOp::getBroadcastDimensionsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastRemOpAdaptor::getBroadcastDimensions() {
  auto attr = getBroadcastDimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::DenseIntElementsAttr BroadcastRemOpAdaptor::broadcast_dimensionsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, BroadcastRemOp::getBroadcastDimensionsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastRemOpAdaptor::broadcast_dimensions() {
  auto attr = broadcast_dimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::LogicalResult BroadcastRemOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_broadcast_dimensions;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == BroadcastRemOp::getBroadcastDimensionsAttrName(*odsOpName)) {
      tblgen_broadcast_dimensions = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_broadcast_dimensions && !(((tblgen_broadcast_dimensions.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_broadcast_dimensions.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))))
    return emitError(loc, "'chlo.broadcast_remainder' op ""attribute 'broadcast_dimensions' failed to satisfy constraint: 64-bit signless integer elements attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> BroadcastRemOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BroadcastRemOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> BroadcastRemOp::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> BroadcastRemOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> BroadcastRemOp::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::TypedValue<::mlir::TensorType> BroadcastRemOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange BroadcastRemOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BroadcastRemOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BroadcastRemOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BroadcastRemOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> BroadcastRemOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BroadcastRemOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::DenseIntElementsAttr BroadcastRemOp::getBroadcastDimensionsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getBroadcastDimensionsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastRemOp::getBroadcastDimensions() {
  auto attr = getBroadcastDimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::DenseIntElementsAttr BroadcastRemOp::broadcast_dimensionsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getBroadcastDimensionsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastRemOp::broadcast_dimensions() {
  auto attr = broadcast_dimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

void BroadcastRemOp::setBroadcastDimensionsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getBroadcastDimensionsAttrName(), attr);
}

void BroadcastRemOp::broadcast_dimensionsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(broadcast_dimensionsAttrName(), attr);
}

::mlir::Attribute BroadcastRemOp::removeBroadcast_dimensionsAttr() {
  return (*this)->removeAttr(getBroadcastDimensionsAttrName());
}

void BroadcastRemOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
    odsState.addAttribute(getBroadcastDimensionsAttrName(odsState.name), broadcast_dimensions);
  }
  odsState.addTypes(resultType0);
}

void BroadcastRemOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
    odsState.addAttribute(getBroadcastDimensionsAttrName(odsState.name), broadcast_dimensions);
  }

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(BroadcastRemOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void BroadcastRemOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
    odsState.addAttribute(getBroadcastDimensionsAttrName(odsState.name), broadcast_dimensions);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BroadcastRemOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void BroadcastRemOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(BroadcastRemOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult BroadcastRemOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_broadcast_dimensions;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getBroadcastDimensionsAttrName()) {
      tblgen_broadcast_dimensions = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ChloOps0(*this, tblgen_broadcast_dimensions, "broadcast_dimensions")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult BroadcastRemOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult BroadcastRemOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type lhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> lhsTypes(lhsRawTypes);
  ::mlir::Type rhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> rhsTypes(rhsRawTypes);
  ::llvm::SmallVector<::mlir::Type, 1> allResultTypes;

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    lhsRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    rhsRawTypes[0] = type;
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseTypeList(allResultTypes))
    return ::mlir::failure();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(lhsOperands, lhsTypes, lhsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, rhsTypes, rhsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void BroadcastRemOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getLhs();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getRhs();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ' << "(";
  {
    auto type = getLhs().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getRhs().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ")";
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  _odsPrinter << getOperation()->getResultTypes();
}

void BroadcastRemOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace chlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::chlo::BroadcastRemOp)

namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// ::mlir::chlo::BroadcastSelectOp definitions
//===----------------------------------------------------------------------===//

BroadcastSelectOpAdaptor::BroadcastSelectOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("chlo.broadcast_select", odsAttrs.getContext());
}

BroadcastSelectOpAdaptor::BroadcastSelectOpAdaptor(BroadcastSelectOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange BroadcastSelectOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> BroadcastSelectOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BroadcastSelectOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BroadcastSelectOpAdaptor::getPred() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastSelectOpAdaptor::pred() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastSelectOpAdaptor::getOnTrue() {
  return *getODSOperands(1).begin();
}

::mlir::Value BroadcastSelectOpAdaptor::on_true() {
  return *getODSOperands(1).begin();
}

::mlir::Value BroadcastSelectOpAdaptor::getOnFalse() {
  return *getODSOperands(2).begin();
}

::mlir::Value BroadcastSelectOpAdaptor::on_false() {
  return *getODSOperands(2).begin();
}

::mlir::DictionaryAttr BroadcastSelectOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult BroadcastSelectOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> BroadcastSelectOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BroadcastSelectOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> BroadcastSelectOp::getPred() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> BroadcastSelectOp::pred() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> BroadcastSelectOp::getOnTrue() {
  return *getODSOperands(1).begin();
}

::mlir::TypedValue<::mlir::TensorType> BroadcastSelectOp::on_true() {
  return *getODSOperands(1).begin();
}

::mlir::TypedValue<::mlir::TensorType> BroadcastSelectOp::getOnFalse() {
  return *getODSOperands(2).begin();
}

::mlir::TypedValue<::mlir::TensorType> BroadcastSelectOp::on_false() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange BroadcastSelectOp::getPredMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BroadcastSelectOp::predMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BroadcastSelectOp::getOnTrueMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BroadcastSelectOp::on_trueMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BroadcastSelectOp::getOnFalseMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BroadcastSelectOp::on_falseMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> BroadcastSelectOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BroadcastSelectOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void BroadcastSelectOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value pred, ::mlir::Value on_true, ::mlir::Value on_false) {
  odsState.addOperands(pred);
  odsState.addOperands(on_true);
  odsState.addOperands(on_false);
  odsState.addTypes(resultType0);
}

void BroadcastSelectOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value pred, ::mlir::Value on_true, ::mlir::Value on_false) {
  odsState.addOperands(pred);
  odsState.addOperands(on_true);
  odsState.addOperands(on_false);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(BroadcastSelectOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void BroadcastSelectOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value pred, ::mlir::Value on_true, ::mlir::Value on_false) {
  odsState.addOperands(pred);
  odsState.addOperands(on_true);
  odsState.addOperands(on_false);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BroadcastSelectOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void BroadcastSelectOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(BroadcastSelectOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult BroadcastSelectOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult BroadcastSelectOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult BroadcastSelectOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand predRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> predOperands(predRawOperands);  ::llvm::SMLoc predOperandsLoc;
  (void)predOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand on_trueRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> on_trueOperands(on_trueRawOperands);  ::llvm::SMLoc on_trueOperandsLoc;
  (void)on_trueOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand on_falseRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> on_falseOperands(on_falseRawOperands);  ::llvm::SMLoc on_falseOperandsLoc;
  (void)on_falseOperandsLoc;
  ::mlir::Type predRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> predTypes(predRawTypes);
  ::mlir::Type on_trueRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> on_trueTypes(on_trueRawTypes);
  ::mlir::Type on_falseRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> on_falseTypes(on_falseRawTypes);
  ::llvm::SmallVector<::mlir::Type, 1> allResultTypes;

  predOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(predRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  on_trueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(on_trueRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  on_falseOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(on_falseRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    predRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    on_trueRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    on_falseRawTypes[0] = type;
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseTypeList(allResultTypes))
    return ::mlir::failure();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(predOperands, predTypes, predOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(on_trueOperands, on_trueTypes, on_trueOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(on_falseOperands, on_falseTypes, on_falseOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void BroadcastSelectOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getPred();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getOnTrue();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getOnFalse();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ' << "(";
  {
    auto type = getPred().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getOnTrue().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getOnFalse().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ")";
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  _odsPrinter << getOperation()->getResultTypes();
}

void BroadcastSelectOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace chlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::chlo::BroadcastSelectOp)

namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// ::mlir::chlo::BroadcastShiftLeftOp definitions
//===----------------------------------------------------------------------===//

BroadcastShiftLeftOpAdaptor::BroadcastShiftLeftOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("chlo.broadcast_shift_left", odsAttrs.getContext());
}

BroadcastShiftLeftOpAdaptor::BroadcastShiftLeftOpAdaptor(BroadcastShiftLeftOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange BroadcastShiftLeftOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> BroadcastShiftLeftOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BroadcastShiftLeftOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BroadcastShiftLeftOpAdaptor::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastShiftLeftOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastShiftLeftOpAdaptor::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value BroadcastShiftLeftOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr BroadcastShiftLeftOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr BroadcastShiftLeftOpAdaptor::getBroadcastDimensionsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, BroadcastShiftLeftOp::getBroadcastDimensionsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastShiftLeftOpAdaptor::getBroadcastDimensions() {
  auto attr = getBroadcastDimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::DenseIntElementsAttr BroadcastShiftLeftOpAdaptor::broadcast_dimensionsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, BroadcastShiftLeftOp::getBroadcastDimensionsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastShiftLeftOpAdaptor::broadcast_dimensions() {
  auto attr = broadcast_dimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::LogicalResult BroadcastShiftLeftOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_broadcast_dimensions;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == BroadcastShiftLeftOp::getBroadcastDimensionsAttrName(*odsOpName)) {
      tblgen_broadcast_dimensions = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_broadcast_dimensions && !(((tblgen_broadcast_dimensions.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_broadcast_dimensions.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))))
    return emitError(loc, "'chlo.broadcast_shift_left' op ""attribute 'broadcast_dimensions' failed to satisfy constraint: 64-bit signless integer elements attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> BroadcastShiftLeftOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BroadcastShiftLeftOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> BroadcastShiftLeftOp::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> BroadcastShiftLeftOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> BroadcastShiftLeftOp::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::TypedValue<::mlir::TensorType> BroadcastShiftLeftOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange BroadcastShiftLeftOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BroadcastShiftLeftOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BroadcastShiftLeftOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BroadcastShiftLeftOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> BroadcastShiftLeftOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BroadcastShiftLeftOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::DenseIntElementsAttr BroadcastShiftLeftOp::getBroadcastDimensionsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getBroadcastDimensionsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastShiftLeftOp::getBroadcastDimensions() {
  auto attr = getBroadcastDimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::DenseIntElementsAttr BroadcastShiftLeftOp::broadcast_dimensionsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getBroadcastDimensionsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastShiftLeftOp::broadcast_dimensions() {
  auto attr = broadcast_dimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

void BroadcastShiftLeftOp::setBroadcastDimensionsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getBroadcastDimensionsAttrName(), attr);
}

void BroadcastShiftLeftOp::broadcast_dimensionsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(broadcast_dimensionsAttrName(), attr);
}

::mlir::Attribute BroadcastShiftLeftOp::removeBroadcast_dimensionsAttr() {
  return (*this)->removeAttr(getBroadcastDimensionsAttrName());
}

void BroadcastShiftLeftOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
    odsState.addAttribute(getBroadcastDimensionsAttrName(odsState.name), broadcast_dimensions);
  }
  odsState.addTypes(resultType0);
}

void BroadcastShiftLeftOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
    odsState.addAttribute(getBroadcastDimensionsAttrName(odsState.name), broadcast_dimensions);
  }

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(BroadcastShiftLeftOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void BroadcastShiftLeftOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
    odsState.addAttribute(getBroadcastDimensionsAttrName(odsState.name), broadcast_dimensions);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BroadcastShiftLeftOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void BroadcastShiftLeftOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(BroadcastShiftLeftOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult BroadcastShiftLeftOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_broadcast_dimensions;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getBroadcastDimensionsAttrName()) {
      tblgen_broadcast_dimensions = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ChloOps0(*this, tblgen_broadcast_dimensions, "broadcast_dimensions")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult BroadcastShiftLeftOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult BroadcastShiftLeftOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type lhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> lhsTypes(lhsRawTypes);
  ::mlir::Type rhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> rhsTypes(rhsRawTypes);
  ::llvm::SmallVector<::mlir::Type, 1> allResultTypes;

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    lhsRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    rhsRawTypes[0] = type;
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseTypeList(allResultTypes))
    return ::mlir::failure();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(lhsOperands, lhsTypes, lhsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, rhsTypes, rhsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void BroadcastShiftLeftOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getLhs();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getRhs();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ' << "(";
  {
    auto type = getLhs().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getRhs().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ")";
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  _odsPrinter << getOperation()->getResultTypes();
}

void BroadcastShiftLeftOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace chlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::chlo::BroadcastShiftLeftOp)

namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// ::mlir::chlo::BroadcastShiftRightArithmeticOp definitions
//===----------------------------------------------------------------------===//

BroadcastShiftRightArithmeticOpAdaptor::BroadcastShiftRightArithmeticOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("chlo.broadcast_shift_right_arithmetic", odsAttrs.getContext());
}

BroadcastShiftRightArithmeticOpAdaptor::BroadcastShiftRightArithmeticOpAdaptor(BroadcastShiftRightArithmeticOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange BroadcastShiftRightArithmeticOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> BroadcastShiftRightArithmeticOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BroadcastShiftRightArithmeticOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BroadcastShiftRightArithmeticOpAdaptor::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastShiftRightArithmeticOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastShiftRightArithmeticOpAdaptor::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value BroadcastShiftRightArithmeticOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr BroadcastShiftRightArithmeticOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr BroadcastShiftRightArithmeticOpAdaptor::getBroadcastDimensionsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, BroadcastShiftRightArithmeticOp::getBroadcastDimensionsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastShiftRightArithmeticOpAdaptor::getBroadcastDimensions() {
  auto attr = getBroadcastDimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::DenseIntElementsAttr BroadcastShiftRightArithmeticOpAdaptor::broadcast_dimensionsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, BroadcastShiftRightArithmeticOp::getBroadcastDimensionsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastShiftRightArithmeticOpAdaptor::broadcast_dimensions() {
  auto attr = broadcast_dimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::LogicalResult BroadcastShiftRightArithmeticOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_broadcast_dimensions;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == BroadcastShiftRightArithmeticOp::getBroadcastDimensionsAttrName(*odsOpName)) {
      tblgen_broadcast_dimensions = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_broadcast_dimensions && !(((tblgen_broadcast_dimensions.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_broadcast_dimensions.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))))
    return emitError(loc, "'chlo.broadcast_shift_right_arithmetic' op ""attribute 'broadcast_dimensions' failed to satisfy constraint: 64-bit signless integer elements attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> BroadcastShiftRightArithmeticOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BroadcastShiftRightArithmeticOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> BroadcastShiftRightArithmeticOp::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> BroadcastShiftRightArithmeticOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> BroadcastShiftRightArithmeticOp::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::TypedValue<::mlir::TensorType> BroadcastShiftRightArithmeticOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange BroadcastShiftRightArithmeticOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BroadcastShiftRightArithmeticOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BroadcastShiftRightArithmeticOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BroadcastShiftRightArithmeticOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> BroadcastShiftRightArithmeticOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BroadcastShiftRightArithmeticOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::DenseIntElementsAttr BroadcastShiftRightArithmeticOp::getBroadcastDimensionsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getBroadcastDimensionsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastShiftRightArithmeticOp::getBroadcastDimensions() {
  auto attr = getBroadcastDimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::DenseIntElementsAttr BroadcastShiftRightArithmeticOp::broadcast_dimensionsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getBroadcastDimensionsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastShiftRightArithmeticOp::broadcast_dimensions() {
  auto attr = broadcast_dimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

void BroadcastShiftRightArithmeticOp::setBroadcastDimensionsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getBroadcastDimensionsAttrName(), attr);
}

void BroadcastShiftRightArithmeticOp::broadcast_dimensionsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(broadcast_dimensionsAttrName(), attr);
}

::mlir::Attribute BroadcastShiftRightArithmeticOp::removeBroadcast_dimensionsAttr() {
  return (*this)->removeAttr(getBroadcastDimensionsAttrName());
}

void BroadcastShiftRightArithmeticOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
    odsState.addAttribute(getBroadcastDimensionsAttrName(odsState.name), broadcast_dimensions);
  }
  odsState.addTypes(resultType0);
}

void BroadcastShiftRightArithmeticOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
    odsState.addAttribute(getBroadcastDimensionsAttrName(odsState.name), broadcast_dimensions);
  }

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(BroadcastShiftRightArithmeticOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void BroadcastShiftRightArithmeticOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
    odsState.addAttribute(getBroadcastDimensionsAttrName(odsState.name), broadcast_dimensions);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BroadcastShiftRightArithmeticOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void BroadcastShiftRightArithmeticOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(BroadcastShiftRightArithmeticOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult BroadcastShiftRightArithmeticOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_broadcast_dimensions;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getBroadcastDimensionsAttrName()) {
      tblgen_broadcast_dimensions = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ChloOps0(*this, tblgen_broadcast_dimensions, "broadcast_dimensions")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult BroadcastShiftRightArithmeticOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult BroadcastShiftRightArithmeticOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type lhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> lhsTypes(lhsRawTypes);
  ::mlir::Type rhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> rhsTypes(rhsRawTypes);
  ::llvm::SmallVector<::mlir::Type, 1> allResultTypes;

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    lhsRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    rhsRawTypes[0] = type;
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseTypeList(allResultTypes))
    return ::mlir::failure();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(lhsOperands, lhsTypes, lhsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, rhsTypes, rhsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void BroadcastShiftRightArithmeticOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getLhs();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getRhs();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ' << "(";
  {
    auto type = getLhs().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getRhs().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ")";
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  _odsPrinter << getOperation()->getResultTypes();
}

void BroadcastShiftRightArithmeticOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace chlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::chlo::BroadcastShiftRightArithmeticOp)

namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// ::mlir::chlo::BroadcastShiftRightLogicalOp definitions
//===----------------------------------------------------------------------===//

BroadcastShiftRightLogicalOpAdaptor::BroadcastShiftRightLogicalOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("chlo.broadcast_shift_right_logical", odsAttrs.getContext());
}

BroadcastShiftRightLogicalOpAdaptor::BroadcastShiftRightLogicalOpAdaptor(BroadcastShiftRightLogicalOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange BroadcastShiftRightLogicalOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> BroadcastShiftRightLogicalOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BroadcastShiftRightLogicalOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BroadcastShiftRightLogicalOpAdaptor::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastShiftRightLogicalOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastShiftRightLogicalOpAdaptor::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value BroadcastShiftRightLogicalOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr BroadcastShiftRightLogicalOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr BroadcastShiftRightLogicalOpAdaptor::getBroadcastDimensionsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, BroadcastShiftRightLogicalOp::getBroadcastDimensionsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastShiftRightLogicalOpAdaptor::getBroadcastDimensions() {
  auto attr = getBroadcastDimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::DenseIntElementsAttr BroadcastShiftRightLogicalOpAdaptor::broadcast_dimensionsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, BroadcastShiftRightLogicalOp::getBroadcastDimensionsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastShiftRightLogicalOpAdaptor::broadcast_dimensions() {
  auto attr = broadcast_dimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::LogicalResult BroadcastShiftRightLogicalOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_broadcast_dimensions;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == BroadcastShiftRightLogicalOp::getBroadcastDimensionsAttrName(*odsOpName)) {
      tblgen_broadcast_dimensions = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_broadcast_dimensions && !(((tblgen_broadcast_dimensions.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_broadcast_dimensions.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))))
    return emitError(loc, "'chlo.broadcast_shift_right_logical' op ""attribute 'broadcast_dimensions' failed to satisfy constraint: 64-bit signless integer elements attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> BroadcastShiftRightLogicalOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BroadcastShiftRightLogicalOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> BroadcastShiftRightLogicalOp::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> BroadcastShiftRightLogicalOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> BroadcastShiftRightLogicalOp::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::TypedValue<::mlir::TensorType> BroadcastShiftRightLogicalOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange BroadcastShiftRightLogicalOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BroadcastShiftRightLogicalOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BroadcastShiftRightLogicalOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BroadcastShiftRightLogicalOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> BroadcastShiftRightLogicalOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BroadcastShiftRightLogicalOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::DenseIntElementsAttr BroadcastShiftRightLogicalOp::getBroadcastDimensionsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getBroadcastDimensionsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastShiftRightLogicalOp::getBroadcastDimensions() {
  auto attr = getBroadcastDimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::DenseIntElementsAttr BroadcastShiftRightLogicalOp::broadcast_dimensionsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getBroadcastDimensionsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastShiftRightLogicalOp::broadcast_dimensions() {
  auto attr = broadcast_dimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

void BroadcastShiftRightLogicalOp::setBroadcastDimensionsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getBroadcastDimensionsAttrName(), attr);
}

void BroadcastShiftRightLogicalOp::broadcast_dimensionsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(broadcast_dimensionsAttrName(), attr);
}

::mlir::Attribute BroadcastShiftRightLogicalOp::removeBroadcast_dimensionsAttr() {
  return (*this)->removeAttr(getBroadcastDimensionsAttrName());
}

void BroadcastShiftRightLogicalOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
    odsState.addAttribute(getBroadcastDimensionsAttrName(odsState.name), broadcast_dimensions);
  }
  odsState.addTypes(resultType0);
}

void BroadcastShiftRightLogicalOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
    odsState.addAttribute(getBroadcastDimensionsAttrName(odsState.name), broadcast_dimensions);
  }

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(BroadcastShiftRightLogicalOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void BroadcastShiftRightLogicalOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
    odsState.addAttribute(getBroadcastDimensionsAttrName(odsState.name), broadcast_dimensions);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BroadcastShiftRightLogicalOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void BroadcastShiftRightLogicalOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(BroadcastShiftRightLogicalOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult BroadcastShiftRightLogicalOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_broadcast_dimensions;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getBroadcastDimensionsAttrName()) {
      tblgen_broadcast_dimensions = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ChloOps0(*this, tblgen_broadcast_dimensions, "broadcast_dimensions")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult BroadcastShiftRightLogicalOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult BroadcastShiftRightLogicalOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type lhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> lhsTypes(lhsRawTypes);
  ::mlir::Type rhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> rhsTypes(rhsRawTypes);
  ::llvm::SmallVector<::mlir::Type, 1> allResultTypes;

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    lhsRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    rhsRawTypes[0] = type;
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseTypeList(allResultTypes))
    return ::mlir::failure();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(lhsOperands, lhsTypes, lhsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, rhsTypes, rhsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void BroadcastShiftRightLogicalOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getLhs();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getRhs();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ' << "(";
  {
    auto type = getLhs().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getRhs().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ")";
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  _odsPrinter << getOperation()->getResultTypes();
}

void BroadcastShiftRightLogicalOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace chlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::chlo::BroadcastShiftRightLogicalOp)

namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// ::mlir::chlo::BroadcastSubOp definitions
//===----------------------------------------------------------------------===//

BroadcastSubOpAdaptor::BroadcastSubOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("chlo.broadcast_subtract", odsAttrs.getContext());
}

BroadcastSubOpAdaptor::BroadcastSubOpAdaptor(BroadcastSubOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange BroadcastSubOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> BroadcastSubOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BroadcastSubOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BroadcastSubOpAdaptor::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastSubOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastSubOpAdaptor::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value BroadcastSubOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr BroadcastSubOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr BroadcastSubOpAdaptor::getBroadcastDimensionsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, BroadcastSubOp::getBroadcastDimensionsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastSubOpAdaptor::getBroadcastDimensions() {
  auto attr = getBroadcastDimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::DenseIntElementsAttr BroadcastSubOpAdaptor::broadcast_dimensionsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, BroadcastSubOp::getBroadcastDimensionsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastSubOpAdaptor::broadcast_dimensions() {
  auto attr = broadcast_dimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::LogicalResult BroadcastSubOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_broadcast_dimensions;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == BroadcastSubOp::getBroadcastDimensionsAttrName(*odsOpName)) {
      tblgen_broadcast_dimensions = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_broadcast_dimensions && !(((tblgen_broadcast_dimensions.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_broadcast_dimensions.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))))
    return emitError(loc, "'chlo.broadcast_subtract' op ""attribute 'broadcast_dimensions' failed to satisfy constraint: 64-bit signless integer elements attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> BroadcastSubOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BroadcastSubOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> BroadcastSubOp::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> BroadcastSubOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> BroadcastSubOp::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::TypedValue<::mlir::TensorType> BroadcastSubOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange BroadcastSubOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BroadcastSubOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BroadcastSubOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BroadcastSubOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> BroadcastSubOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BroadcastSubOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::DenseIntElementsAttr BroadcastSubOp::getBroadcastDimensionsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getBroadcastDimensionsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastSubOp::getBroadcastDimensions() {
  auto attr = getBroadcastDimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::DenseIntElementsAttr BroadcastSubOp::broadcast_dimensionsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getBroadcastDimensionsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastSubOp::broadcast_dimensions() {
  auto attr = broadcast_dimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

void BroadcastSubOp::setBroadcastDimensionsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getBroadcastDimensionsAttrName(), attr);
}

void BroadcastSubOp::broadcast_dimensionsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(broadcast_dimensionsAttrName(), attr);
}

::mlir::Attribute BroadcastSubOp::removeBroadcast_dimensionsAttr() {
  return (*this)->removeAttr(getBroadcastDimensionsAttrName());
}

void BroadcastSubOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
    odsState.addAttribute(getBroadcastDimensionsAttrName(odsState.name), broadcast_dimensions);
  }
  odsState.addTypes(resultType0);
}

void BroadcastSubOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
    odsState.addAttribute(getBroadcastDimensionsAttrName(odsState.name), broadcast_dimensions);
  }

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(BroadcastSubOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void BroadcastSubOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
    odsState.addAttribute(getBroadcastDimensionsAttrName(odsState.name), broadcast_dimensions);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BroadcastSubOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void BroadcastSubOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(BroadcastSubOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult BroadcastSubOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_broadcast_dimensions;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getBroadcastDimensionsAttrName()) {
      tblgen_broadcast_dimensions = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ChloOps0(*this, tblgen_broadcast_dimensions, "broadcast_dimensions")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult BroadcastSubOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult BroadcastSubOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type lhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> lhsTypes(lhsRawTypes);
  ::mlir::Type rhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> rhsTypes(rhsRawTypes);
  ::llvm::SmallVector<::mlir::Type, 1> allResultTypes;

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    lhsRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    rhsRawTypes[0] = type;
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseTypeList(allResultTypes))
    return ::mlir::failure();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(lhsOperands, lhsTypes, lhsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, rhsTypes, rhsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void BroadcastSubOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getLhs();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getRhs();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ' << "(";
  {
    auto type = getLhs().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getRhs().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ")";
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  _odsPrinter << getOperation()->getResultTypes();
}

void BroadcastSubOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace chlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::chlo::BroadcastSubOp)

namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// ::mlir::chlo::BroadcastXorOp definitions
//===----------------------------------------------------------------------===//

BroadcastXorOpAdaptor::BroadcastXorOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("chlo.broadcast_xor", odsAttrs.getContext());
}

BroadcastXorOpAdaptor::BroadcastXorOpAdaptor(BroadcastXorOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange BroadcastXorOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> BroadcastXorOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BroadcastXorOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BroadcastXorOpAdaptor::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastXorOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastXorOpAdaptor::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value BroadcastXorOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr BroadcastXorOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr BroadcastXorOpAdaptor::getBroadcastDimensionsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, BroadcastXorOp::getBroadcastDimensionsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastXorOpAdaptor::getBroadcastDimensions() {
  auto attr = getBroadcastDimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::DenseIntElementsAttr BroadcastXorOpAdaptor::broadcast_dimensionsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, BroadcastXorOp::getBroadcastDimensionsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastXorOpAdaptor::broadcast_dimensions() {
  auto attr = broadcast_dimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::LogicalResult BroadcastXorOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_broadcast_dimensions;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == BroadcastXorOp::getBroadcastDimensionsAttrName(*odsOpName)) {
      tblgen_broadcast_dimensions = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_broadcast_dimensions && !(((tblgen_broadcast_dimensions.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_broadcast_dimensions.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))))
    return emitError(loc, "'chlo.broadcast_xor' op ""attribute 'broadcast_dimensions' failed to satisfy constraint: 64-bit signless integer elements attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> BroadcastXorOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BroadcastXorOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> BroadcastXorOp::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> BroadcastXorOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> BroadcastXorOp::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::TypedValue<::mlir::TensorType> BroadcastXorOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange BroadcastXorOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BroadcastXorOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BroadcastXorOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BroadcastXorOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> BroadcastXorOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BroadcastXorOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::DenseIntElementsAttr BroadcastXorOp::getBroadcastDimensionsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getBroadcastDimensionsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastXorOp::getBroadcastDimensions() {
  auto attr = getBroadcastDimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::DenseIntElementsAttr BroadcastXorOp::broadcast_dimensionsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getBroadcastDimensionsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastXorOp::broadcast_dimensions() {
  auto attr = broadcast_dimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

void BroadcastXorOp::setBroadcastDimensionsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getBroadcastDimensionsAttrName(), attr);
}

void BroadcastXorOp::broadcast_dimensionsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(broadcast_dimensionsAttrName(), attr);
}

::mlir::Attribute BroadcastXorOp::removeBroadcast_dimensionsAttr() {
  return (*this)->removeAttr(getBroadcastDimensionsAttrName());
}

void BroadcastXorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
    odsState.addAttribute(getBroadcastDimensionsAttrName(odsState.name), broadcast_dimensions);
  }
  odsState.addTypes(resultType0);
}

void BroadcastXorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
    odsState.addAttribute(getBroadcastDimensionsAttrName(odsState.name), broadcast_dimensions);
  }

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(BroadcastXorOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void BroadcastXorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
    odsState.addAttribute(getBroadcastDimensionsAttrName(odsState.name), broadcast_dimensions);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BroadcastXorOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void BroadcastXorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(BroadcastXorOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult BroadcastXorOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_broadcast_dimensions;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getBroadcastDimensionsAttrName()) {
      tblgen_broadcast_dimensions = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ChloOps0(*this, tblgen_broadcast_dimensions, "broadcast_dimensions")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult BroadcastXorOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult BroadcastXorOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type lhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> lhsTypes(lhsRawTypes);
  ::mlir::Type rhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> rhsTypes(rhsRawTypes);
  ::llvm::SmallVector<::mlir::Type, 1> allResultTypes;

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    lhsRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    rhsRawTypes[0] = type;
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseTypeList(allResultTypes))
    return ::mlir::failure();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(lhsOperands, lhsTypes, lhsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, rhsTypes, rhsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void BroadcastXorOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getLhs();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getRhs();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ' << "(";
  {
    auto type = getLhs().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getRhs().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ")";
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  _odsPrinter << getOperation()->getResultTypes();
}

void BroadcastXorOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace chlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::chlo::BroadcastXorOp)

namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// ::mlir::chlo::BroadcastZetaOp definitions
//===----------------------------------------------------------------------===//

BroadcastZetaOpAdaptor::BroadcastZetaOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("chlo.broadcast_zeta", odsAttrs.getContext());
}

BroadcastZetaOpAdaptor::BroadcastZetaOpAdaptor(BroadcastZetaOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange BroadcastZetaOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> BroadcastZetaOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BroadcastZetaOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BroadcastZetaOpAdaptor::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastZetaOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastZetaOpAdaptor::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value BroadcastZetaOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr BroadcastZetaOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr BroadcastZetaOpAdaptor::getBroadcastDimensionsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, BroadcastZetaOp::getBroadcastDimensionsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastZetaOpAdaptor::getBroadcastDimensions() {
  auto attr = getBroadcastDimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::DenseIntElementsAttr BroadcastZetaOpAdaptor::broadcast_dimensionsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, BroadcastZetaOp::getBroadcastDimensionsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastZetaOpAdaptor::broadcast_dimensions() {
  auto attr = broadcast_dimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::LogicalResult BroadcastZetaOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_broadcast_dimensions;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == BroadcastZetaOp::getBroadcastDimensionsAttrName(*odsOpName)) {
      tblgen_broadcast_dimensions = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_broadcast_dimensions && !(((tblgen_broadcast_dimensions.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_broadcast_dimensions.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))))
    return emitError(loc, "'chlo.broadcast_zeta' op ""attribute 'broadcast_dimensions' failed to satisfy constraint: 64-bit signless integer elements attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> BroadcastZetaOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BroadcastZetaOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> BroadcastZetaOp::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> BroadcastZetaOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> BroadcastZetaOp::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::TypedValue<::mlir::TensorType> BroadcastZetaOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange BroadcastZetaOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BroadcastZetaOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BroadcastZetaOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BroadcastZetaOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> BroadcastZetaOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BroadcastZetaOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::DenseIntElementsAttr BroadcastZetaOp::getBroadcastDimensionsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getBroadcastDimensionsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastZetaOp::getBroadcastDimensions() {
  auto attr = getBroadcastDimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::DenseIntElementsAttr BroadcastZetaOp::broadcast_dimensionsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getBroadcastDimensionsAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastZetaOp::broadcast_dimensions() {
  auto attr = broadcast_dimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

void BroadcastZetaOp::setBroadcastDimensionsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getBroadcastDimensionsAttrName(), attr);
}

void BroadcastZetaOp::broadcast_dimensionsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(broadcast_dimensionsAttrName(), attr);
}

::mlir::Attribute BroadcastZetaOp::removeBroadcast_dimensionsAttr() {
  return (*this)->removeAttr(getBroadcastDimensionsAttrName());
}

void BroadcastZetaOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
    odsState.addAttribute(getBroadcastDimensionsAttrName(odsState.name), broadcast_dimensions);
  }
  odsState.addTypes(resultType0);
}

void BroadcastZetaOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
    odsState.addAttribute(getBroadcastDimensionsAttrName(odsState.name), broadcast_dimensions);
  }

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(BroadcastZetaOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void BroadcastZetaOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
    odsState.addAttribute(getBroadcastDimensionsAttrName(odsState.name), broadcast_dimensions);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BroadcastZetaOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void BroadcastZetaOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(BroadcastZetaOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult BroadcastZetaOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_broadcast_dimensions;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getBroadcastDimensionsAttrName()) {
      tblgen_broadcast_dimensions = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ChloOps0(*this, tblgen_broadcast_dimensions, "broadcast_dimensions")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps4(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult BroadcastZetaOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult BroadcastZetaOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type lhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> lhsTypes(lhsRawTypes);
  ::mlir::Type rhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> rhsTypes(rhsRawTypes);
  ::llvm::SmallVector<::mlir::Type, 1> allResultTypes;

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    lhsRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    rhsRawTypes[0] = type;
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseTypeList(allResultTypes))
    return ::mlir::failure();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(lhsOperands, lhsTypes, lhsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, rhsTypes, rhsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void BroadcastZetaOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getLhs();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getRhs();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ' << "(";
  {
    auto type = getLhs().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getRhs().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ")";
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  _odsPrinter << getOperation()->getResultTypes();
}

void BroadcastZetaOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace chlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::chlo::BroadcastZetaOp)

namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// ::mlir::chlo::ConjOp definitions
//===----------------------------------------------------------------------===//

ConjOpAdaptor::ConjOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("chlo.conj", odsAttrs.getContext());
}

ConjOpAdaptor::ConjOpAdaptor(ConjOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange ConjOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ConjOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ConjOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ConjOpAdaptor::getOperand() {
  return *getODSOperands(0).begin();
}

::mlir::Value ConjOpAdaptor::operand() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr ConjOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ConjOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ConjOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ConjOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> ConjOp::getOperand() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> ConjOp::operand() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange ConjOp::getOperandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ConjOp::operandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ConjOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ConjOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> ConjOp::getResult() {
  return *getODSResults(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> ConjOp::result() {
  return *getODSResults(0).begin();
}

void ConjOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes(result);
}

void ConjOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand) {
  odsState.addOperands(operand);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(ConjOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void ConjOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand) {
  odsState.addOperands(operand);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ConjOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void ConjOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(ConjOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult ConjOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ConjOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult ConjOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand operandRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> operandOperands(operandRawOperands);  ::llvm::SMLoc operandOperandsLoc;
  (void)operandOperandsLoc;
  ::mlir::Type operandRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> operandTypes(operandRawTypes);
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);

  operandOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(operandRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    operandRawTypes[0] = type;
  }
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawTypes[0] = type;
  }
  result.addTypes(resultTypes);
  if (parser.resolveOperands(operandOperands, operandTypes, operandOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ConjOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getOperand();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getOperand().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void ConjOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace chlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::chlo::ConjOp)

namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// ::mlir::chlo::ConstantLikeOp definitions
//===----------------------------------------------------------------------===//

ConstantLikeOpAdaptor::ConstantLikeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("chlo.constant_like", odsAttrs.getContext());
}

ConstantLikeOpAdaptor::ConstantLikeOpAdaptor(ConstantLikeOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange ConstantLikeOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ConstantLikeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ConstantLikeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ConstantLikeOpAdaptor::getOperand() {
  return *getODSOperands(0).begin();
}

::mlir::Value ConstantLikeOpAdaptor::operand() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr ConstantLikeOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::TypedAttr ConstantLikeOpAdaptor::getValueAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, ConstantLikeOp::getValueAttrName(*odsOpName)).cast<::mlir::TypedAttr>();
  return attr;
}

::mlir::TypedAttr ConstantLikeOpAdaptor::getValue() {
  auto attr = getValueAttr();
  return attr;
}

::mlir::TypedAttr ConstantLikeOpAdaptor::valueAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, ConstantLikeOp::getValueAttrName(*odsOpName)).cast<::mlir::TypedAttr>();
  return attr;
}

::mlir::TypedAttr ConstantLikeOpAdaptor::value() {
  auto attr = valueAttr();
  return attr;
}

::mlir::LogicalResult ConstantLikeOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_value;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'chlo.constant_like' op ""requires attribute 'value'");
    if (namedAttrIt->getName() == ConstantLikeOp::getValueAttrName(*odsOpName)) {
      tblgen_value = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_value && !((tblgen_value.isa<::mlir::TypedAttr>())))
    return emitError(loc, "'chlo.constant_like' op ""attribute 'value' failed to satisfy constraint: TypedAttr instance");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ConstantLikeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ConstantLikeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> ConstantLikeOp::getOperand() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> ConstantLikeOp::operand() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange ConstantLikeOp::getOperandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ConstantLikeOp::operandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ConstantLikeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ConstantLikeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedAttr ConstantLikeOp::getValueAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getValueAttrName()).cast<::mlir::TypedAttr>();
}

::mlir::TypedAttr ConstantLikeOp::getValue() {
  auto attr = getValueAttr();
  return attr;
}

::mlir::TypedAttr ConstantLikeOp::valueAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getValueAttrName()).cast<::mlir::TypedAttr>();
}

::mlir::TypedAttr ConstantLikeOp::value() {
  auto attr = valueAttr();
  return attr;
}

void ConstantLikeOp::setValueAttr(::mlir::TypedAttr attr) {
  (*this)->setAttr(getValueAttrName(), attr);
}

void ConstantLikeOp::valueAttr(::mlir::TypedAttr attr) {
  (*this)->setAttr(valueAttrName(), attr);
}

void ConstantLikeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::TypedAttr value, ::mlir::Value operand) {
  odsState.addOperands(operand);
  odsState.addAttribute(getValueAttrName(odsState.name), value);
  odsState.addTypes(resultType0);
}

void ConstantLikeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypedAttr value, ::mlir::Value operand) {
  odsState.addOperands(operand);
  odsState.addAttribute(getValueAttrName(odsState.name), value);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(ConstantLikeOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void ConstantLikeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::TypedAttr value, ::mlir::Value operand) {
  odsState.addOperands(operand);
  odsState.addAttribute(getValueAttrName(odsState.name), value);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ConstantLikeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void ConstantLikeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(ConstantLikeOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult ConstantLikeOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_value;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'value'");
    if (namedAttrIt->getName() == getValueAttrName()) {
      tblgen_value = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ChloOps3(*this, tblgen_value, "value")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ConstantLikeOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

void ConstantLikeOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace chlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::chlo::ConstantLikeOp)

namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// ::mlir::chlo::ConstantOp definitions
//===----------------------------------------------------------------------===//

ConstantOpAdaptor::ConstantOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("chlo.constant", odsAttrs.getContext());
}

ConstantOpAdaptor::ConstantOpAdaptor(ConstantOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange ConstantOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ConstantOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ConstantOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr ConstantOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ElementsAttr ConstantOpAdaptor::getValueAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, ConstantOp::getValueAttrName(*odsOpName)).cast<::mlir::ElementsAttr>();
  return attr;
}

::mlir::ElementsAttr ConstantOpAdaptor::getValue() {
  auto attr = getValueAttr();
  return attr;
}

::mlir::ElementsAttr ConstantOpAdaptor::valueAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, ConstantOp::getValueAttrName(*odsOpName)).cast<::mlir::ElementsAttr>();
  return attr;
}

::mlir::ElementsAttr ConstantOpAdaptor::value() {
  auto attr = valueAttr();
  return attr;
}

::mlir::LogicalResult ConstantOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_value;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'chlo.constant' op ""requires attribute 'value'");
    if (namedAttrIt->getName() == ConstantOp::getValueAttrName(*odsOpName)) {
      tblgen_value = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_value && !((tblgen_value.isa<::mlir::ElementsAttr>())))
    return emitError(loc, "'chlo.constant' op ""attribute 'value' failed to satisfy constraint: constant vector/tensor attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ConstantOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ConstantOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> ConstantOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ConstantOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> ConstantOp::getOutput() {
  return *getODSResults(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> ConstantOp::output() {
  return *getODSResults(0).begin();
}

::mlir::ElementsAttr ConstantOp::getValueAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getValueAttrName()).cast<::mlir::ElementsAttr>();
}

::mlir::ElementsAttr ConstantOp::getValue() {
  auto attr = getValueAttr();
  return attr;
}

::mlir::ElementsAttr ConstantOp::valueAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getValueAttrName()).cast<::mlir::ElementsAttr>();
}

::mlir::ElementsAttr ConstantOp::value() {
  auto attr = valueAttr();
  return attr;
}

void ConstantOp::setValueAttr(::mlir::ElementsAttr attr) {
  (*this)->setAttr(getValueAttrName(), attr);
}

void ConstantOp::valueAttr(::mlir::ElementsAttr attr) {
  (*this)->setAttr(valueAttrName(), attr);
}

void ConstantOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::ElementsAttr value) {
  odsState.addAttribute(getValueAttrName(odsState.name), value);
  odsState.addTypes(output);
}

void ConstantOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ElementsAttr value) {
  odsState.addAttribute(getValueAttrName(odsState.name), value);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(ConstantOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void ConstantOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ElementsAttr value) {
  odsState.addAttribute(getValueAttrName(odsState.name), value);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ConstantOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void ConstantOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(ConstantOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult ConstantOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_value;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'value'");
    if (namedAttrIt->getName() == getValueAttrName()) {
      tblgen_value = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ChloOps4(*this, tblgen_value, "value")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps6(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ConstantOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult ConstantOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::ElementsAttr valueAttr;
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(valueAttr, ::mlir::Type{}, "value",
          result.attributes)) {
    return ::mlir::failure();
  }

  ::llvm::SmallVector<::mlir::Type> inferredReturnTypes;
  if (::mlir::failed(ConstantOp::inferReturnTypes(parser.getContext(),
      result.location, result.operands,
      result.attributes.getDictionary(parser.getContext()),
      result.regions, inferredReturnTypes)))
    return ::mlir::failure();
  result.addTypes(inferredReturnTypes);
  return ::mlir::success();
}

void ConstantOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"value"});
  _odsPrinter << ' ';
_odsPrinter.printStrippedAttrOrType(getValueAttr());
}

void ConstantOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace chlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::chlo::ConstantOp)

namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// ::mlir::chlo::CoshOp definitions
//===----------------------------------------------------------------------===//

CoshOpAdaptor::CoshOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("chlo.cosh", odsAttrs.getContext());
}

CoshOpAdaptor::CoshOpAdaptor(CoshOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange CoshOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CoshOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CoshOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CoshOpAdaptor::getOperand() {
  return *getODSOperands(0).begin();
}

::mlir::Value CoshOpAdaptor::operand() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr CoshOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult CoshOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CoshOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CoshOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> CoshOp::getOperand() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> CoshOp::operand() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange CoshOp::getOperandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange CoshOp::operandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> CoshOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CoshOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> CoshOp::getResult() {
  return *getODSResults(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> CoshOp::result() {
  return *getODSResults(0).begin();
}

void CoshOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes(result);
}

void CoshOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand) {
  odsState.addOperands(operand);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(CoshOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void CoshOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand) {
  odsState.addOperands(operand);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CoshOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void CoshOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(CoshOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult CoshOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult CoshOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult CoshOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand operandRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> operandOperands(operandRawOperands);  ::llvm::SMLoc operandOperandsLoc;
  (void)operandOperandsLoc;
  ::mlir::Type operandRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> operandTypes(operandRawTypes);
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);

  operandOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(operandRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    operandRawTypes[0] = type;
  }
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawTypes[0] = type;
  }
  result.addTypes(resultTypes);
  if (parser.resolveOperands(operandOperands, operandTypes, operandOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CoshOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getOperand();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getOperand().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void CoshOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace chlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::chlo::CoshOp)

namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// ::mlir::chlo::DigammaOp definitions
//===----------------------------------------------------------------------===//

DigammaOpAdaptor::DigammaOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("chlo.digamma", odsAttrs.getContext());
}

DigammaOpAdaptor::DigammaOpAdaptor(DigammaOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange DigammaOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> DigammaOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange DigammaOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value DigammaOpAdaptor::getOperand() {
  return *getODSOperands(0).begin();
}

::mlir::Value DigammaOpAdaptor::operand() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr DigammaOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult DigammaOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> DigammaOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range DigammaOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> DigammaOp::getOperand() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> DigammaOp::operand() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange DigammaOp::getOperandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange DigammaOp::operandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> DigammaOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range DigammaOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> DigammaOp::getResult() {
  return *getODSResults(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> DigammaOp::result() {
  return *getODSResults(0).begin();
}

void DigammaOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes(result);
}

void DigammaOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand) {
  odsState.addOperands(operand);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(DigammaOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void DigammaOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand) {
  odsState.addOperands(operand);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void DigammaOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void DigammaOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(DigammaOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult DigammaOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps4(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult DigammaOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult DigammaOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand operandRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> operandOperands(operandRawOperands);  ::llvm::SMLoc operandOperandsLoc;
  (void)operandOperandsLoc;
  ::mlir::Type operandRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> operandTypes(operandRawTypes);
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);

  operandOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(operandRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    operandRawTypes[0] = type;
  }
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawTypes[0] = type;
  }
  result.addTypes(resultTypes);
  if (parser.resolveOperands(operandOperands, operandTypes, operandOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void DigammaOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getOperand();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getOperand().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void DigammaOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace chlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::chlo::DigammaOp)

namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// ::mlir::chlo::DynamicReshapeOp definitions
//===----------------------------------------------------------------------===//

DynamicReshapeOpAdaptor::DynamicReshapeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("chlo.dynamic_reshape", odsAttrs.getContext());
}

DynamicReshapeOpAdaptor::DynamicReshapeOpAdaptor(DynamicReshapeOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange DynamicReshapeOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> DynamicReshapeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange DynamicReshapeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value DynamicReshapeOpAdaptor::getOperand() {
  return *getODSOperands(0).begin();
}

::mlir::Value DynamicReshapeOpAdaptor::operand() {
  return *getODSOperands(0).begin();
}

::mlir::Value DynamicReshapeOpAdaptor::getOutputShape() {
  return *getODSOperands(1).begin();
}

::mlir::Value DynamicReshapeOpAdaptor::output_shape() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr DynamicReshapeOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult DynamicReshapeOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> DynamicReshapeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range DynamicReshapeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> DynamicReshapeOp::getOperand() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> DynamicReshapeOp::operand() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> DynamicReshapeOp::getOutputShape() {
  return *getODSOperands(1).begin();
}

::mlir::TypedValue<::mlir::TensorType> DynamicReshapeOp::output_shape() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange DynamicReshapeOp::getOperandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange DynamicReshapeOp::operandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange DynamicReshapeOp::getOutputShapeMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange DynamicReshapeOp::output_shapeMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> DynamicReshapeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range DynamicReshapeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> DynamicReshapeOp::getResult() {
  return *getODSResults(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> DynamicReshapeOp::result() {
  return *getODSResults(0).begin();
}

void DynamicReshapeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value operand, ::mlir::Value output_shape) {
  odsState.addOperands(operand);
  odsState.addOperands(output_shape);
  odsState.addTypes(result);
}

void DynamicReshapeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand, ::mlir::Value output_shape) {
  odsState.addOperands(operand);
  odsState.addOperands(output_shape);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void DynamicReshapeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult DynamicReshapeOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult DynamicReshapeOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void DynamicReshapeOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace chlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::chlo::DynamicReshapeOp)

namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// ::mlir::chlo::ErfOp definitions
//===----------------------------------------------------------------------===//

ErfOpAdaptor::ErfOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("chlo.erf", odsAttrs.getContext());
}

ErfOpAdaptor::ErfOpAdaptor(ErfOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange ErfOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ErfOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ErfOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ErfOpAdaptor::getOperand() {
  return *getODSOperands(0).begin();
}

::mlir::Value ErfOpAdaptor::operand() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr ErfOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ErfOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ErfOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ErfOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> ErfOp::getOperand() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> ErfOp::operand() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange ErfOp::getOperandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ErfOp::operandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ErfOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ErfOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> ErfOp::getResult() {
  return *getODSResults(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> ErfOp::result() {
  return *getODSResults(0).begin();
}

void ErfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes(result);
}

void ErfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand) {
  odsState.addOperands(operand);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(ErfOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void ErfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand) {
  odsState.addOperands(operand);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ErfOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void ErfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(ErfOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult ErfOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps4(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ErfOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult ErfOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand operandRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> operandOperands(operandRawOperands);  ::llvm::SMLoc operandOperandsLoc;
  (void)operandOperandsLoc;
  ::mlir::Type operandRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> operandTypes(operandRawTypes);
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);

  operandOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(operandRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    operandRawTypes[0] = type;
  }
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawTypes[0] = type;
  }
  result.addTypes(resultTypes);
  if (parser.resolveOperands(operandOperands, operandTypes, operandOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ErfOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getOperand();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getOperand().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void ErfOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace chlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::chlo::ErfOp)

namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// ::mlir::chlo::ErfcOp definitions
//===----------------------------------------------------------------------===//

ErfcOpAdaptor::ErfcOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("chlo.erfc", odsAttrs.getContext());
}

ErfcOpAdaptor::ErfcOpAdaptor(ErfcOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange ErfcOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ErfcOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ErfcOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ErfcOpAdaptor::getOperand() {
  return *getODSOperands(0).begin();
}

::mlir::Value ErfcOpAdaptor::operand() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr ErfcOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ErfcOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ErfcOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ErfcOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> ErfcOp::getOperand() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> ErfcOp::operand() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange ErfcOp::getOperandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ErfcOp::operandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ErfcOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ErfcOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> ErfcOp::getResult() {
  return *getODSResults(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> ErfcOp::result() {
  return *getODSResults(0).begin();
}

void ErfcOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes(result);
}

void ErfcOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand) {
  odsState.addOperands(operand);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(ErfcOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void ErfcOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand) {
  odsState.addOperands(operand);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ErfcOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void ErfcOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(ErfcOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult ErfcOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps4(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ErfcOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult ErfcOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand operandRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> operandOperands(operandRawOperands);  ::llvm::SMLoc operandOperandsLoc;
  (void)operandOperandsLoc;
  ::mlir::Type operandRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> operandTypes(operandRawTypes);
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);

  operandOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(operandRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    operandRawTypes[0] = type;
  }
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawTypes[0] = type;
  }
  result.addTypes(resultTypes);
  if (parser.resolveOperands(operandOperands, operandTypes, operandOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ErfcOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getOperand();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getOperand().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void ErfcOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace chlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::chlo::ErfcOp)

namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// ::mlir::chlo::IsInfOp definitions
//===----------------------------------------------------------------------===//

IsInfOpAdaptor::IsInfOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("chlo.is_inf", odsAttrs.getContext());
}

IsInfOpAdaptor::IsInfOpAdaptor(IsInfOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange IsInfOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> IsInfOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange IsInfOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value IsInfOpAdaptor::getOperand() {
  return *getODSOperands(0).begin();
}

::mlir::Value IsInfOpAdaptor::operand() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr IsInfOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult IsInfOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> IsInfOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range IsInfOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> IsInfOp::getOperand() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> IsInfOp::operand() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange IsInfOp::getOperandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange IsInfOp::operandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> IsInfOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range IsInfOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> IsInfOp::getResult() {
  return *getODSResults(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> IsInfOp::result() {
  return *getODSResults(0).begin();
}

void IsInfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes(result);
}

void IsInfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand) {
  odsState.addOperands(operand);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(IsInfOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void IsInfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand) {
  odsState.addOperands(operand);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void IsInfOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void IsInfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(IsInfOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult IsInfOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult IsInfOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult IsInfOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand operandRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> operandOperands(operandRawOperands);  ::llvm::SMLoc operandOperandsLoc;
  (void)operandOperandsLoc;
  ::mlir::Type operandRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> operandTypes(operandRawTypes);
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);

  operandOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(operandRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    operandRawTypes[0] = type;
  }
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawTypes[0] = type;
  }
  result.addTypes(resultTypes);
  if (parser.resolveOperands(operandOperands, operandTypes, operandOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void IsInfOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getOperand();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getOperand().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void IsInfOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace chlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::chlo::IsInfOp)

namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// ::mlir::chlo::IsNegInfOp definitions
//===----------------------------------------------------------------------===//

IsNegInfOpAdaptor::IsNegInfOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("chlo.is_neg_inf", odsAttrs.getContext());
}

IsNegInfOpAdaptor::IsNegInfOpAdaptor(IsNegInfOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange IsNegInfOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> IsNegInfOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange IsNegInfOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value IsNegInfOpAdaptor::getOperand() {
  return *getODSOperands(0).begin();
}

::mlir::Value IsNegInfOpAdaptor::operand() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr IsNegInfOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult IsNegInfOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> IsNegInfOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range IsNegInfOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> IsNegInfOp::getOperand() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> IsNegInfOp::operand() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange IsNegInfOp::getOperandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange IsNegInfOp::operandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> IsNegInfOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range IsNegInfOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> IsNegInfOp::getResult() {
  return *getODSResults(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> IsNegInfOp::result() {
  return *getODSResults(0).begin();
}

void IsNegInfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes(result);
}

void IsNegInfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand) {
  odsState.addOperands(operand);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(IsNegInfOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void IsNegInfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand) {
  odsState.addOperands(operand);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void IsNegInfOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void IsNegInfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(IsNegInfOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult IsNegInfOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult IsNegInfOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult IsNegInfOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand operandRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> operandOperands(operandRawOperands);  ::llvm::SMLoc operandOperandsLoc;
  (void)operandOperandsLoc;
  ::mlir::Type operandRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> operandTypes(operandRawTypes);
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);

  operandOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(operandRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    operandRawTypes[0] = type;
  }
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawTypes[0] = type;
  }
  result.addTypes(resultTypes);
  if (parser.resolveOperands(operandOperands, operandTypes, operandOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void IsNegInfOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getOperand();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getOperand().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void IsNegInfOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace chlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::chlo::IsNegInfOp)

namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// ::mlir::chlo::IsPosInfOp definitions
//===----------------------------------------------------------------------===//

IsPosInfOpAdaptor::IsPosInfOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("chlo.is_pos_inf", odsAttrs.getContext());
}

IsPosInfOpAdaptor::IsPosInfOpAdaptor(IsPosInfOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange IsPosInfOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> IsPosInfOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange IsPosInfOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value IsPosInfOpAdaptor::getOperand() {
  return *getODSOperands(0).begin();
}

::mlir::Value IsPosInfOpAdaptor::operand() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr IsPosInfOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult IsPosInfOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> IsPosInfOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range IsPosInfOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> IsPosInfOp::getOperand() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> IsPosInfOp::operand() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange IsPosInfOp::getOperandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange IsPosInfOp::operandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> IsPosInfOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range IsPosInfOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> IsPosInfOp::getResult() {
  return *getODSResults(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> IsPosInfOp::result() {
  return *getODSResults(0).begin();
}

void IsPosInfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes(result);
}

void IsPosInfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand) {
  odsState.addOperands(operand);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(IsPosInfOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void IsPosInfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand) {
  odsState.addOperands(operand);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void IsPosInfOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void IsPosInfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(IsPosInfOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult IsPosInfOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult IsPosInfOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult IsPosInfOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand operandRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> operandOperands(operandRawOperands);  ::llvm::SMLoc operandOperandsLoc;
  (void)operandOperandsLoc;
  ::mlir::Type operandRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> operandTypes(operandRawTypes);
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);

  operandOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(operandRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    operandRawTypes[0] = type;
  }
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawTypes[0] = type;
  }
  result.addTypes(resultTypes);
  if (parser.resolveOperands(operandOperands, operandTypes, operandOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void IsPosInfOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getOperand();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getOperand().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void IsPosInfOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace chlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::chlo::IsPosInfOp)

namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// ::mlir::chlo::LgammaOp definitions
//===----------------------------------------------------------------------===//

LgammaOpAdaptor::LgammaOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("chlo.lgamma", odsAttrs.getContext());
}

LgammaOpAdaptor::LgammaOpAdaptor(LgammaOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange LgammaOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> LgammaOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange LgammaOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LgammaOpAdaptor::getOperand() {
  return *getODSOperands(0).begin();
}

::mlir::Value LgammaOpAdaptor::operand() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr LgammaOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult LgammaOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> LgammaOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range LgammaOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> LgammaOp::getOperand() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> LgammaOp::operand() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange LgammaOp::getOperandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange LgammaOp::operandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> LgammaOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range LgammaOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> LgammaOp::getResult() {
  return *getODSResults(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> LgammaOp::result() {
  return *getODSResults(0).begin();
}

void LgammaOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes(result);
}

void LgammaOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand) {
  odsState.addOperands(operand);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(LgammaOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void LgammaOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand) {
  odsState.addOperands(operand);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LgammaOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void LgammaOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(LgammaOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult LgammaOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps4(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult LgammaOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult LgammaOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand operandRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> operandOperands(operandRawOperands);  ::llvm::SMLoc operandOperandsLoc;
  (void)operandOperandsLoc;
  ::mlir::Type operandRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> operandTypes(operandRawTypes);
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);

  operandOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(operandRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    operandRawTypes[0] = type;
  }
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawTypes[0] = type;
  }
  result.addTypes(resultTypes);
  if (parser.resolveOperands(operandOperands, operandTypes, operandOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void LgammaOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getOperand();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getOperand().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void LgammaOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace chlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::chlo::LgammaOp)

namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// ::mlir::chlo::MinimumBroadcastShapesOp definitions
//===----------------------------------------------------------------------===//

MinimumBroadcastShapesOpAdaptor::MinimumBroadcastShapesOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("chlo.minimum_broadcast_shapes", odsAttrs.getContext());
}

MinimumBroadcastShapesOpAdaptor::MinimumBroadcastShapesOpAdaptor(MinimumBroadcastShapesOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange MinimumBroadcastShapesOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MinimumBroadcastShapesOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange MinimumBroadcastShapesOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange MinimumBroadcastShapesOpAdaptor::getShapes() {
  return getODSOperands(0);
}

::mlir::ValueRange MinimumBroadcastShapesOpAdaptor::shapes() {
  return getODSOperands(0);
}

::mlir::DictionaryAttr MinimumBroadcastShapesOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult MinimumBroadcastShapesOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> MinimumBroadcastShapesOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range MinimumBroadcastShapesOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range MinimumBroadcastShapesOp::getShapes() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range MinimumBroadcastShapesOp::shapes() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange MinimumBroadcastShapesOp::getShapesMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange MinimumBroadcastShapesOp::shapesMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> MinimumBroadcastShapesOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range MinimumBroadcastShapesOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range MinimumBroadcastShapesOp::getResults() {
  return getODSResults(0);
}

::mlir::Operation::result_range MinimumBroadcastShapesOp::results() {
  return getODSResults(0);
}

void MinimumBroadcastShapesOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult MinimumBroadcastShapesOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps8(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps8(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult MinimumBroadcastShapesOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult MinimumBroadcastShapesOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> shapesOperands;
  ::llvm::SMLoc shapesOperandsLoc;
  (void)shapesOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> shapesTypes;
  ::llvm::SmallVector<::mlir::Type, 1> resultsTypes;

  shapesOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(shapesOperands))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(shapesTypes))
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseTypeList(resultsTypes))
    return ::mlir::failure();
  result.addTypes(resultsTypes);
  if (parser.resolveOperands(shapesOperands, shapesTypes, shapesOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void MinimumBroadcastShapesOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getShapes();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getShapes().getTypes();
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  _odsPrinter << getResults().getTypes();
}

void MinimumBroadcastShapesOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace chlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::chlo::MinimumBroadcastShapesOp)

namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// ::mlir::chlo::NextAfterOp definitions
//===----------------------------------------------------------------------===//

NextAfterOpAdaptor::NextAfterOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("chlo.next_after", odsAttrs.getContext());
}

NextAfterOpAdaptor::NextAfterOpAdaptor(NextAfterOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange NextAfterOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> NextAfterOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange NextAfterOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value NextAfterOpAdaptor::getX() {
  return *getODSOperands(0).begin();
}

::mlir::Value NextAfterOpAdaptor::x() {
  return *getODSOperands(0).begin();
}

::mlir::Value NextAfterOpAdaptor::getY() {
  return *getODSOperands(1).begin();
}

::mlir::Value NextAfterOpAdaptor::y() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr NextAfterOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult NextAfterOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> NextAfterOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range NextAfterOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> NextAfterOp::getX() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> NextAfterOp::x() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> NextAfterOp::getY() {
  return *getODSOperands(1).begin();
}

::mlir::TypedValue<::mlir::TensorType> NextAfterOp::y() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange NextAfterOp::getXMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange NextAfterOp::xMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange NextAfterOp::getYMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange NextAfterOp::yMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> NextAfterOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range NextAfterOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> NextAfterOp::getResult() {
  return *getODSResults(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> NextAfterOp::result() {
  return *getODSResults(0).begin();
}

void NextAfterOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value x, ::mlir::Value y) {
  odsState.addOperands(x);
  odsState.addOperands(y);
  odsState.addTypes(result);
}

void NextAfterOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value x, ::mlir::Value y) {
  odsState.addOperands(x);
  odsState.addOperands(y);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(NextAfterOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void NextAfterOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value x, ::mlir::Value y) {
  odsState.addOperands(x);
  odsState.addOperands(y);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void NextAfterOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void NextAfterOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(NextAfterOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult NextAfterOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps4(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult NextAfterOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult NextAfterOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand xRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> xOperands(xRawOperands);  ::llvm::SMLoc xOperandsLoc;
  (void)xOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand yRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> yOperands(yRawOperands);  ::llvm::SMLoc yOperandsLoc;
  (void)yOperandsLoc;
  ::mlir::Type xRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> xTypes(xRawTypes);
  ::mlir::Type yRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> yTypes(yRawTypes);
  ::llvm::SmallVector<::mlir::Type, 1> allResultTypes;

  xOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(xRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  yOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(yRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    xRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    yRawTypes[0] = type;
  }
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseTypeList(allResultTypes))
    return ::mlir::failure();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(xOperands, xTypes, xOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(yOperands, yTypes, yOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void NextAfterOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getX();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getY();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getX().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getY().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  _odsPrinter << getOperation()->getResultTypes();
}

void NextAfterOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace chlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::chlo::NextAfterOp)

namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// ::mlir::chlo::PolygammaOp definitions
//===----------------------------------------------------------------------===//

PolygammaOpAdaptor::PolygammaOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("chlo.polygamma", odsAttrs.getContext());
}

PolygammaOpAdaptor::PolygammaOpAdaptor(PolygammaOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange PolygammaOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> PolygammaOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange PolygammaOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value PolygammaOpAdaptor::getN() {
  return *getODSOperands(0).begin();
}

::mlir::Value PolygammaOpAdaptor::n() {
  return *getODSOperands(0).begin();
}

::mlir::Value PolygammaOpAdaptor::getX() {
  return *getODSOperands(1).begin();
}

::mlir::Value PolygammaOpAdaptor::x() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr PolygammaOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult PolygammaOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> PolygammaOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range PolygammaOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> PolygammaOp::getN() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> PolygammaOp::n() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> PolygammaOp::getX() {
  return *getODSOperands(1).begin();
}

::mlir::TypedValue<::mlir::TensorType> PolygammaOp::x() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange PolygammaOp::getNMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange PolygammaOp::nMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange PolygammaOp::getXMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange PolygammaOp::xMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> PolygammaOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range PolygammaOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> PolygammaOp::getResult() {
  return *getODSResults(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> PolygammaOp::result() {
  return *getODSResults(0).begin();
}

void PolygammaOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value n, ::mlir::Value x) {
  odsState.addOperands(n);
  odsState.addOperands(x);
  odsState.addTypes(result);
}

void PolygammaOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value n, ::mlir::Value x) {
  odsState.addOperands(n);
  odsState.addOperands(x);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(PolygammaOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void PolygammaOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value n, ::mlir::Value x) {
  odsState.addOperands(n);
  odsState.addOperands(x);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void PolygammaOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void PolygammaOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(PolygammaOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult PolygammaOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps4(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult PolygammaOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult PolygammaOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand nRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> nOperands(nRawOperands);  ::llvm::SMLoc nOperandsLoc;
  (void)nOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand xRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> xOperands(xRawOperands);  ::llvm::SMLoc xOperandsLoc;
  (void)xOperandsLoc;
  ::mlir::Type nRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> nTypes(nRawTypes);
  ::mlir::Type xRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> xTypes(xRawTypes);
  ::llvm::SmallVector<::mlir::Type, 1> allResultTypes;

  nOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(nRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  xOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(xRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    nRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    xRawTypes[0] = type;
  }
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseTypeList(allResultTypes))
    return ::mlir::failure();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(nOperands, nTypes, nOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(xOperands, xTypes, xOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void PolygammaOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getN();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getX();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getN().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getX().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  _odsPrinter << getOperation()->getResultTypes();
}

void PolygammaOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace chlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::chlo::PolygammaOp)

namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// ::mlir::chlo::RankSpecializationClusterOp definitions
//===----------------------------------------------------------------------===//

RankSpecializationClusterOpAdaptor::RankSpecializationClusterOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("chlo.rank_specialization_cluster", odsAttrs.getContext());
}

RankSpecializationClusterOpAdaptor::RankSpecializationClusterOpAdaptor(RankSpecializationClusterOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange RankSpecializationClusterOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> RankSpecializationClusterOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange RankSpecializationClusterOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange RankSpecializationClusterOpAdaptor::operands() {
  return getODSOperands(0);
}

::mlir::DictionaryAttr RankSpecializationClusterOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange RankSpecializationClusterOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &RankSpecializationClusterOpAdaptor::getBody() {
  return *odsRegions[0];
}

::mlir::Region &RankSpecializationClusterOpAdaptor::body() {
  return *odsRegions[0];
}

::mlir::LogicalResult RankSpecializationClusterOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> RankSpecializationClusterOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range RankSpecializationClusterOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range RankSpecializationClusterOp::operands() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange RankSpecializationClusterOp::operandsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> RankSpecializationClusterOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range RankSpecializationClusterOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range RankSpecializationClusterOp::getResults() {
  return getODSResults(0);
}

::mlir::Operation::result_range RankSpecializationClusterOp::results() {
  return getODSResults(0);
}

::mlir::Region &RankSpecializationClusterOp::getBody() {
  return (*this)->getRegion(0);
}

::mlir::Region &RankSpecializationClusterOp::body() {
  return (*this)->getRegion(0);
}

void RankSpecializationClusterOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult RankSpecializationClusterOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_ChloOps0(*this, region, "body", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult RankSpecializationClusterOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

} // namespace chlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::chlo::RankSpecializationClusterOp)

namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// ::mlir::chlo::RankSpecializationClusterYieldOp definitions
//===----------------------------------------------------------------------===//

RankSpecializationClusterYieldOpAdaptor::RankSpecializationClusterYieldOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("chlo.rank_specialization_cluster_yield", odsAttrs.getContext());
}

RankSpecializationClusterYieldOpAdaptor::RankSpecializationClusterYieldOpAdaptor(RankSpecializationClusterYieldOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange RankSpecializationClusterYieldOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> RankSpecializationClusterYieldOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange RankSpecializationClusterYieldOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange RankSpecializationClusterYieldOpAdaptor::getResults() {
  return getODSOperands(0);
}

::mlir::ValueRange RankSpecializationClusterYieldOpAdaptor::results() {
  return getODSOperands(0);
}

::mlir::DictionaryAttr RankSpecializationClusterYieldOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult RankSpecializationClusterYieldOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> RankSpecializationClusterYieldOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range RankSpecializationClusterYieldOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range RankSpecializationClusterYieldOp::getResults() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range RankSpecializationClusterYieldOp::results() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange RankSpecializationClusterYieldOp::getResultsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange RankSpecializationClusterYieldOp::resultsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> RankSpecializationClusterYieldOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range RankSpecializationClusterYieldOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void RankSpecializationClusterYieldOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange results) {
  odsState.addOperands(results);
}

void RankSpecializationClusterYieldOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult RankSpecializationClusterYieldOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult RankSpecializationClusterYieldOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void RankSpecializationClusterYieldOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace chlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::chlo::RankSpecializationClusterYieldOp)

namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// ::mlir::chlo::SinhOp definitions
//===----------------------------------------------------------------------===//

SinhOpAdaptor::SinhOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("chlo.sinh", odsAttrs.getContext());
}

SinhOpAdaptor::SinhOpAdaptor(SinhOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange SinhOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> SinhOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SinhOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SinhOpAdaptor::getOperand() {
  return *getODSOperands(0).begin();
}

::mlir::Value SinhOpAdaptor::operand() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr SinhOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult SinhOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> SinhOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SinhOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> SinhOp::getOperand() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> SinhOp::operand() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange SinhOp::getOperandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange SinhOp::operandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> SinhOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SinhOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> SinhOp::getResult() {
  return *getODSResults(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> SinhOp::result() {
  return *getODSResults(0).begin();
}

void SinhOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes(result);
}

void SinhOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand) {
  odsState.addOperands(operand);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(SinhOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void SinhOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand) {
  odsState.addOperands(operand);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SinhOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void SinhOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(SinhOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult SinhOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult SinhOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult SinhOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand operandRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> operandOperands(operandRawOperands);  ::llvm::SMLoc operandOperandsLoc;
  (void)operandOperandsLoc;
  ::mlir::Type operandRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> operandTypes(operandRawTypes);
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);

  operandOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(operandRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    operandRawTypes[0] = type;
  }
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawTypes[0] = type;
  }
  result.addTypes(resultTypes);
  if (parser.resolveOperands(operandOperands, operandTypes, operandOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void SinhOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getOperand();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getOperand().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void SinhOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace chlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::chlo::SinhOp)

namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// ::mlir::chlo::TanOp definitions
//===----------------------------------------------------------------------===//

TanOpAdaptor::TanOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("chlo.tan", odsAttrs.getContext());
}

TanOpAdaptor::TanOpAdaptor(TanOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange TanOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> TanOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange TanOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TanOpAdaptor::getOperand() {
  return *getODSOperands(0).begin();
}

::mlir::Value TanOpAdaptor::operand() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr TanOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult TanOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> TanOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range TanOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> TanOp::getOperand() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> TanOp::operand() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange TanOp::getOperandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange TanOp::operandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> TanOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range TanOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> TanOp::getResult() {
  return *getODSResults(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> TanOp::result() {
  return *getODSResults(0).begin();
}

void TanOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes(result);
}

void TanOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand) {
  odsState.addOperands(operand);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(TanOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void TanOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand) {
  odsState.addOperands(operand);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TanOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void TanOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(TanOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult TanOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult TanOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult TanOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand operandRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> operandOperands(operandRawOperands);  ::llvm::SMLoc operandOperandsLoc;
  (void)operandOperandsLoc;
  ::mlir::Type operandRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> operandTypes(operandRawTypes);
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);

  operandOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(operandRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    operandRawTypes[0] = type;
  }
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawTypes[0] = type;
  }
  result.addTypes(resultTypes);
  if (parser.resolveOperands(operandOperands, operandTypes, operandOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void TanOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getOperand();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getOperand().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void TanOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace chlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::chlo::TanOp)

namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// ::mlir::chlo::TopKOp definitions
//===----------------------------------------------------------------------===//

TopKOpAdaptor::TopKOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("chlo.top_k", odsAttrs.getContext());
}

TopKOpAdaptor::TopKOpAdaptor(TopKOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange TopKOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> TopKOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange TopKOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TopKOpAdaptor::getOperand() {
  return *getODSOperands(0).begin();
}

::mlir::Value TopKOpAdaptor::operand() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr TopKOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr TopKOpAdaptor::getKAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, TopKOp::getKAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
  return attr;
}

uint64_t TopKOpAdaptor::getK() {
  auto attr = getKAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr TopKOpAdaptor::kAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, TopKOp::getKAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
  return attr;
}

uint64_t TopKOpAdaptor::k() {
  auto attr = kAttr();
  return attr.getValue().getZExtValue();
}

::mlir::LogicalResult TopKOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_k;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'chlo.top_k' op ""requires attribute 'k'");
    if (namedAttrIt->getName() == TopKOp::getKAttrName(*odsOpName)) {
      tblgen_k = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_k && !(((tblgen_k.isa<::mlir::IntegerAttr>())) && ((tblgen_k.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
    return emitError(loc, "'chlo.top_k' op ""attribute 'k' failed to satisfy constraint: 64-bit signless integer attribute");
  return ::mlir::success();
}

void TopKOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn) {
  auto resultGroup0 = getODSResults(0);
  if (!resultGroup0.empty())
    setNameFn(*resultGroup0.begin(), "values");
  auto resultGroup1 = getODSResults(1);
  if (!resultGroup1.empty())
    setNameFn(*resultGroup1.begin(), "indices");
}

std::pair<unsigned, unsigned> TopKOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range TopKOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> TopKOp::getOperand() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> TopKOp::operand() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange TopKOp::getOperandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange TopKOp::operandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> TopKOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range TopKOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> TopKOp::getValues() {
  return *getODSResults(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> TopKOp::values() {
  return *getODSResults(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> TopKOp::getIndices() {
  return *getODSResults(1).begin();
}

::mlir::TypedValue<::mlir::TensorType> TopKOp::indices() {
  return *getODSResults(1).begin();
}

::mlir::IntegerAttr TopKOp::getKAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getKAttrName()).cast<::mlir::IntegerAttr>();
}

uint64_t TopKOp::getK() {
  auto attr = getKAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr TopKOp::kAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getKAttrName()).cast<::mlir::IntegerAttr>();
}

uint64_t TopKOp::k() {
  auto attr = kAttr();
  return attr.getValue().getZExtValue();
}

void TopKOp::setKAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getKAttrName(), attr);
}

void TopKOp::kAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(kAttrName(), attr);
}

void TopKOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type values, ::mlir::Type indices, ::mlir::Value operand, ::mlir::IntegerAttr k) {
  odsState.addOperands(operand);
  odsState.addAttribute(getKAttrName(odsState.name), k);
  odsState.addTypes(values);
  odsState.addTypes(indices);
}

void TopKOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand, ::mlir::IntegerAttr k) {
  odsState.addOperands(operand);
  odsState.addAttribute(getKAttrName(odsState.name), k);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(TopKOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void TopKOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand, ::mlir::IntegerAttr k) {
  odsState.addOperands(operand);
  odsState.addAttribute(getKAttrName(odsState.name), k);
  assert(resultTypes.size() == 2u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TopKOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type values, ::mlir::Type indices, ::mlir::Value operand, uint64_t k) {
  odsState.addOperands(operand);
  odsState.addAttribute(getKAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), k));
  odsState.addTypes(values);
  odsState.addTypes(indices);
}

void TopKOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand, uint64_t k) {
  odsState.addOperands(operand);
  odsState.addAttribute(getKAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), k));

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(TopKOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void TopKOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand, uint64_t k) {
  odsState.addOperands(operand);
  odsState.addAttribute(getKAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), k));
  assert(resultTypes.size() == 2u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TopKOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 2u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void TopKOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(TopKOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 2u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult TopKOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_k;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'k'");
    if (namedAttrIt->getName() == getKAttrName()) {
      tblgen_k = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ChloOps5(*this, tblgen_k, "k")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSResults(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult TopKOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult TopKOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand operandRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> operandOperands(operandRawOperands);  ::llvm::SMLoc operandOperandsLoc;
  (void)operandOperandsLoc;
  ::mlir::IntegerAttr kAttr;
  ::mlir::Type operandRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> operandTypes(operandRawTypes);
  ::mlir::Type valuesRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> valuesTypes(valuesRawTypes);
  ::mlir::Type indicesRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> indicesTypes(indicesRawTypes);
  if (parser.parseLParen())
    return ::mlir::failure();

  operandOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(operandRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();
  if (parser.parseKeyword("k"))
    return ::mlir::failure();
  if (parser.parseEqual())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(kAttr, parser.getBuilder().getIntegerType(64), "k",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    operandRawTypes[0] = type;
  }
  if (parser.parseArrow())
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    valuesRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    indicesRawTypes[0] = type;
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  result.addTypes(valuesTypes);
  result.addTypes(indicesTypes);
  if (parser.resolveOperands(operandOperands, operandTypes, operandOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void TopKOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << "(";
  _odsPrinter << getOperand();
  _odsPrinter << ",";
  _odsPrinter << ' ' << "k";
  _odsPrinter << ' ' << "=";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getKAttr());
  _odsPrinter << ")";
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"k"});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getOperand().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ' << "(";
  {
    auto type = getValues().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getIndices().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ")";
}

void TopKOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace chlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::chlo::TopKOp)

namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// ::mlir::chlo::ZetaOp definitions
//===----------------------------------------------------------------------===//

ZetaOpAdaptor::ZetaOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("chlo.zeta", odsAttrs.getContext());
}

ZetaOpAdaptor::ZetaOpAdaptor(ZetaOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange ZetaOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ZetaOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ZetaOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ZetaOpAdaptor::getX() {
  return *getODSOperands(0).begin();
}

::mlir::Value ZetaOpAdaptor::x() {
  return *getODSOperands(0).begin();
}

::mlir::Value ZetaOpAdaptor::getQ() {
  return *getODSOperands(1).begin();
}

::mlir::Value ZetaOpAdaptor::q() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr ZetaOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ZetaOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ZetaOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ZetaOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> ZetaOp::getX() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> ZetaOp::x() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> ZetaOp::getQ() {
  return *getODSOperands(1).begin();
}

::mlir::TypedValue<::mlir::TensorType> ZetaOp::q() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange ZetaOp::getXMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ZetaOp::xMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ZetaOp::getQMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ZetaOp::qMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ZetaOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ZetaOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> ZetaOp::getResult() {
  return *getODSResults(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> ZetaOp::result() {
  return *getODSResults(0).begin();
}

void ZetaOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value x, ::mlir::Value q) {
  odsState.addOperands(x);
  odsState.addOperands(q);
  odsState.addTypes(result);
}

void ZetaOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value x, ::mlir::Value q) {
  odsState.addOperands(x);
  odsState.addOperands(q);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(ZetaOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void ZetaOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value x, ::mlir::Value q) {
  odsState.addOperands(x);
  odsState.addOperands(q);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ZetaOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void ZetaOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(ZetaOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult ZetaOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ChloOps4(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ZetaOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult ZetaOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand xRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> xOperands(xRawOperands);  ::llvm::SMLoc xOperandsLoc;
  (void)xOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand qRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> qOperands(qRawOperands);  ::llvm::SMLoc qOperandsLoc;
  (void)qOperandsLoc;
  ::mlir::Type xRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> xTypes(xRawTypes);
  ::mlir::Type qRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> qTypes(qRawTypes);
  ::llvm::SmallVector<::mlir::Type, 1> allResultTypes;

  xOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(xRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  qOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(qRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    xRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    qRawTypes[0] = type;
  }
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseTypeList(allResultTypes))
    return ::mlir::failure();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(xOperands, xTypes, xOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(qOperands, qTypes, qOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ZetaOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getX();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getQ();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getX().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getQ().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  _odsPrinter << getOperation()->getResultTypes();
}

void ZetaOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace chlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::chlo::ZetaOp)


#endif  // GET_OP_CLASSES

