/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Declarations                                                            *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#if defined(GET_OP_CLASSES) || defined(GET_OP_FWD_DEFINES)
#undef GET_OP_FWD_DEFINES
namespace mlir {
namespace lmhlo_gpu {
class AllReduceDoneOp;
} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {
class AllReduceStartOp;
} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {
class CholeskyOp;
} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {
class ConvBackwardFilterOp;
} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {
class ConvBackwardInputOp;
} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {
class ConvForwardFusedOp;
} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {
class ConvForwardFusedSideInputOp;
} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {
class ConvForwardOp;
} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {
class CublasLtMatmulOp;
} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {
class GEMMOp;
} // namespace lmhlo_gpu
} // namespace mlir
#endif

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::AllReduceDoneOp declarations
//===----------------------------------------------------------------------===//

class AllReduceDoneOpAdaptor {
public:
  AllReduceDoneOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  AllReduceDoneOpAdaptor(AllReduceDoneOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getToken();
  ::mlir::ValueRange getInputs();
  ::mlir::ValueRange getOutputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class AllReduceDoneOp : public ::mlir::Op<AllReduceDoneOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AllReduceDoneOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("lmhlo_gpu.all_reduce_done");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value getToken();
  ::mlir::Operation::operand_range getInputs();
  ::mlir::Operation::operand_range getOutputs();
  ::mlir::MutableOperandRange getTokenMutable();
  ::mlir::MutableOperandRange getInputsMutable();
  ::mlir::MutableOperandRange getOutputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value token, ::mlir::ValueRange inputs, ::mlir::ValueRange outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value token, ::mlir::ValueRange inputs, ::mlir::ValueRange outputs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace lmhlo_gpu
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::lmhlo_gpu::AllReduceDoneOp)

namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::AllReduceStartOp declarations
//===----------------------------------------------------------------------===//

class AllReduceStartOpAdaptor {
public:
  AllReduceStartOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  AllReduceStartOpAdaptor(AllReduceStartOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange getInputs();
  ::mlir::ValueRange getOutputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr getReplicaGroupsAttr();
  ::mlir::DenseIntElementsAttr getReplicaGroups();
  ::mlir::BoolAttr getConstrainLayoutAttr();
  bool getConstrainLayout();
  ::mlir::mhlo::ChannelHandleAttr getChannelIdAttr();
  ::llvm::Optional<::mlir::mhlo::ChannelHandleAttr> getChannelId();
  ::mlir::BoolAttr getUseGlobalDeviceIdsAttr();
  bool getUseGlobalDeviceIds();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &getComputation();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class AllReduceStartOp : public ::mlir::Op<AllReduceStartOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::SameOperandsElementType> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AllReduceStartOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("channel_id"), ::llvm::StringRef("constrain_layout"), ::llvm::StringRef("replica_groups"), ::llvm::StringRef("use_global_device_ids")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getChannelIdAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getChannelIdAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getConstrainLayoutAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getConstrainLayoutAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getReplicaGroupsAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getReplicaGroupsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getUseGlobalDeviceIdsAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getUseGlobalDeviceIdsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("lmhlo_gpu.all_reduce_start");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getInputs();
  ::mlir::Operation::operand_range getOutputs();
  ::mlir::MutableOperandRange getInputsMutable();
  ::mlir::MutableOperandRange getOutputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value getToken();
  ::mlir::Region &getComputation();
  ::mlir::DenseIntElementsAttr getReplicaGroupsAttr();
  ::mlir::DenseIntElementsAttr getReplicaGroups();
  ::mlir::BoolAttr getConstrainLayoutAttr();
  bool getConstrainLayout();
  ::mlir::mhlo::ChannelHandleAttr getChannelIdAttr();
  ::llvm::Optional<::mlir::mhlo::ChannelHandleAttr> getChannelId();
  ::mlir::BoolAttr getUseGlobalDeviceIdsAttr();
  bool getUseGlobalDeviceIds();
  void setReplicaGroupsAttr(::mlir::DenseIntElementsAttr attr);
  void setConstrainLayoutAttr(::mlir::BoolAttr attr);
  void setChannelIdAttr(::mlir::mhlo::ChannelHandleAttr attr);
  void setUseGlobalDeviceIdsAttr(::mlir::BoolAttr attr);
  ::mlir::Attribute removeConstrain_layoutAttr();
  ::mlir::Attribute removeChannel_idAttr();
  ::mlir::Attribute removeUse_global_device_idsAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type token, ::mlir::ValueRange inputs, ::mlir::ValueRange outputs, ::mlir::DenseIntElementsAttr replica_groups, /*optional*/::mlir::BoolAttr constrain_layout, /*optional*/::mlir::mhlo::ChannelHandleAttr channel_id, /*optional*/::mlir::BoolAttr use_global_device_ids);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange inputs, ::mlir::ValueRange outputs, ::mlir::DenseIntElementsAttr replica_groups, /*optional*/::mlir::BoolAttr constrain_layout, /*optional*/::mlir::mhlo::ChannelHandleAttr channel_id, /*optional*/::mlir::BoolAttr use_global_device_ids);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type token, ::mlir::ValueRange inputs, ::mlir::ValueRange outputs, ::mlir::DenseIntElementsAttr replica_groups, /*optional*/bool constrain_layout, /*optional*/::mlir::mhlo::ChannelHandleAttr channel_id, /*optional*/bool use_global_device_ids = false);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange inputs, ::mlir::ValueRange outputs, ::mlir::DenseIntElementsAttr replica_groups, /*optional*/bool constrain_layout, /*optional*/::mlir::mhlo::ChannelHandleAttr channel_id, /*optional*/bool use_global_device_ids = false);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 4 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
};
} // namespace lmhlo_gpu
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::lmhlo_gpu::AllReduceStartOp)

namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::CholeskyOp declarations
//===----------------------------------------------------------------------===//

class CholeskyOpAdaptor {
public:
  CholeskyOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  CholeskyOpAdaptor(CholeskyOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getInput();
  ::mlir::Value getOutput();
  ::mlir::Value getScratch();
  ::mlir::Value getInfo();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::BoolAttr getIsLowerAttr();
  bool getIsLower();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class CholeskyOp : public ::mlir::Op<CholeskyOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<4>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = CholeskyOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("is_lower")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getIsLowerAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIsLowerAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("lmhlo_gpu.cholesky");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::MemRefType> getInput();
  ::mlir::TypedValue<::mlir::MemRefType> getOutput();
  ::mlir::TypedValue<::mlir::MemRefType> getScratch();
  ::mlir::TypedValue<::mlir::MemRefType> getInfo();
  ::mlir::MutableOperandRange getInputMutable();
  ::mlir::MutableOperandRange getOutputMutable();
  ::mlir::MutableOperandRange getScratchMutable();
  ::mlir::MutableOperandRange getInfoMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::BoolAttr getIsLowerAttr();
  bool getIsLower();
  void setIsLowerAttr(::mlir::BoolAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value output, ::mlir::Value scratch, ::mlir::Value info, ::mlir::BoolAttr is_lower);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value output, ::mlir::Value scratch, ::mlir::Value info, ::mlir::BoolAttr is_lower);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value output, ::mlir::Value scratch, ::mlir::Value info, bool is_lower);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value output, ::mlir::Value scratch, ::mlir::Value info, bool is_lower);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
};
} // namespace lmhlo_gpu
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::lmhlo_gpu::CholeskyOp)

namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::ConvBackwardFilterOp declarations
//===----------------------------------------------------------------------===//

class ConvBackwardFilterOpAdaptor {
public:
  ConvBackwardFilterOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  ConvBackwardFilterOpAdaptor(ConvBackwardFilterOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getInput();
  ::mlir::Value getDOutput();
  ::mlir::Value getDFilter();
  ::mlir::Value getScratch();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr getWindowStridesAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > getWindowStrides();
  ::mlir::DenseIntElementsAttr getPaddingAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > getPadding();
  ::mlir::DenseIntElementsAttr getLhsDilationAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > getLhsDilation();
  ::mlir::DenseIntElementsAttr getRhsDilationAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > getRhsDilation();
  ::mlir::DenseElementsAttr getWindowReversalAttr();
  ::llvm::Optional< ::mlir::DenseElementsAttr > getWindowReversal();
  ::mlir::mhlo::ConvDimensionNumbersAttr getDimensionNumbersAttr();
  ::mlir::mhlo::ConvDimensionNumbersAttr getDimensionNumbers();
  ::mlir::IntegerAttr getFeatureGroupCountAttr();
  uint64_t getFeatureGroupCount();
  ::mlir::IntegerAttr getBatchGroupCountAttr();
  uint64_t getBatchGroupCount();
  ::mlir::ArrayAttr getPrecisionConfigAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getPrecisionConfig();
  ::mlir::FloatAttr getResultScaleAttr();
  ::llvm::APFloat getResultScale();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr getBackendConfigAttr();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr getBackendConfig();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class ConvBackwardFilterOp : public ::mlir::Op<ConvBackwardFilterOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<4>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ConvBackwardFilterOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("backend_config"), ::llvm::StringRef("batch_group_count"), ::llvm::StringRef("dimension_numbers"), ::llvm::StringRef("feature_group_count"), ::llvm::StringRef("lhs_dilation"), ::llvm::StringRef("padding"), ::llvm::StringRef("precision_config"), ::llvm::StringRef("result_scale"), ::llvm::StringRef("rhs_dilation"), ::llvm::StringRef("window_reversal"), ::llvm::StringRef("window_strides")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getBackendConfigAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getBackendConfigAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getBatchGroupCountAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getBatchGroupCountAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getDimensionNumbersAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getDimensionNumbersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getFeatureGroupCountAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getFeatureGroupCountAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getLhsDilationAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getLhsDilationAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr getPaddingAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr getPaddingAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  ::mlir::StringAttr getPrecisionConfigAttrName() {
    return getAttributeNameForIndex(6);
  }

  static ::mlir::StringAttr getPrecisionConfigAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 6);
  }

  ::mlir::StringAttr getResultScaleAttrName() {
    return getAttributeNameForIndex(7);
  }

  static ::mlir::StringAttr getResultScaleAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 7);
  }

  ::mlir::StringAttr getRhsDilationAttrName() {
    return getAttributeNameForIndex(8);
  }

  static ::mlir::StringAttr getRhsDilationAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 8);
  }

  ::mlir::StringAttr getWindowReversalAttrName() {
    return getAttributeNameForIndex(9);
  }

  static ::mlir::StringAttr getWindowReversalAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 9);
  }

  ::mlir::StringAttr getWindowStridesAttrName() {
    return getAttributeNameForIndex(10);
  }

  static ::mlir::StringAttr getWindowStridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 10);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("lmhlo_gpu.conv_backwardfilter");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::MemRefType> getInput();
  ::mlir::TypedValue<::mlir::MemRefType> getDOutput();
  ::mlir::TypedValue<::mlir::MemRefType> getDFilter();
  ::mlir::TypedValue<::mlir::MemRefType> getScratch();
  ::mlir::MutableOperandRange getInputMutable();
  ::mlir::MutableOperandRange getDOutputMutable();
  ::mlir::MutableOperandRange getDFilterMutable();
  ::mlir::MutableOperandRange getScratchMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::DenseIntElementsAttr getWindowStridesAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > getWindowStrides();
  ::mlir::DenseIntElementsAttr getPaddingAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > getPadding();
  ::mlir::DenseIntElementsAttr getLhsDilationAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > getLhsDilation();
  ::mlir::DenseIntElementsAttr getRhsDilationAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > getRhsDilation();
  ::mlir::DenseElementsAttr getWindowReversalAttr();
  ::llvm::Optional< ::mlir::DenseElementsAttr > getWindowReversal();
  ::mlir::mhlo::ConvDimensionNumbersAttr getDimensionNumbersAttr();
  ::mlir::mhlo::ConvDimensionNumbersAttr getDimensionNumbers();
  ::mlir::IntegerAttr getFeatureGroupCountAttr();
  uint64_t getFeatureGroupCount();
  ::mlir::IntegerAttr getBatchGroupCountAttr();
  uint64_t getBatchGroupCount();
  ::mlir::ArrayAttr getPrecisionConfigAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getPrecisionConfig();
  ::mlir::FloatAttr getResultScaleAttr();
  ::llvm::APFloat getResultScale();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr getBackendConfigAttr();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr getBackendConfig();
  void setWindowStridesAttr(::mlir::DenseIntElementsAttr attr);
  void setPaddingAttr(::mlir::DenseIntElementsAttr attr);
  void setLhsDilationAttr(::mlir::DenseIntElementsAttr attr);
  void setRhsDilationAttr(::mlir::DenseIntElementsAttr attr);
  void setWindowReversalAttr(::mlir::DenseElementsAttr attr);
  void setDimensionNumbersAttr(::mlir::mhlo::ConvDimensionNumbersAttr attr);
  void setFeatureGroupCountAttr(::mlir::IntegerAttr attr);
  void setBatchGroupCountAttr(::mlir::IntegerAttr attr);
  void setPrecisionConfigAttr(::mlir::ArrayAttr attr);
  void setResultScaleAttr(::mlir::FloatAttr attr);
  void setBackendConfigAttr(::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr attr);
  ::mlir::Attribute removeWindow_stridesAttr();
  ::mlir::Attribute removePaddingAttr();
  ::mlir::Attribute removeLhs_dilationAttr();
  ::mlir::Attribute removeRhs_dilationAttr();
  ::mlir::Attribute removeWindow_reversalAttr();
  ::mlir::Attribute removePrecision_configAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value d_output, ::mlir::Value d_filter, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers, ::mlir::IntegerAttr feature_group_count, ::mlir::IntegerAttr batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::mlir::FloatAttr result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value d_output, ::mlir::Value d_filter, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers, ::mlir::IntegerAttr feature_group_count, ::mlir::IntegerAttr batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::mlir::FloatAttr result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value d_output, ::mlir::Value d_filter, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers, uint64_t feature_group_count, uint64_t batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::llvm::APFloat result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value d_output, ::mlir::Value d_filter, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers, uint64_t feature_group_count, uint64_t batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::llvm::APFloat result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr backend_config);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 11 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
};
} // namespace lmhlo_gpu
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::lmhlo_gpu::ConvBackwardFilterOp)

namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::ConvBackwardInputOp declarations
//===----------------------------------------------------------------------===//

class ConvBackwardInputOpAdaptor {
public:
  ConvBackwardInputOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  ConvBackwardInputOpAdaptor(ConvBackwardInputOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getDOutput();
  ::mlir::Value getFilter();
  ::mlir::Value getDInput();
  ::mlir::Value getScratch();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr getWindowStridesAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > getWindowStrides();
  ::mlir::DenseIntElementsAttr getPaddingAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > getPadding();
  ::mlir::DenseIntElementsAttr getLhsDilationAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > getLhsDilation();
  ::mlir::DenseIntElementsAttr getRhsDilationAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > getRhsDilation();
  ::mlir::DenseElementsAttr getWindowReversalAttr();
  ::llvm::Optional< ::mlir::DenseElementsAttr > getWindowReversal();
  ::mlir::mhlo::ConvDimensionNumbersAttr getDimensionNumbersAttr();
  ::mlir::mhlo::ConvDimensionNumbersAttr getDimensionNumbers();
  ::mlir::IntegerAttr getFeatureGroupCountAttr();
  uint64_t getFeatureGroupCount();
  ::mlir::IntegerAttr getBatchGroupCountAttr();
  uint64_t getBatchGroupCount();
  ::mlir::ArrayAttr getPrecisionConfigAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getPrecisionConfig();
  ::mlir::FloatAttr getResultScaleAttr();
  ::llvm::APFloat getResultScale();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr getBackendConfigAttr();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr getBackendConfig();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class ConvBackwardInputOp : public ::mlir::Op<ConvBackwardInputOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<4>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ConvBackwardInputOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("backend_config"), ::llvm::StringRef("batch_group_count"), ::llvm::StringRef("dimension_numbers"), ::llvm::StringRef("feature_group_count"), ::llvm::StringRef("lhs_dilation"), ::llvm::StringRef("padding"), ::llvm::StringRef("precision_config"), ::llvm::StringRef("result_scale"), ::llvm::StringRef("rhs_dilation"), ::llvm::StringRef("window_reversal"), ::llvm::StringRef("window_strides")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getBackendConfigAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getBackendConfigAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getBatchGroupCountAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getBatchGroupCountAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getDimensionNumbersAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getDimensionNumbersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getFeatureGroupCountAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getFeatureGroupCountAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getLhsDilationAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getLhsDilationAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr getPaddingAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr getPaddingAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  ::mlir::StringAttr getPrecisionConfigAttrName() {
    return getAttributeNameForIndex(6);
  }

  static ::mlir::StringAttr getPrecisionConfigAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 6);
  }

  ::mlir::StringAttr getResultScaleAttrName() {
    return getAttributeNameForIndex(7);
  }

  static ::mlir::StringAttr getResultScaleAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 7);
  }

  ::mlir::StringAttr getRhsDilationAttrName() {
    return getAttributeNameForIndex(8);
  }

  static ::mlir::StringAttr getRhsDilationAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 8);
  }

  ::mlir::StringAttr getWindowReversalAttrName() {
    return getAttributeNameForIndex(9);
  }

  static ::mlir::StringAttr getWindowReversalAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 9);
  }

  ::mlir::StringAttr getWindowStridesAttrName() {
    return getAttributeNameForIndex(10);
  }

  static ::mlir::StringAttr getWindowStridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 10);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("lmhlo_gpu.conv_backwardinput");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::MemRefType> getDOutput();
  ::mlir::TypedValue<::mlir::MemRefType> getFilter();
  ::mlir::TypedValue<::mlir::MemRefType> getDInput();
  ::mlir::TypedValue<::mlir::MemRefType> getScratch();
  ::mlir::MutableOperandRange getDOutputMutable();
  ::mlir::MutableOperandRange getFilterMutable();
  ::mlir::MutableOperandRange getDInputMutable();
  ::mlir::MutableOperandRange getScratchMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::DenseIntElementsAttr getWindowStridesAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > getWindowStrides();
  ::mlir::DenseIntElementsAttr getPaddingAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > getPadding();
  ::mlir::DenseIntElementsAttr getLhsDilationAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > getLhsDilation();
  ::mlir::DenseIntElementsAttr getRhsDilationAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > getRhsDilation();
  ::mlir::DenseElementsAttr getWindowReversalAttr();
  ::llvm::Optional< ::mlir::DenseElementsAttr > getWindowReversal();
  ::mlir::mhlo::ConvDimensionNumbersAttr getDimensionNumbersAttr();
  ::mlir::mhlo::ConvDimensionNumbersAttr getDimensionNumbers();
  ::mlir::IntegerAttr getFeatureGroupCountAttr();
  uint64_t getFeatureGroupCount();
  ::mlir::IntegerAttr getBatchGroupCountAttr();
  uint64_t getBatchGroupCount();
  ::mlir::ArrayAttr getPrecisionConfigAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getPrecisionConfig();
  ::mlir::FloatAttr getResultScaleAttr();
  ::llvm::APFloat getResultScale();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr getBackendConfigAttr();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr getBackendConfig();
  void setWindowStridesAttr(::mlir::DenseIntElementsAttr attr);
  void setPaddingAttr(::mlir::DenseIntElementsAttr attr);
  void setLhsDilationAttr(::mlir::DenseIntElementsAttr attr);
  void setRhsDilationAttr(::mlir::DenseIntElementsAttr attr);
  void setWindowReversalAttr(::mlir::DenseElementsAttr attr);
  void setDimensionNumbersAttr(::mlir::mhlo::ConvDimensionNumbersAttr attr);
  void setFeatureGroupCountAttr(::mlir::IntegerAttr attr);
  void setBatchGroupCountAttr(::mlir::IntegerAttr attr);
  void setPrecisionConfigAttr(::mlir::ArrayAttr attr);
  void setResultScaleAttr(::mlir::FloatAttr attr);
  void setBackendConfigAttr(::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr attr);
  ::mlir::Attribute removeWindow_stridesAttr();
  ::mlir::Attribute removePaddingAttr();
  ::mlir::Attribute removeLhs_dilationAttr();
  ::mlir::Attribute removeRhs_dilationAttr();
  ::mlir::Attribute removeWindow_reversalAttr();
  ::mlir::Attribute removePrecision_configAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value d_output, ::mlir::Value filter, ::mlir::Value d_input, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers, ::mlir::IntegerAttr feature_group_count, ::mlir::IntegerAttr batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::mlir::FloatAttr result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value d_output, ::mlir::Value filter, ::mlir::Value d_input, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers, ::mlir::IntegerAttr feature_group_count, ::mlir::IntegerAttr batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::mlir::FloatAttr result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value d_output, ::mlir::Value filter, ::mlir::Value d_input, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers, uint64_t feature_group_count, uint64_t batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::llvm::APFloat result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value d_output, ::mlir::Value filter, ::mlir::Value d_input, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers, uint64_t feature_group_count, uint64_t batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::llvm::APFloat result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr backend_config);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 11 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
};
} // namespace lmhlo_gpu
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::lmhlo_gpu::ConvBackwardInputOp)

namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::ConvForwardFusedOp declarations
//===----------------------------------------------------------------------===//

class ConvForwardFusedOpAdaptor {
public:
  ConvForwardFusedOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  ConvForwardFusedOpAdaptor(ConvForwardFusedOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getInput();
  ::mlir::Value getFilter();
  ::mlir::Value getBias();
  ::mlir::Value getOutput();
  ::mlir::Value getScratch();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr getWindowStridesAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > getWindowStrides();
  ::mlir::DenseIntElementsAttr getPaddingAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > getPadding();
  ::mlir::DenseIntElementsAttr getLhsDilationAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > getLhsDilation();
  ::mlir::DenseIntElementsAttr getRhsDilationAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > getRhsDilation();
  ::mlir::DenseElementsAttr getWindowReversalAttr();
  ::llvm::Optional< ::mlir::DenseElementsAttr > getWindowReversal();
  ::mlir::mhlo::ConvDimensionNumbersAttr getDimensionNumbersAttr();
  ::mlir::mhlo::ConvDimensionNumbersAttr getDimensionNumbers();
  ::mlir::IntegerAttr getFeatureGroupCountAttr();
  uint64_t getFeatureGroupCount();
  ::mlir::IntegerAttr getBatchGroupCountAttr();
  uint64_t getBatchGroupCount();
  ::mlir::ArrayAttr getPrecisionConfigAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getPrecisionConfig();
  ::mlir::FloatAttr getResultScaleAttr();
  ::llvm::APFloat getResultScale();
  ::mlir::lmhlo_gpu::ActivationAttr getActivationModeAttr();
  ::mlir::lmhlo_gpu::Activation getActivationMode();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr getBackendConfigAttr();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr getBackendConfig();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class ConvForwardFusedOp : public ::mlir::Op<ConvForwardFusedOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<5>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ConvForwardFusedOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("activation_mode"), ::llvm::StringRef("backend_config"), ::llvm::StringRef("batch_group_count"), ::llvm::StringRef("dimension_numbers"), ::llvm::StringRef("feature_group_count"), ::llvm::StringRef("lhs_dilation"), ::llvm::StringRef("padding"), ::llvm::StringRef("precision_config"), ::llvm::StringRef("result_scale"), ::llvm::StringRef("rhs_dilation"), ::llvm::StringRef("window_reversal"), ::llvm::StringRef("window_strides")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getActivationModeAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getActivationModeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getBackendConfigAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getBackendConfigAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getBatchGroupCountAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getBatchGroupCountAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getDimensionNumbersAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getDimensionNumbersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getFeatureGroupCountAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getFeatureGroupCountAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr getLhsDilationAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr getLhsDilationAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  ::mlir::StringAttr getPaddingAttrName() {
    return getAttributeNameForIndex(6);
  }

  static ::mlir::StringAttr getPaddingAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 6);
  }

  ::mlir::StringAttr getPrecisionConfigAttrName() {
    return getAttributeNameForIndex(7);
  }

  static ::mlir::StringAttr getPrecisionConfigAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 7);
  }

  ::mlir::StringAttr getResultScaleAttrName() {
    return getAttributeNameForIndex(8);
  }

  static ::mlir::StringAttr getResultScaleAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 8);
  }

  ::mlir::StringAttr getRhsDilationAttrName() {
    return getAttributeNameForIndex(9);
  }

  static ::mlir::StringAttr getRhsDilationAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 9);
  }

  ::mlir::StringAttr getWindowReversalAttrName() {
    return getAttributeNameForIndex(10);
  }

  static ::mlir::StringAttr getWindowReversalAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 10);
  }

  ::mlir::StringAttr getWindowStridesAttrName() {
    return getAttributeNameForIndex(11);
  }

  static ::mlir::StringAttr getWindowStridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 11);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("lmhlo_gpu.conv_forward_fused");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::MemRefType> getInput();
  ::mlir::TypedValue<::mlir::MemRefType> getFilter();
  ::mlir::TypedValue<::mlir::MemRefType> getBias();
  ::mlir::TypedValue<::mlir::MemRefType> getOutput();
  ::mlir::TypedValue<::mlir::MemRefType> getScratch();
  ::mlir::MutableOperandRange getInputMutable();
  ::mlir::MutableOperandRange getFilterMutable();
  ::mlir::MutableOperandRange getBiasMutable();
  ::mlir::MutableOperandRange getOutputMutable();
  ::mlir::MutableOperandRange getScratchMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::DenseIntElementsAttr getWindowStridesAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > getWindowStrides();
  ::mlir::DenseIntElementsAttr getPaddingAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > getPadding();
  ::mlir::DenseIntElementsAttr getLhsDilationAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > getLhsDilation();
  ::mlir::DenseIntElementsAttr getRhsDilationAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > getRhsDilation();
  ::mlir::DenseElementsAttr getWindowReversalAttr();
  ::llvm::Optional< ::mlir::DenseElementsAttr > getWindowReversal();
  ::mlir::mhlo::ConvDimensionNumbersAttr getDimensionNumbersAttr();
  ::mlir::mhlo::ConvDimensionNumbersAttr getDimensionNumbers();
  ::mlir::IntegerAttr getFeatureGroupCountAttr();
  uint64_t getFeatureGroupCount();
  ::mlir::IntegerAttr getBatchGroupCountAttr();
  uint64_t getBatchGroupCount();
  ::mlir::ArrayAttr getPrecisionConfigAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getPrecisionConfig();
  ::mlir::FloatAttr getResultScaleAttr();
  ::llvm::APFloat getResultScale();
  ::mlir::lmhlo_gpu::ActivationAttr getActivationModeAttr();
  ::mlir::lmhlo_gpu::Activation getActivationMode();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr getBackendConfigAttr();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr getBackendConfig();
  void setWindowStridesAttr(::mlir::DenseIntElementsAttr attr);
  void setPaddingAttr(::mlir::DenseIntElementsAttr attr);
  void setLhsDilationAttr(::mlir::DenseIntElementsAttr attr);
  void setRhsDilationAttr(::mlir::DenseIntElementsAttr attr);
  void setWindowReversalAttr(::mlir::DenseElementsAttr attr);
  void setDimensionNumbersAttr(::mlir::mhlo::ConvDimensionNumbersAttr attr);
  void setFeatureGroupCountAttr(::mlir::IntegerAttr attr);
  void setBatchGroupCountAttr(::mlir::IntegerAttr attr);
  void setPrecisionConfigAttr(::mlir::ArrayAttr attr);
  void setResultScaleAttr(::mlir::FloatAttr attr);
  void setActivationModeAttr(::mlir::lmhlo_gpu::ActivationAttr attr);
  void setBackendConfigAttr(::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr attr);
  ::mlir::Attribute removeWindow_stridesAttr();
  ::mlir::Attribute removePaddingAttr();
  ::mlir::Attribute removeLhs_dilationAttr();
  ::mlir::Attribute removeRhs_dilationAttr();
  ::mlir::Attribute removeWindow_reversalAttr();
  ::mlir::Attribute removePrecision_configAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value bias, ::mlir::Value output, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers, ::mlir::IntegerAttr feature_group_count, ::mlir::IntegerAttr batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::mlir::FloatAttr result_scale, ::mlir::lmhlo_gpu::ActivationAttr activation_mode, ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value bias, ::mlir::Value output, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers, ::mlir::IntegerAttr feature_group_count, ::mlir::IntegerAttr batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::mlir::FloatAttr result_scale, ::mlir::lmhlo_gpu::ActivationAttr activation_mode, ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value bias, ::mlir::Value output, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers, uint64_t feature_group_count, uint64_t batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::llvm::APFloat result_scale, ::mlir::lmhlo_gpu::Activation activation_mode, ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value bias, ::mlir::Value output, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers, uint64_t feature_group_count, uint64_t batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::llvm::APFloat result_scale, ::mlir::lmhlo_gpu::Activation activation_mode, ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr backend_config);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 12 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
};
} // namespace lmhlo_gpu
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::lmhlo_gpu::ConvForwardFusedOp)

namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::ConvForwardFusedSideInputOp declarations
//===----------------------------------------------------------------------===//

class ConvForwardFusedSideInputOpAdaptor {
public:
  ConvForwardFusedSideInputOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  ConvForwardFusedSideInputOpAdaptor(ConvForwardFusedSideInputOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getInput();
  ::mlir::Value getFilter();
  ::mlir::Value getBias();
  ::mlir::Value getSideInput();
  ::mlir::Value getOutput();
  ::mlir::Value getScratch();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr getWindowStridesAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > getWindowStrides();
  ::mlir::DenseIntElementsAttr getPaddingAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > getPadding();
  ::mlir::DenseIntElementsAttr getLhsDilationAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > getLhsDilation();
  ::mlir::DenseIntElementsAttr getRhsDilationAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > getRhsDilation();
  ::mlir::DenseElementsAttr getWindowReversalAttr();
  ::llvm::Optional< ::mlir::DenseElementsAttr > getWindowReversal();
  ::mlir::mhlo::ConvDimensionNumbersAttr getDimensionNumbersAttr();
  ::mlir::mhlo::ConvDimensionNumbersAttr getDimensionNumbers();
  ::mlir::IntegerAttr getFeatureGroupCountAttr();
  uint64_t getFeatureGroupCount();
  ::mlir::IntegerAttr getBatchGroupCountAttr();
  uint64_t getBatchGroupCount();
  ::mlir::ArrayAttr getPrecisionConfigAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getPrecisionConfig();
  ::mlir::FloatAttr getResultScaleAttr();
  ::llvm::APFloat getResultScale();
  ::mlir::lmhlo_gpu::ActivationAttr getActivationModeAttr();
  ::mlir::lmhlo_gpu::Activation getActivationMode();
  ::mlir::FloatAttr getSideInputScaleAttr();
  ::llvm::APFloat getSideInputScale();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr getBackendConfigAttr();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr getBackendConfig();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class ConvForwardFusedSideInputOp : public ::mlir::Op<ConvForwardFusedSideInputOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<6>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ConvForwardFusedSideInputOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("activation_mode"), ::llvm::StringRef("backend_config"), ::llvm::StringRef("batch_group_count"), ::llvm::StringRef("dimension_numbers"), ::llvm::StringRef("feature_group_count"), ::llvm::StringRef("lhs_dilation"), ::llvm::StringRef("padding"), ::llvm::StringRef("precision_config"), ::llvm::StringRef("result_scale"), ::llvm::StringRef("rhs_dilation"), ::llvm::StringRef("side_input_scale"), ::llvm::StringRef("window_reversal"), ::llvm::StringRef("window_strides")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getActivationModeAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getActivationModeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getBackendConfigAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getBackendConfigAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getBatchGroupCountAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getBatchGroupCountAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getDimensionNumbersAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getDimensionNumbersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getFeatureGroupCountAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getFeatureGroupCountAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr getLhsDilationAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr getLhsDilationAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  ::mlir::StringAttr getPaddingAttrName() {
    return getAttributeNameForIndex(6);
  }

  static ::mlir::StringAttr getPaddingAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 6);
  }

  ::mlir::StringAttr getPrecisionConfigAttrName() {
    return getAttributeNameForIndex(7);
  }

  static ::mlir::StringAttr getPrecisionConfigAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 7);
  }

  ::mlir::StringAttr getResultScaleAttrName() {
    return getAttributeNameForIndex(8);
  }

  static ::mlir::StringAttr getResultScaleAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 8);
  }

  ::mlir::StringAttr getRhsDilationAttrName() {
    return getAttributeNameForIndex(9);
  }

  static ::mlir::StringAttr getRhsDilationAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 9);
  }

  ::mlir::StringAttr getSideInputScaleAttrName() {
    return getAttributeNameForIndex(10);
  }

  static ::mlir::StringAttr getSideInputScaleAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 10);
  }

  ::mlir::StringAttr getWindowReversalAttrName() {
    return getAttributeNameForIndex(11);
  }

  static ::mlir::StringAttr getWindowReversalAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 11);
  }

  ::mlir::StringAttr getWindowStridesAttrName() {
    return getAttributeNameForIndex(12);
  }

  static ::mlir::StringAttr getWindowStridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 12);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("lmhlo_gpu.conv_forward_fused_with_side_input");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::MemRefType> getInput();
  ::mlir::TypedValue<::mlir::MemRefType> getFilter();
  ::mlir::TypedValue<::mlir::MemRefType> getBias();
  ::mlir::TypedValue<::mlir::MemRefType> getSideInput();
  ::mlir::TypedValue<::mlir::MemRefType> getOutput();
  ::mlir::TypedValue<::mlir::MemRefType> getScratch();
  ::mlir::MutableOperandRange getInputMutable();
  ::mlir::MutableOperandRange getFilterMutable();
  ::mlir::MutableOperandRange getBiasMutable();
  ::mlir::MutableOperandRange getSideInputMutable();
  ::mlir::MutableOperandRange getOutputMutable();
  ::mlir::MutableOperandRange getScratchMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::DenseIntElementsAttr getWindowStridesAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > getWindowStrides();
  ::mlir::DenseIntElementsAttr getPaddingAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > getPadding();
  ::mlir::DenseIntElementsAttr getLhsDilationAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > getLhsDilation();
  ::mlir::DenseIntElementsAttr getRhsDilationAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > getRhsDilation();
  ::mlir::DenseElementsAttr getWindowReversalAttr();
  ::llvm::Optional< ::mlir::DenseElementsAttr > getWindowReversal();
  ::mlir::mhlo::ConvDimensionNumbersAttr getDimensionNumbersAttr();
  ::mlir::mhlo::ConvDimensionNumbersAttr getDimensionNumbers();
  ::mlir::IntegerAttr getFeatureGroupCountAttr();
  uint64_t getFeatureGroupCount();
  ::mlir::IntegerAttr getBatchGroupCountAttr();
  uint64_t getBatchGroupCount();
  ::mlir::ArrayAttr getPrecisionConfigAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getPrecisionConfig();
  ::mlir::FloatAttr getResultScaleAttr();
  ::llvm::APFloat getResultScale();
  ::mlir::lmhlo_gpu::ActivationAttr getActivationModeAttr();
  ::mlir::lmhlo_gpu::Activation getActivationMode();
  ::mlir::FloatAttr getSideInputScaleAttr();
  ::llvm::APFloat getSideInputScale();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr getBackendConfigAttr();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr getBackendConfig();
  void setWindowStridesAttr(::mlir::DenseIntElementsAttr attr);
  void setPaddingAttr(::mlir::DenseIntElementsAttr attr);
  void setLhsDilationAttr(::mlir::DenseIntElementsAttr attr);
  void setRhsDilationAttr(::mlir::DenseIntElementsAttr attr);
  void setWindowReversalAttr(::mlir::DenseElementsAttr attr);
  void setDimensionNumbersAttr(::mlir::mhlo::ConvDimensionNumbersAttr attr);
  void setFeatureGroupCountAttr(::mlir::IntegerAttr attr);
  void setBatchGroupCountAttr(::mlir::IntegerAttr attr);
  void setPrecisionConfigAttr(::mlir::ArrayAttr attr);
  void setResultScaleAttr(::mlir::FloatAttr attr);
  void setActivationModeAttr(::mlir::lmhlo_gpu::ActivationAttr attr);
  void setSideInputScaleAttr(::mlir::FloatAttr attr);
  void setBackendConfigAttr(::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr attr);
  ::mlir::Attribute removeWindow_stridesAttr();
  ::mlir::Attribute removePaddingAttr();
  ::mlir::Attribute removeLhs_dilationAttr();
  ::mlir::Attribute removeRhs_dilationAttr();
  ::mlir::Attribute removeWindow_reversalAttr();
  ::mlir::Attribute removePrecision_configAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value bias, ::mlir::Value side_input, ::mlir::Value output, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers, ::mlir::IntegerAttr feature_group_count, ::mlir::IntegerAttr batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::mlir::FloatAttr result_scale, ::mlir::lmhlo_gpu::ActivationAttr activation_mode, ::mlir::FloatAttr side_input_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value bias, ::mlir::Value side_input, ::mlir::Value output, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers, ::mlir::IntegerAttr feature_group_count, ::mlir::IntegerAttr batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::mlir::FloatAttr result_scale, ::mlir::lmhlo_gpu::ActivationAttr activation_mode, ::mlir::FloatAttr side_input_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value bias, ::mlir::Value side_input, ::mlir::Value output, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers, uint64_t feature_group_count, uint64_t batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::llvm::APFloat result_scale, ::mlir::lmhlo_gpu::Activation activation_mode, ::llvm::APFloat side_input_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value bias, ::mlir::Value side_input, ::mlir::Value output, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers, uint64_t feature_group_count, uint64_t batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::llvm::APFloat result_scale, ::mlir::lmhlo_gpu::Activation activation_mode, ::llvm::APFloat side_input_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr backend_config);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 13 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
};
} // namespace lmhlo_gpu
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::lmhlo_gpu::ConvForwardFusedSideInputOp)

namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::ConvForwardOp declarations
//===----------------------------------------------------------------------===//

class ConvForwardOpAdaptor {
public:
  ConvForwardOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  ConvForwardOpAdaptor(ConvForwardOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getInput();
  ::mlir::Value getFilter();
  ::mlir::Value getOutput();
  ::mlir::Value getScratch();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr getWindowStridesAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > getWindowStrides();
  ::mlir::DenseIntElementsAttr getPaddingAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > getPadding();
  ::mlir::DenseIntElementsAttr getLhsDilationAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > getLhsDilation();
  ::mlir::DenseIntElementsAttr getRhsDilationAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > getRhsDilation();
  ::mlir::DenseElementsAttr getWindowReversalAttr();
  ::llvm::Optional< ::mlir::DenseElementsAttr > getWindowReversal();
  ::mlir::mhlo::ConvDimensionNumbersAttr getDimensionNumbersAttr();
  ::mlir::mhlo::ConvDimensionNumbersAttr getDimensionNumbers();
  ::mlir::IntegerAttr getFeatureGroupCountAttr();
  uint64_t getFeatureGroupCount();
  ::mlir::IntegerAttr getBatchGroupCountAttr();
  uint64_t getBatchGroupCount();
  ::mlir::ArrayAttr getPrecisionConfigAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getPrecisionConfig();
  ::mlir::FloatAttr getResultScaleAttr();
  ::llvm::APFloat getResultScale();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr getBackendConfigAttr();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr getBackendConfig();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class ConvForwardOp : public ::mlir::Op<ConvForwardOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<4>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ConvForwardOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("backend_config"), ::llvm::StringRef("batch_group_count"), ::llvm::StringRef("dimension_numbers"), ::llvm::StringRef("feature_group_count"), ::llvm::StringRef("lhs_dilation"), ::llvm::StringRef("padding"), ::llvm::StringRef("precision_config"), ::llvm::StringRef("result_scale"), ::llvm::StringRef("rhs_dilation"), ::llvm::StringRef("window_reversal"), ::llvm::StringRef("window_strides")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getBackendConfigAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getBackendConfigAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getBatchGroupCountAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getBatchGroupCountAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getDimensionNumbersAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getDimensionNumbersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getFeatureGroupCountAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getFeatureGroupCountAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getLhsDilationAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getLhsDilationAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr getPaddingAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr getPaddingAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  ::mlir::StringAttr getPrecisionConfigAttrName() {
    return getAttributeNameForIndex(6);
  }

  static ::mlir::StringAttr getPrecisionConfigAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 6);
  }

  ::mlir::StringAttr getResultScaleAttrName() {
    return getAttributeNameForIndex(7);
  }

  static ::mlir::StringAttr getResultScaleAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 7);
  }

  ::mlir::StringAttr getRhsDilationAttrName() {
    return getAttributeNameForIndex(8);
  }

  static ::mlir::StringAttr getRhsDilationAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 8);
  }

  ::mlir::StringAttr getWindowReversalAttrName() {
    return getAttributeNameForIndex(9);
  }

  static ::mlir::StringAttr getWindowReversalAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 9);
  }

  ::mlir::StringAttr getWindowStridesAttrName() {
    return getAttributeNameForIndex(10);
  }

  static ::mlir::StringAttr getWindowStridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 10);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("lmhlo_gpu.conv_forward");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::MemRefType> getInput();
  ::mlir::TypedValue<::mlir::MemRefType> getFilter();
  ::mlir::TypedValue<::mlir::MemRefType> getOutput();
  ::mlir::TypedValue<::mlir::MemRefType> getScratch();
  ::mlir::MutableOperandRange getInputMutable();
  ::mlir::MutableOperandRange getFilterMutable();
  ::mlir::MutableOperandRange getOutputMutable();
  ::mlir::MutableOperandRange getScratchMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::DenseIntElementsAttr getWindowStridesAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > getWindowStrides();
  ::mlir::DenseIntElementsAttr getPaddingAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > getPadding();
  ::mlir::DenseIntElementsAttr getLhsDilationAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > getLhsDilation();
  ::mlir::DenseIntElementsAttr getRhsDilationAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > getRhsDilation();
  ::mlir::DenseElementsAttr getWindowReversalAttr();
  ::llvm::Optional< ::mlir::DenseElementsAttr > getWindowReversal();
  ::mlir::mhlo::ConvDimensionNumbersAttr getDimensionNumbersAttr();
  ::mlir::mhlo::ConvDimensionNumbersAttr getDimensionNumbers();
  ::mlir::IntegerAttr getFeatureGroupCountAttr();
  uint64_t getFeatureGroupCount();
  ::mlir::IntegerAttr getBatchGroupCountAttr();
  uint64_t getBatchGroupCount();
  ::mlir::ArrayAttr getPrecisionConfigAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getPrecisionConfig();
  ::mlir::FloatAttr getResultScaleAttr();
  ::llvm::APFloat getResultScale();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr getBackendConfigAttr();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr getBackendConfig();
  void setWindowStridesAttr(::mlir::DenseIntElementsAttr attr);
  void setPaddingAttr(::mlir::DenseIntElementsAttr attr);
  void setLhsDilationAttr(::mlir::DenseIntElementsAttr attr);
  void setRhsDilationAttr(::mlir::DenseIntElementsAttr attr);
  void setWindowReversalAttr(::mlir::DenseElementsAttr attr);
  void setDimensionNumbersAttr(::mlir::mhlo::ConvDimensionNumbersAttr attr);
  void setFeatureGroupCountAttr(::mlir::IntegerAttr attr);
  void setBatchGroupCountAttr(::mlir::IntegerAttr attr);
  void setPrecisionConfigAttr(::mlir::ArrayAttr attr);
  void setResultScaleAttr(::mlir::FloatAttr attr);
  void setBackendConfigAttr(::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr attr);
  ::mlir::Attribute removeWindow_stridesAttr();
  ::mlir::Attribute removePaddingAttr();
  ::mlir::Attribute removeLhs_dilationAttr();
  ::mlir::Attribute removeRhs_dilationAttr();
  ::mlir::Attribute removeWindow_reversalAttr();
  ::mlir::Attribute removePrecision_configAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value output, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers, ::mlir::IntegerAttr feature_group_count, ::mlir::IntegerAttr batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::mlir::FloatAttr result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value output, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers, ::mlir::IntegerAttr feature_group_count, ::mlir::IntegerAttr batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::mlir::FloatAttr result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value output, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers, uint64_t feature_group_count, uint64_t batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::llvm::APFloat result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value output, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers, uint64_t feature_group_count, uint64_t batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::llvm::APFloat result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfigAttr backend_config);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 11 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
};
} // namespace lmhlo_gpu
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::lmhlo_gpu::ConvForwardOp)

namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::CublasLtMatmulOp declarations
//===----------------------------------------------------------------------===//

class CublasLtMatmulOpAdaptor {
public:
  CublasLtMatmulOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  CublasLtMatmulOpAdaptor(CublasLtMatmulOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getA();
  ::mlir::Value getB();
  ::mlir::Value getC();
  ::mlir::Value getD();
  ::mlir::Value getBias();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::mhlo::DotDimensionNumbersAttr getDotDimensionNumbersAttr();
  ::mlir::mhlo::DotDimensionNumbersAttr getDotDimensionNumbers();
  ::mlir::ArrayAttr getPrecisionConfigAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getPrecisionConfig();
  ::mlir::FloatAttr getAlphaRealAttr();
  ::llvm::APFloat getAlphaReal();
  ::mlir::FloatAttr getAlphaImagAttr();
  ::llvm::APFloat getAlphaImag();
  ::mlir::FloatAttr getBetaAttr();
  ::llvm::APFloat getBeta();
  ::mlir::lmhlo_gpu::CublasLtMatmulEpilogueAttr getEpilogueAttr();
  ::mlir::lmhlo_gpu::CublasLtMatmulEpilogue getEpilogue();
  ::mlir::IntegerAttr getAlgorithmAttr();
  uint64_t getAlgorithm();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class CublasLtMatmulOp : public ::mlir::Op<CublasLtMatmulOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<4>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = CublasLtMatmulOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("algorithm"), ::llvm::StringRef("alpha_imag"), ::llvm::StringRef("alpha_real"), ::llvm::StringRef("beta"), ::llvm::StringRef("dot_dimension_numbers"), ::llvm::StringRef("epilogue"), ::llvm::StringRef("precision_config")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getAlgorithmAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getAlgorithmAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getAlphaImagAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getAlphaImagAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getAlphaRealAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getAlphaRealAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getBetaAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getBetaAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getDotDimensionNumbersAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getDotDimensionNumbersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr getEpilogueAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr getEpilogueAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  ::mlir::StringAttr getPrecisionConfigAttrName() {
    return getAttributeNameForIndex(6);
  }

  static ::mlir::StringAttr getPrecisionConfigAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 6);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("lmhlo_gpu.cublas.lt.matmul");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::MemRefType> getA();
  ::mlir::TypedValue<::mlir::MemRefType> getB();
  ::mlir::TypedValue<::mlir::MemRefType> getC();
  ::mlir::TypedValue<::mlir::MemRefType> getD();
  ::mlir::TypedValue<::mlir::MemRefType> getBias();
  ::mlir::MutableOperandRange getAMutable();
  ::mlir::MutableOperandRange getBMutable();
  ::mlir::MutableOperandRange getCMutable();
  ::mlir::MutableOperandRange getDMutable();
  ::mlir::MutableOperandRange getBiasMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::mhlo::DotDimensionNumbersAttr getDotDimensionNumbersAttr();
  ::mlir::mhlo::DotDimensionNumbersAttr getDotDimensionNumbers();
  ::mlir::ArrayAttr getPrecisionConfigAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getPrecisionConfig();
  ::mlir::FloatAttr getAlphaRealAttr();
  ::llvm::APFloat getAlphaReal();
  ::mlir::FloatAttr getAlphaImagAttr();
  ::llvm::APFloat getAlphaImag();
  ::mlir::FloatAttr getBetaAttr();
  ::llvm::APFloat getBeta();
  ::mlir::lmhlo_gpu::CublasLtMatmulEpilogueAttr getEpilogueAttr();
  ::mlir::lmhlo_gpu::CublasLtMatmulEpilogue getEpilogue();
  ::mlir::IntegerAttr getAlgorithmAttr();
  uint64_t getAlgorithm();
  void setDotDimensionNumbersAttr(::mlir::mhlo::DotDimensionNumbersAttr attr);
  void setPrecisionConfigAttr(::mlir::ArrayAttr attr);
  void setAlphaRealAttr(::mlir::FloatAttr attr);
  void setAlphaImagAttr(::mlir::FloatAttr attr);
  void setBetaAttr(::mlir::FloatAttr attr);
  void setEpilogueAttr(::mlir::lmhlo_gpu::CublasLtMatmulEpilogueAttr attr);
  void setAlgorithmAttr(::mlir::IntegerAttr attr);
  ::mlir::Attribute removePrecision_configAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value a, ::mlir::Value b, ::mlir::Value c, ::mlir::Value d, /*optional*/::mlir::Value bias, ::mlir::mhlo::DotDimensionNumbersAttr dot_dimension_numbers, /*optional*/::mlir::ArrayAttr precision_config, ::mlir::FloatAttr alpha_real, ::mlir::FloatAttr alpha_imag, ::mlir::FloatAttr beta, ::mlir::lmhlo_gpu::CublasLtMatmulEpilogueAttr epilogue, ::mlir::IntegerAttr algorithm);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b, ::mlir::Value c, ::mlir::Value d, /*optional*/::mlir::Value bias, ::mlir::mhlo::DotDimensionNumbersAttr dot_dimension_numbers, /*optional*/::mlir::ArrayAttr precision_config, ::mlir::FloatAttr alpha_real, ::mlir::FloatAttr alpha_imag, ::mlir::FloatAttr beta, ::mlir::lmhlo_gpu::CublasLtMatmulEpilogueAttr epilogue, ::mlir::IntegerAttr algorithm);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value a, ::mlir::Value b, ::mlir::Value c, ::mlir::Value d, /*optional*/::mlir::Value bias, ::mlir::mhlo::DotDimensionNumbersAttr dot_dimension_numbers, /*optional*/::mlir::ArrayAttr precision_config, ::llvm::APFloat alpha_real, ::llvm::APFloat alpha_imag, ::llvm::APFloat beta, ::mlir::lmhlo_gpu::CublasLtMatmulEpilogue epilogue, uint64_t algorithm);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b, ::mlir::Value c, ::mlir::Value d, /*optional*/::mlir::Value bias, ::mlir::mhlo::DotDimensionNumbersAttr dot_dimension_numbers, /*optional*/::mlir::ArrayAttr precision_config, ::llvm::APFloat alpha_real, ::llvm::APFloat alpha_imag, ::llvm::APFloat beta, ::mlir::lmhlo_gpu::CublasLtMatmulEpilogue epilogue, uint64_t algorithm);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 7 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
};
} // namespace lmhlo_gpu
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::lmhlo_gpu::CublasLtMatmulOp)

namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::GEMMOp declarations
//===----------------------------------------------------------------------===//

class GEMMOpAdaptor {
public:
  GEMMOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  GEMMOpAdaptor(GEMMOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getA();
  ::mlir::Value getB();
  ::mlir::Value getC();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::mhlo::DotDimensionNumbersAttr getDotDimensionNumbersAttr();
  ::mlir::mhlo::DotDimensionNumbersAttr getDotDimensionNumbers();
  ::mlir::ArrayAttr getPrecisionConfigAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getPrecisionConfig();
  ::mlir::FloatAttr getAlphaRealAttr();
  ::llvm::APFloat getAlphaReal();
  ::mlir::FloatAttr getAlphaImagAttr();
  ::llvm::APFloat getAlphaImag();
  ::mlir::FloatAttr getBetaAttr();
  ::llvm::APFloat getBeta();
  ::mlir::IntegerAttr getAlgorithmAttr();
  ::llvm::Optional<uint64_t> getAlgorithm();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class GEMMOp : public ::mlir::Op<GEMMOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<3>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = GEMMOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("algorithm"), ::llvm::StringRef("alpha_imag"), ::llvm::StringRef("alpha_real"), ::llvm::StringRef("beta"), ::llvm::StringRef("dot_dimension_numbers"), ::llvm::StringRef("precision_config")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getAlgorithmAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getAlgorithmAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getAlphaImagAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getAlphaImagAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getAlphaRealAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getAlphaRealAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getBetaAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getBetaAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getDotDimensionNumbersAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getDotDimensionNumbersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr getPrecisionConfigAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr getPrecisionConfigAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("lmhlo_gpu.gemm");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::MemRefType> getA();
  ::mlir::TypedValue<::mlir::MemRefType> getB();
  ::mlir::TypedValue<::mlir::MemRefType> getC();
  ::mlir::MutableOperandRange getAMutable();
  ::mlir::MutableOperandRange getBMutable();
  ::mlir::MutableOperandRange getCMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::mhlo::DotDimensionNumbersAttr getDotDimensionNumbersAttr();
  ::mlir::mhlo::DotDimensionNumbersAttr getDotDimensionNumbers();
  ::mlir::ArrayAttr getPrecisionConfigAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getPrecisionConfig();
  ::mlir::FloatAttr getAlphaRealAttr();
  ::llvm::APFloat getAlphaReal();
  ::mlir::FloatAttr getAlphaImagAttr();
  ::llvm::APFloat getAlphaImag();
  ::mlir::FloatAttr getBetaAttr();
  ::llvm::APFloat getBeta();
  ::mlir::IntegerAttr getAlgorithmAttr();
  ::llvm::Optional<uint64_t> getAlgorithm();
  void setDotDimensionNumbersAttr(::mlir::mhlo::DotDimensionNumbersAttr attr);
  void setPrecisionConfigAttr(::mlir::ArrayAttr attr);
  void setAlphaRealAttr(::mlir::FloatAttr attr);
  void setAlphaImagAttr(::mlir::FloatAttr attr);
  void setBetaAttr(::mlir::FloatAttr attr);
  void setAlgorithmAttr(::mlir::IntegerAttr attr);
  ::mlir::Attribute removePrecision_configAttr();
  ::mlir::Attribute removeAlgorithmAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value a, ::mlir::Value b, ::mlir::Value c, ::mlir::mhlo::DotDimensionNumbersAttr dot_dimension_numbers, /*optional*/::mlir::ArrayAttr precision_config, ::mlir::FloatAttr alpha_real, ::mlir::FloatAttr alpha_imag, ::mlir::FloatAttr beta, /*optional*/::mlir::IntegerAttr algorithm);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b, ::mlir::Value c, ::mlir::mhlo::DotDimensionNumbersAttr dot_dimension_numbers, /*optional*/::mlir::ArrayAttr precision_config, ::mlir::FloatAttr alpha_real, ::mlir::FloatAttr alpha_imag, ::mlir::FloatAttr beta, /*optional*/::mlir::IntegerAttr algorithm);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value a, ::mlir::Value b, ::mlir::Value c, ::mlir::mhlo::DotDimensionNumbersAttr dot_dimension_numbers, /*optional*/::mlir::ArrayAttr precision_config, ::llvm::APFloat alpha_real, ::llvm::APFloat alpha_imag, ::llvm::APFloat beta, /*optional*/::mlir::IntegerAttr algorithm);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b, ::mlir::Value c, ::mlir::mhlo::DotDimensionNumbersAttr dot_dimension_numbers, /*optional*/::mlir::ArrayAttr precision_config, ::llvm::APFloat alpha_real, ::llvm::APFloat alpha_imag, ::llvm::APFloat beta, /*optional*/::mlir::IntegerAttr algorithm);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 6 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
};
} // namespace lmhlo_gpu
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::lmhlo_gpu::GEMMOp)


#endif  // GET_OP_CLASSES

