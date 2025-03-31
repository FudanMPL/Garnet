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
namespace TF {
class _ArrayToListOp;
} // namespace TF
} // namespace mlir
namespace mlir {
namespace TF {
class _EagerConstOp;
} // namespace TF
} // namespace mlir
namespace mlir {
namespace TF {
class _FusedBatchNormExOp;
} // namespace TF
} // namespace mlir
namespace mlir {
namespace TF {
class _FusedConv2DOp;
} // namespace TF
} // namespace mlir
namespace mlir {
namespace TF {
class _FusedMatMulOp;
} // namespace TF
} // namespace mlir
namespace mlir {
namespace TF {
class _HostRecvOp;
} // namespace TF
} // namespace mlir
namespace mlir {
namespace TF {
class _HostSendOp;
} // namespace TF
} // namespace mlir
namespace mlir {
namespace TF {
class _InternalTestMustExecuteTrait_;
} // namespace TF
} // namespace mlir
namespace mlir {
namespace TF {
class _InternalTestNonResourceValueSideEffects_;
} // namespace TF
} // namespace mlir
namespace mlir {
namespace TF {
class _ListToArrayOp;
} // namespace TF
} // namespace mlir
namespace mlir {
namespace TF {
class _RecvOp;
} // namespace TF
} // namespace mlir
namespace mlir {
namespace TF {
class _SendOp;
} // namespace TF
} // namespace mlir
namespace mlir {
namespace TF {
class _TPUCompileMlirOp;
} // namespace TF
} // namespace mlir
namespace mlir {
namespace TF {
class _TPUCompileMlirPlaceholderProgramKeyOp;
} // namespace TF
} // namespace mlir
namespace mlir {
namespace TF {
class _TPUDeviceOrdinalPlaceholderOp;
} // namespace TF
} // namespace mlir
namespace mlir {
namespace TF {
class _UnaryOpsCompositionOp;
} // namespace TF
} // namespace mlir
namespace mlir {
namespace TF {
class _XlaHostComputeMlirOp;
} // namespace TF
} // namespace mlir
namespace mlir {
namespace TF {
class _XlaRecvAtHostOp;
} // namespace TF
} // namespace mlir
namespace mlir {
namespace TF {
class _XlaRecvAtHostV2Op;
} // namespace TF
} // namespace mlir
namespace mlir {
namespace TF {
class _XlaSendFromHostOp;
} // namespace TF
} // namespace mlir
namespace mlir {
namespace TF {
class _XlaSendFromHostV2Op;
} // namespace TF
} // namespace mlir
#endif

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace TF {

//===----------------------------------------------------------------------===//
// ::mlir::TF::_ArrayToListOp declarations
//===----------------------------------------------------------------------===//

class _ArrayToListOpAdaptor {
public:
  _ArrayToListOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  _ArrayToListOpAdaptor(_ArrayToListOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange input();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class _ArrayToListOp : public ::mlir::Op<_ArrayToListOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::DerivedAttributeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = _ArrayToListOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("N"), ::llvm::StringRef("T"), ::llvm::StringRef("out_types")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr NAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr NAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr TAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr TAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr out_typesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr out_typesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tf._ArrayToList");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range input();
  ::mlir::MutableOperandRange inputMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range output();
  size_t N();
  ::mlir::Type T();
  mlir::ResultElementTypeRange out_types();
  static bool isDerivedAttribute(::llvm::StringRef name);
  ::mlir::DictionaryAttr materializeDerivedAttributes();
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 3 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
};
} // namespace TF
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::TF::_ArrayToListOp)

namespace mlir {
namespace TF {

//===----------------------------------------------------------------------===//
// ::mlir::TF::_EagerConstOp declarations
//===----------------------------------------------------------------------===//

class _EagerConstOpAdaptor {
public:
  _EagerConstOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  _EagerConstOpAdaptor(_EagerConstOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value input();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class _EagerConstOp : public ::mlir::Op<_EagerConstOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::DerivedAttributeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = _EagerConstOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("T")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr TAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr TAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tf._EagerConst");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::TensorType> input();
  ::mlir::MutableOperandRange inputMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::TensorType> output();
  ::mlir::Type T();
  static bool isDerivedAttribute(::llvm::StringRef name);
  ::mlir::DictionaryAttr materializeDerivedAttributes();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input);
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
} // namespace TF
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::TF::_EagerConstOp)

namespace mlir {
namespace TF {

//===----------------------------------------------------------------------===//
// ::mlir::TF::_FusedBatchNormExOp declarations
//===----------------------------------------------------------------------===//

class _FusedBatchNormExOpAdaptor {
public:
  _FusedBatchNormExOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  _FusedBatchNormExOpAdaptor(_FusedBatchNormExOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value x();
  ::mlir::Value scale();
  ::mlir::Value offset();
  ::mlir::Value mean();
  ::mlir::Value variance();
  ::mlir::ValueRange side_input();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::FloatAttr epsilonAttr();
  ::llvm::APFloat epsilon();
  ::mlir::FloatAttr exponential_avg_factorAttr();
  ::llvm::APFloat exponential_avg_factor();
  ::mlir::StringAttr activation_modeAttr();
  ::llvm::StringRef activation_mode();
  ::mlir::StringAttr data_formatAttr();
  ::llvm::StringRef data_format();
  ::mlir::BoolAttr is_trainingAttr();
  bool is_training();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class _FusedBatchNormExOp : public ::mlir::Op<_FusedBatchNormExOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::NResults<6>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<5>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpAsmOpInterface::Trait, ::mlir::DerivedAttributeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = _FusedBatchNormExOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("T"), ::llvm::StringRef("U"), ::llvm::StringRef("activation_mode"), ::llvm::StringRef("data_format"), ::llvm::StringRef("epsilon"), ::llvm::StringRef("exponential_avg_factor"), ::llvm::StringRef("is_training"), ::llvm::StringRef("num_side_inputs")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr TAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr TAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr UAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr UAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr activation_modeAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr activation_modeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr data_formatAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr data_formatAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr epsilonAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr epsilonAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr exponential_avg_factorAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr exponential_avg_factorAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  ::mlir::StringAttr is_trainingAttrName() {
    return getAttributeNameForIndex(6);
  }

  static ::mlir::StringAttr is_trainingAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 6);
  }

  ::mlir::StringAttr num_side_inputsAttrName() {
    return getAttributeNameForIndex(7);
  }

  static ::mlir::StringAttr num_side_inputsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 7);
  }

  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tf._FusedBatchNormEx");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::TensorType> x();
  ::mlir::TypedValue<::mlir::TensorType> scale();
  ::mlir::TypedValue<::mlir::TensorType> offset();
  ::mlir::TypedValue<::mlir::TensorType> mean();
  ::mlir::TypedValue<::mlir::TensorType> variance();
  ::mlir::Operation::operand_range side_input();
  ::mlir::MutableOperandRange xMutable();
  ::mlir::MutableOperandRange scaleMutable();
  ::mlir::MutableOperandRange offsetMutable();
  ::mlir::MutableOperandRange meanMutable();
  ::mlir::MutableOperandRange varianceMutable();
  ::mlir::MutableOperandRange side_inputMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::TensorType> y();
  ::mlir::TypedValue<::mlir::TensorType> batch_mean();
  ::mlir::TypedValue<::mlir::TensorType> batch_variance();
  ::mlir::TypedValue<::mlir::TensorType> reserve_space_1();
  ::mlir::TypedValue<::mlir::TensorType> reserve_space_2();
  ::mlir::TypedValue<::mlir::TensorType> reserve_space_3();
  ::mlir::FloatAttr epsilonAttr();
  ::llvm::APFloat epsilon();
  ::mlir::FloatAttr exponential_avg_factorAttr();
  ::llvm::APFloat exponential_avg_factor();
  ::mlir::StringAttr activation_modeAttr();
  ::llvm::StringRef activation_mode();
  ::mlir::StringAttr data_formatAttr();
  ::llvm::StringRef data_format();
  ::mlir::BoolAttr is_trainingAttr();
  bool is_training();
  size_t num_side_inputs();
  ::mlir::Type T();
  ::mlir::Type U();
  static bool isDerivedAttribute(::llvm::StringRef name);
  ::mlir::DictionaryAttr materializeDerivedAttributes();
  void epsilonAttr(::mlir::FloatAttr attr);
  void exponential_avg_factorAttr(::mlir::FloatAttr attr);
  void activation_modeAttr(::mlir::StringAttr attr);
  void data_formatAttr(::mlir::StringAttr attr);
  void is_trainingAttr(::mlir::BoolAttr attr);
  ::mlir::Attribute removeEpsilonAttr();
  ::mlir::Attribute removeExponential_avg_factorAttr();
  ::mlir::Attribute removeActivation_modeAttr();
  ::mlir::Attribute removeData_formatAttr();
  ::mlir::Attribute removeIs_trainingAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type y, ::mlir::Type batch_mean, ::mlir::Type batch_variance, ::mlir::Type reserve_space_1, ::mlir::Type reserve_space_2, ::mlir::Type reserve_space_3, ::mlir::Value x, ::mlir::Value scale, ::mlir::Value offset, ::mlir::Value mean, ::mlir::Value variance, ::mlir::ValueRange side_input, /*optional*/::mlir::FloatAttr epsilon, /*optional*/::mlir::FloatAttr exponential_avg_factor, /*optional*/::mlir::StringAttr activation_mode, /*optional*/::mlir::StringAttr data_format, /*optional*/::mlir::BoolAttr is_training);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value x, ::mlir::Value scale, ::mlir::Value offset, ::mlir::Value mean, ::mlir::Value variance, ::mlir::ValueRange side_input, /*optional*/::mlir::FloatAttr epsilon, /*optional*/::mlir::FloatAttr exponential_avg_factor, /*optional*/::mlir::StringAttr activation_mode, /*optional*/::mlir::StringAttr data_format, /*optional*/::mlir::BoolAttr is_training);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type y, ::mlir::Type batch_mean, ::mlir::Type batch_variance, ::mlir::Type reserve_space_1, ::mlir::Type reserve_space_2, ::mlir::Type reserve_space_3, ::mlir::Value x, ::mlir::Value scale, ::mlir::Value offset, ::mlir::Value mean, ::mlir::Value variance, ::mlir::ValueRange side_input, /*optional*/::llvm::APFloat epsilon, /*optional*/::llvm::APFloat exponential_avg_factor, /*optional*/::llvm::StringRef activation_mode = "Identity", /*optional*/::llvm::StringRef data_format = "NHWC", /*optional*/bool is_training = true);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value x, ::mlir::Value scale, ::mlir::Value offset, ::mlir::Value mean, ::mlir::Value variance, ::mlir::ValueRange side_input, /*optional*/::llvm::APFloat epsilon, /*optional*/::llvm::APFloat exponential_avg_factor, /*optional*/::llvm::StringRef activation_mode = "Identity", /*optional*/::llvm::StringRef data_format = "NHWC", /*optional*/bool is_training = true);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 8 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
};
} // namespace TF
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::TF::_FusedBatchNormExOp)

namespace mlir {
namespace TF {

//===----------------------------------------------------------------------===//
// ::mlir::TF::_FusedConv2DOp declarations
//===----------------------------------------------------------------------===//

class _FusedConv2DOpAdaptor {
public:
  _FusedConv2DOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  _FusedConv2DOpAdaptor(_FusedConv2DOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value input();
  ::mlir::Value filter();
  ::mlir::ValueRange args();
  ::mlir::ValueRange host_args();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::IntegerAttr num_argsAttr();
  uint64_t num_args();
  ::mlir::ArrayAttr stridesAttr();
  ::mlir::ArrayAttr strides();
  ::mlir::StringAttr paddingAttr();
  ::llvm::StringRef padding();
  ::mlir::ArrayAttr explicit_paddingsAttr();
  ::mlir::ArrayAttr explicit_paddings();
  ::mlir::StringAttr data_formatAttr();
  ::llvm::StringRef data_format();
  ::mlir::StringAttr filter_formatAttr();
  ::llvm::StringRef filter_format();
  ::mlir::ArrayAttr dilationsAttr();
  ::mlir::ArrayAttr dilations();
  ::mlir::BoolAttr use_cudnn_on_gpuAttr();
  bool use_cudnn_on_gpu();
  ::mlir::ArrayAttr fused_opsAttr();
  ::mlir::ArrayAttr fused_ops();
  ::mlir::FloatAttr epsilonAttr();
  ::llvm::APFloat epsilon();
  ::mlir::FloatAttr leakyrelu_alphaAttr();
  ::llvm::APFloat leakyrelu_alpha();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class _FusedConv2DOp : public ::mlir::Op<_FusedConv2DOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<2>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::DerivedAttributeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = _FusedConv2DOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("T"), ::llvm::StringRef("TArgs"), ::llvm::StringRef("data_format"), ::llvm::StringRef("dilations"), ::llvm::StringRef("epsilon"), ::llvm::StringRef("explicit_paddings"), ::llvm::StringRef("filter_format"), ::llvm::StringRef("fused_ops"), ::llvm::StringRef("leakyrelu_alpha"), ::llvm::StringRef("num_args"), ::llvm::StringRef("num_host_args"), ::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("padding"), ::llvm::StringRef("strides"), ::llvm::StringRef("use_cudnn_on_gpu")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr TAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr TAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr TArgsAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr TArgsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr data_formatAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr data_formatAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr dilationsAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr dilationsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr epsilonAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr epsilonAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr explicit_paddingsAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr explicit_paddingsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  ::mlir::StringAttr filter_formatAttrName() {
    return getAttributeNameForIndex(6);
  }

  static ::mlir::StringAttr filter_formatAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 6);
  }

  ::mlir::StringAttr fused_opsAttrName() {
    return getAttributeNameForIndex(7);
  }

  static ::mlir::StringAttr fused_opsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 7);
  }

  ::mlir::StringAttr leakyrelu_alphaAttrName() {
    return getAttributeNameForIndex(8);
  }

  static ::mlir::StringAttr leakyrelu_alphaAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 8);
  }

  ::mlir::StringAttr num_argsAttrName() {
    return getAttributeNameForIndex(9);
  }

  static ::mlir::StringAttr num_argsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 9);
  }

  ::mlir::StringAttr num_host_argsAttrName() {
    return getAttributeNameForIndex(10);
  }

  static ::mlir::StringAttr num_host_argsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 10);
  }

  ::mlir::StringAttr operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(11);
  }

  static ::mlir::StringAttr operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 11);
  }

  ::mlir::StringAttr paddingAttrName() {
    return getAttributeNameForIndex(12);
  }

  static ::mlir::StringAttr paddingAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 12);
  }

  ::mlir::StringAttr stridesAttrName() {
    return getAttributeNameForIndex(13);
  }

  static ::mlir::StringAttr stridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 13);
  }

  ::mlir::StringAttr use_cudnn_on_gpuAttrName() {
    return getAttributeNameForIndex(14);
  }

  static ::mlir::StringAttr use_cudnn_on_gpuAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 14);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tf._FusedConv2D");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::TensorType> input();
  ::mlir::TypedValue<::mlir::TensorType> filter();
  ::mlir::Operation::operand_range args();
  ::mlir::Operation::operand_range host_args();
  ::mlir::MutableOperandRange inputMutable();
  ::mlir::MutableOperandRange filterMutable();
  ::mlir::MutableOperandRange argsMutable();
  ::mlir::MutableOperandRange host_argsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::TensorType> output();
  ::mlir::IntegerAttr num_argsAttr();
  uint64_t num_args();
  ::mlir::ArrayAttr stridesAttr();
  ::mlir::ArrayAttr strides();
  ::mlir::StringAttr paddingAttr();
  ::llvm::StringRef padding();
  ::mlir::ArrayAttr explicit_paddingsAttr();
  ::mlir::ArrayAttr explicit_paddings();
  ::mlir::StringAttr data_formatAttr();
  ::llvm::StringRef data_format();
  ::mlir::StringAttr filter_formatAttr();
  ::llvm::StringRef filter_format();
  ::mlir::ArrayAttr dilationsAttr();
  ::mlir::ArrayAttr dilations();
  ::mlir::BoolAttr use_cudnn_on_gpuAttr();
  bool use_cudnn_on_gpu();
  ::mlir::ArrayAttr fused_opsAttr();
  ::mlir::ArrayAttr fused_ops();
  ::mlir::FloatAttr epsilonAttr();
  ::llvm::APFloat epsilon();
  ::mlir::FloatAttr leakyrelu_alphaAttr();
  ::llvm::APFloat leakyrelu_alpha();
  size_t num_host_args();
  ::mlir::Type T();
  mlir::OperandElementTypeRange TArgs();
  static bool isDerivedAttribute(::llvm::StringRef name);
  ::mlir::DictionaryAttr materializeDerivedAttributes();
  void num_argsAttr(::mlir::IntegerAttr attr);
  void stridesAttr(::mlir::ArrayAttr attr);
  void paddingAttr(::mlir::StringAttr attr);
  void explicit_paddingsAttr(::mlir::ArrayAttr attr);
  void data_formatAttr(::mlir::StringAttr attr);
  void filter_formatAttr(::mlir::StringAttr attr);
  void dilationsAttr(::mlir::ArrayAttr attr);
  void use_cudnn_on_gpuAttr(::mlir::BoolAttr attr);
  void fused_opsAttr(::mlir::ArrayAttr attr);
  void epsilonAttr(::mlir::FloatAttr attr);
  void leakyrelu_alphaAttr(::mlir::FloatAttr attr);
  ::mlir::Attribute removeExplicit_paddingsAttr();
  ::mlir::Attribute removeData_formatAttr();
  ::mlir::Attribute removeFilter_formatAttr();
  ::mlir::Attribute removeDilationsAttr();
  ::mlir::Attribute removeUse_cudnn_on_gpuAttr();
  ::mlir::Attribute removeFused_opsAttr();
  ::mlir::Attribute removeEpsilonAttr();
  ::mlir::Attribute removeLeakyrelu_alphaAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::Value filter, ::mlir::ValueRange args, ::mlir::ValueRange host_args, ::mlir::IntegerAttr num_args, ::mlir::ArrayAttr strides, ::mlir::StringAttr padding, /*optional*/::mlir::ArrayAttr explicit_paddings, /*optional*/::mlir::StringAttr data_format, /*optional*/::mlir::StringAttr filter_format, /*optional*/::mlir::ArrayAttr dilations, /*optional*/::mlir::BoolAttr use_cudnn_on_gpu, /*optional*/::mlir::ArrayAttr fused_ops, /*optional*/::mlir::FloatAttr epsilon, /*optional*/::mlir::FloatAttr leakyrelu_alpha);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value filter, ::mlir::ValueRange args, ::mlir::ValueRange host_args, ::mlir::IntegerAttr num_args, ::mlir::ArrayAttr strides, ::mlir::StringAttr padding, /*optional*/::mlir::ArrayAttr explicit_paddings, /*optional*/::mlir::StringAttr data_format, /*optional*/::mlir::StringAttr filter_format, /*optional*/::mlir::ArrayAttr dilations, /*optional*/::mlir::BoolAttr use_cudnn_on_gpu, /*optional*/::mlir::ArrayAttr fused_ops, /*optional*/::mlir::FloatAttr epsilon, /*optional*/::mlir::FloatAttr leakyrelu_alpha);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::Value filter, ::mlir::ValueRange args, ::mlir::ValueRange host_args, uint64_t num_args, ::mlir::ArrayAttr strides, ::llvm::StringRef padding, /*optional*/::mlir::ArrayAttr explicit_paddings, /*optional*/::llvm::StringRef data_format, /*optional*/::llvm::StringRef filter_format, /*optional*/::mlir::ArrayAttr dilations, /*optional*/bool use_cudnn_on_gpu, /*optional*/::mlir::ArrayAttr fused_ops, /*optional*/::llvm::APFloat epsilon, /*optional*/::llvm::APFloat leakyrelu_alpha);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value filter, ::mlir::ValueRange args, ::mlir::ValueRange host_args, uint64_t num_args, ::mlir::ArrayAttr strides, ::llvm::StringRef padding, /*optional*/::mlir::ArrayAttr explicit_paddings, /*optional*/::llvm::StringRef data_format, /*optional*/::llvm::StringRef filter_format, /*optional*/::mlir::ArrayAttr dilations, /*optional*/bool use_cudnn_on_gpu, /*optional*/::mlir::ArrayAttr fused_ops, /*optional*/::llvm::APFloat epsilon, /*optional*/::llvm::APFloat leakyrelu_alpha);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 15 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
};
} // namespace TF
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::TF::_FusedConv2DOp)

namespace mlir {
namespace TF {

//===----------------------------------------------------------------------===//
// ::mlir::TF::_FusedMatMulOp declarations
//===----------------------------------------------------------------------===//

class _FusedMatMulOpAdaptor {
public:
  _FusedMatMulOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  _FusedMatMulOpAdaptor(_FusedMatMulOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value a();
  ::mlir::Value b();
  ::mlir::ValueRange args();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::BoolAttr transpose_aAttr();
  bool transpose_a();
  ::mlir::BoolAttr transpose_bAttr();
  bool transpose_b();
  ::mlir::ArrayAttr fused_opsAttr();
  ::mlir::ArrayAttr fused_ops();
  ::mlir::FloatAttr epsilonAttr();
  ::llvm::APFloat epsilon();
  ::mlir::FloatAttr leakyrelu_alphaAttr();
  ::llvm::APFloat leakyrelu_alpha();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class _FusedMatMulOp : public ::mlir::Op<_FusedMatMulOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::TF::SameOperandsAndResultElementTypeResolveRef, ::mlir::DerivedAttributeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = _FusedMatMulOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("T"), ::llvm::StringRef("epsilon"), ::llvm::StringRef("fused_ops"), ::llvm::StringRef("leakyrelu_alpha"), ::llvm::StringRef("num_args"), ::llvm::StringRef("transpose_a"), ::llvm::StringRef("transpose_b")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr TAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr TAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr epsilonAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr epsilonAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr fused_opsAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr fused_opsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr leakyrelu_alphaAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr leakyrelu_alphaAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr num_argsAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr num_argsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr transpose_aAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr transpose_aAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  ::mlir::StringAttr transpose_bAttrName() {
    return getAttributeNameForIndex(6);
  }

  static ::mlir::StringAttr transpose_bAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 6);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tf._FusedMatMul");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::TensorType> a();
  ::mlir::TypedValue<::mlir::TensorType> b();
  ::mlir::Operation::operand_range args();
  ::mlir::MutableOperandRange aMutable();
  ::mlir::MutableOperandRange bMutable();
  ::mlir::MutableOperandRange argsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::TensorType> product();
  ::mlir::BoolAttr transpose_aAttr();
  bool transpose_a();
  ::mlir::BoolAttr transpose_bAttr();
  bool transpose_b();
  ::mlir::ArrayAttr fused_opsAttr();
  ::mlir::ArrayAttr fused_ops();
  ::mlir::FloatAttr epsilonAttr();
  ::llvm::APFloat epsilon();
  ::mlir::FloatAttr leakyrelu_alphaAttr();
  ::llvm::APFloat leakyrelu_alpha();
  size_t num_args();
  ::mlir::Type T();
  static bool isDerivedAttribute(::llvm::StringRef name);
  ::mlir::DictionaryAttr materializeDerivedAttributes();
  void transpose_aAttr(::mlir::BoolAttr attr);
  void transpose_bAttr(::mlir::BoolAttr attr);
  void fused_opsAttr(::mlir::ArrayAttr attr);
  void epsilonAttr(::mlir::FloatAttr attr);
  void leakyrelu_alphaAttr(::mlir::FloatAttr attr);
  ::mlir::Attribute removeTranspose_aAttr();
  ::mlir::Attribute removeTranspose_bAttr();
  ::mlir::Attribute removeFused_opsAttr();
  ::mlir::Attribute removeEpsilonAttr();
  ::mlir::Attribute removeLeakyrelu_alphaAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type product, ::mlir::Value a, ::mlir::Value b, ::mlir::ValueRange args, /*optional*/::mlir::BoolAttr transpose_a, /*optional*/::mlir::BoolAttr transpose_b, /*optional*/::mlir::ArrayAttr fused_ops, /*optional*/::mlir::FloatAttr epsilon, /*optional*/::mlir::FloatAttr leakyrelu_alpha);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b, ::mlir::ValueRange args, /*optional*/::mlir::BoolAttr transpose_a, /*optional*/::mlir::BoolAttr transpose_b, /*optional*/::mlir::ArrayAttr fused_ops, /*optional*/::mlir::FloatAttr epsilon, /*optional*/::mlir::FloatAttr leakyrelu_alpha);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type product, ::mlir::Value a, ::mlir::Value b, ::mlir::ValueRange args, /*optional*/bool transpose_a, /*optional*/bool transpose_b, /*optional*/::mlir::ArrayAttr fused_ops, /*optional*/::llvm::APFloat epsilon, /*optional*/::llvm::APFloat leakyrelu_alpha);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b, ::mlir::ValueRange args, /*optional*/bool transpose_a, /*optional*/bool transpose_b, /*optional*/::mlir::ArrayAttr fused_ops, /*optional*/::llvm::APFloat epsilon, /*optional*/::llvm::APFloat leakyrelu_alpha);
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
} // namespace TF
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::TF::_FusedMatMulOp)

namespace mlir {
namespace TF {

//===----------------------------------------------------------------------===//
// ::mlir::TF::_HostRecvOp declarations
//===----------------------------------------------------------------------===//

class _HostRecvOpAdaptor {
public:
  _HostRecvOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  _HostRecvOpAdaptor(_HostRecvOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::StringAttr tensor_nameAttr();
  ::llvm::StringRef tensor_name();
  ::mlir::StringAttr send_deviceAttr();
  ::llvm::StringRef send_device();
  ::mlir::IntegerAttr send_device_incarnationAttr();
  uint64_t send_device_incarnation();
  ::mlir::StringAttr recv_deviceAttr();
  ::llvm::StringRef recv_device();
  ::mlir::BoolAttr client_terminatedAttr();
  bool client_terminated();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class _HostRecvOp : public ::mlir::Op<_HostRecvOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, GetResourceInstanceInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::DerivedAttributeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = _HostRecvOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("client_terminated"), ::llvm::StringRef("recv_device"), ::llvm::StringRef("send_device"), ::llvm::StringRef("send_device_incarnation"), ::llvm::StringRef("tensor_name"), ::llvm::StringRef("tensor_type")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr client_terminatedAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr client_terminatedAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr recv_deviceAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr recv_deviceAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr send_deviceAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr send_deviceAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr send_device_incarnationAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr send_device_incarnationAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr tensor_nameAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr tensor_nameAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr tensor_typeAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr tensor_typeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tf._HostRecv");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::TensorType> tensor();
  ::mlir::StringAttr tensor_nameAttr();
  ::llvm::StringRef tensor_name();
  ::mlir::StringAttr send_deviceAttr();
  ::llvm::StringRef send_device();
  ::mlir::IntegerAttr send_device_incarnationAttr();
  uint64_t send_device_incarnation();
  ::mlir::StringAttr recv_deviceAttr();
  ::llvm::StringRef recv_device();
  ::mlir::BoolAttr client_terminatedAttr();
  bool client_terminated();
  ::mlir::Type tensor_type();
  static bool isDerivedAttribute(::llvm::StringRef name);
  ::mlir::DictionaryAttr materializeDerivedAttributes();
  void tensor_nameAttr(::mlir::StringAttr attr);
  void send_deviceAttr(::mlir::StringAttr attr);
  void send_device_incarnationAttr(::mlir::IntegerAttr attr);
  void recv_deviceAttr(::mlir::StringAttr attr);
  void client_terminatedAttr(::mlir::BoolAttr attr);
  ::mlir::Attribute removeClient_terminatedAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type tensor, ::mlir::StringAttr tensor_name, ::mlir::StringAttr send_device, ::mlir::IntegerAttr send_device_incarnation, ::mlir::StringAttr recv_device, /*optional*/::mlir::BoolAttr client_terminated);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr tensor_name, ::mlir::StringAttr send_device, ::mlir::IntegerAttr send_device_incarnation, ::mlir::StringAttr recv_device, /*optional*/::mlir::BoolAttr client_terminated);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type tensor, ::llvm::StringRef tensor_name, ::llvm::StringRef send_device, uint64_t send_device_incarnation, ::llvm::StringRef recv_device, /*optional*/bool client_terminated = false);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef tensor_name, ::llvm::StringRef send_device, uint64_t send_device_incarnation, ::llvm::StringRef recv_device, /*optional*/bool client_terminated = false);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  std::string GetResourceInstanceStr();
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
} // namespace TF
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::TF::_HostRecvOp)

namespace mlir {
namespace TF {

//===----------------------------------------------------------------------===//
// ::mlir::TF::_HostSendOp declarations
//===----------------------------------------------------------------------===//

class _HostSendOpAdaptor {
public:
  _HostSendOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  _HostSendOpAdaptor(_HostSendOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value tensor();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::StringAttr tensor_nameAttr();
  ::llvm::StringRef tensor_name();
  ::mlir::StringAttr send_deviceAttr();
  ::llvm::StringRef send_device();
  ::mlir::IntegerAttr send_device_incarnationAttr();
  uint64_t send_device_incarnation();
  ::mlir::StringAttr recv_deviceAttr();
  ::llvm::StringRef recv_device();
  ::mlir::BoolAttr client_terminatedAttr();
  bool client_terminated();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class _HostSendOp : public ::mlir::Op<_HostSendOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, GetResourceInstanceInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::DerivedAttributeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = _HostSendOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("T"), ::llvm::StringRef("client_terminated"), ::llvm::StringRef("recv_device"), ::llvm::StringRef("send_device"), ::llvm::StringRef("send_device_incarnation"), ::llvm::StringRef("tensor_name")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr TAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr TAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr client_terminatedAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr client_terminatedAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr recv_deviceAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr recv_deviceAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr send_deviceAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr send_deviceAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr send_device_incarnationAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr send_device_incarnationAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr tensor_nameAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr tensor_nameAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tf._HostSend");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::TensorType> tensor();
  ::mlir::MutableOperandRange tensorMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::StringAttr tensor_nameAttr();
  ::llvm::StringRef tensor_name();
  ::mlir::StringAttr send_deviceAttr();
  ::llvm::StringRef send_device();
  ::mlir::IntegerAttr send_device_incarnationAttr();
  uint64_t send_device_incarnation();
  ::mlir::StringAttr recv_deviceAttr();
  ::llvm::StringRef recv_device();
  ::mlir::BoolAttr client_terminatedAttr();
  bool client_terminated();
  ::mlir::Type T();
  static bool isDerivedAttribute(::llvm::StringRef name);
  ::mlir::DictionaryAttr materializeDerivedAttributes();
  void tensor_nameAttr(::mlir::StringAttr attr);
  void send_deviceAttr(::mlir::StringAttr attr);
  void send_device_incarnationAttr(::mlir::IntegerAttr attr);
  void recv_deviceAttr(::mlir::StringAttr attr);
  void client_terminatedAttr(::mlir::BoolAttr attr);
  ::mlir::Attribute removeClient_terminatedAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value tensor, ::mlir::StringAttr tensor_name, ::mlir::StringAttr send_device, ::mlir::IntegerAttr send_device_incarnation, ::mlir::StringAttr recv_device, /*optional*/::mlir::BoolAttr client_terminated);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value tensor, ::mlir::StringAttr tensor_name, ::mlir::StringAttr send_device, ::mlir::IntegerAttr send_device_incarnation, ::mlir::StringAttr recv_device, /*optional*/::mlir::BoolAttr client_terminated);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value tensor, ::llvm::StringRef tensor_name, ::llvm::StringRef send_device, uint64_t send_device_incarnation, ::llvm::StringRef recv_device, /*optional*/bool client_terminated = false);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value tensor, ::llvm::StringRef tensor_name, ::llvm::StringRef send_device, uint64_t send_device_incarnation, ::llvm::StringRef recv_device, /*optional*/bool client_terminated = false);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  std::string GetResourceInstanceStr();
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
} // namespace TF
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::TF::_HostSendOp)

namespace mlir {
namespace TF {

//===----------------------------------------------------------------------===//
// ::mlir::TF::_InternalTestMustExecuteTrait_ declarations
//===----------------------------------------------------------------------===//

class _InternalTestMustExecuteTrait_Adaptor {
public:
  _InternalTestMustExecuteTrait_Adaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  _InternalTestMustExecuteTrait_Adaptor(_InternalTestMustExecuteTrait_ op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class _InternalTestMustExecuteTrait_ : public ::mlir::Op<_InternalTestMustExecuteTrait_, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = _InternalTestMustExecuteTrait_Adaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tf._InternalTestMustExecuteTrait_");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace TF
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::TF::_InternalTestMustExecuteTrait_)

namespace mlir {
namespace TF {

//===----------------------------------------------------------------------===//
// ::mlir::TF::_InternalTestNonResourceValueSideEffects_ declarations
//===----------------------------------------------------------------------===//

class _InternalTestNonResourceValueSideEffects_Adaptor {
public:
  _InternalTestNonResourceValueSideEffects_Adaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  _InternalTestNonResourceValueSideEffects_Adaptor(_InternalTestNonResourceValueSideEffects_ op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value key();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class _InternalTestNonResourceValueSideEffects_ : public ::mlir::Op<_InternalTestNonResourceValueSideEffects_, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = _InternalTestNonResourceValueSideEffects_Adaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tf._InternalTestNonResourceValueSideEffects_");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::TensorType> key();
  ::mlir::MutableOperandRange keyMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value key);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value key);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace TF
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::TF::_InternalTestNonResourceValueSideEffects_)

namespace mlir {
namespace TF {

//===----------------------------------------------------------------------===//
// ::mlir::TF::_ListToArrayOp declarations
//===----------------------------------------------------------------------===//

class _ListToArrayOpAdaptor {
public:
  _ListToArrayOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  _ListToArrayOpAdaptor(_ListToArrayOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange input();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class _ListToArrayOp : public ::mlir::Op<_ListToArrayOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::DerivedAttributeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = _ListToArrayOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("N"), ::llvm::StringRef("T"), ::llvm::StringRef("Tin")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr NAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr NAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr TAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr TAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr TinAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr TinAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tf._ListToArray");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range input();
  ::mlir::MutableOperandRange inputMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range output();
  mlir::OperandElementTypeRange Tin();
  size_t N();
  ::mlir::Type T();
  static bool isDerivedAttribute(::llvm::StringRef name);
  ::mlir::DictionaryAttr materializeDerivedAttributes();
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 3 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
};
} // namespace TF
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::TF::_ListToArrayOp)

namespace mlir {
namespace TF {

//===----------------------------------------------------------------------===//
// ::mlir::TF::_RecvOp declarations
//===----------------------------------------------------------------------===//

class _RecvOpAdaptor {
public:
  _RecvOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  _RecvOpAdaptor(_RecvOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::StringAttr tensor_nameAttr();
  ::llvm::StringRef tensor_name();
  ::mlir::StringAttr send_deviceAttr();
  ::llvm::StringRef send_device();
  ::mlir::IntegerAttr send_device_incarnationAttr();
  uint64_t send_device_incarnation();
  ::mlir::StringAttr recv_deviceAttr();
  ::llvm::StringRef recv_device();
  ::mlir::BoolAttr client_terminatedAttr();
  bool client_terminated();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class _RecvOp : public ::mlir::Op<_RecvOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, GetResourceInstanceInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::DerivedAttributeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = _RecvOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("client_terminated"), ::llvm::StringRef("recv_device"), ::llvm::StringRef("send_device"), ::llvm::StringRef("send_device_incarnation"), ::llvm::StringRef("tensor_name"), ::llvm::StringRef("tensor_type")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr client_terminatedAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr client_terminatedAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr recv_deviceAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr recv_deviceAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr send_deviceAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr send_deviceAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr send_device_incarnationAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr send_device_incarnationAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr tensor_nameAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr tensor_nameAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr tensor_typeAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr tensor_typeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tf._Recv");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::TensorType> tensor();
  ::mlir::StringAttr tensor_nameAttr();
  ::llvm::StringRef tensor_name();
  ::mlir::StringAttr send_deviceAttr();
  ::llvm::StringRef send_device();
  ::mlir::IntegerAttr send_device_incarnationAttr();
  uint64_t send_device_incarnation();
  ::mlir::StringAttr recv_deviceAttr();
  ::llvm::StringRef recv_device();
  ::mlir::BoolAttr client_terminatedAttr();
  bool client_terminated();
  ::mlir::Type tensor_type();
  static bool isDerivedAttribute(::llvm::StringRef name);
  ::mlir::DictionaryAttr materializeDerivedAttributes();
  void tensor_nameAttr(::mlir::StringAttr attr);
  void send_deviceAttr(::mlir::StringAttr attr);
  void send_device_incarnationAttr(::mlir::IntegerAttr attr);
  void recv_deviceAttr(::mlir::StringAttr attr);
  void client_terminatedAttr(::mlir::BoolAttr attr);
  ::mlir::Attribute removeClient_terminatedAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type tensor, ::mlir::StringAttr tensor_name, ::mlir::StringAttr send_device, ::mlir::IntegerAttr send_device_incarnation, ::mlir::StringAttr recv_device, /*optional*/::mlir::BoolAttr client_terminated);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr tensor_name, ::mlir::StringAttr send_device, ::mlir::IntegerAttr send_device_incarnation, ::mlir::StringAttr recv_device, /*optional*/::mlir::BoolAttr client_terminated);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type tensor, ::llvm::StringRef tensor_name, ::llvm::StringRef send_device, uint64_t send_device_incarnation, ::llvm::StringRef recv_device, /*optional*/bool client_terminated = false);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef tensor_name, ::llvm::StringRef send_device, uint64_t send_device_incarnation, ::llvm::StringRef recv_device, /*optional*/bool client_terminated = false);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  std::string GetResourceInstanceStr();
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
} // namespace TF
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::TF::_RecvOp)

namespace mlir {
namespace TF {

//===----------------------------------------------------------------------===//
// ::mlir::TF::_SendOp declarations
//===----------------------------------------------------------------------===//

class _SendOpAdaptor {
public:
  _SendOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  _SendOpAdaptor(_SendOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value tensor();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::StringAttr tensor_nameAttr();
  ::llvm::StringRef tensor_name();
  ::mlir::StringAttr send_deviceAttr();
  ::llvm::StringRef send_device();
  ::mlir::IntegerAttr send_device_incarnationAttr();
  uint64_t send_device_incarnation();
  ::mlir::StringAttr recv_deviceAttr();
  ::llvm::StringRef recv_device();
  ::mlir::BoolAttr client_terminatedAttr();
  bool client_terminated();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class _SendOp : public ::mlir::Op<_SendOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, GetResourceInstanceInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::DerivedAttributeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = _SendOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("T"), ::llvm::StringRef("client_terminated"), ::llvm::StringRef("recv_device"), ::llvm::StringRef("send_device"), ::llvm::StringRef("send_device_incarnation"), ::llvm::StringRef("tensor_name")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr TAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr TAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr client_terminatedAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr client_terminatedAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr recv_deviceAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr recv_deviceAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr send_deviceAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr send_deviceAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr send_device_incarnationAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr send_device_incarnationAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr tensor_nameAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr tensor_nameAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tf._Send");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::TensorType> tensor();
  ::mlir::MutableOperandRange tensorMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::StringAttr tensor_nameAttr();
  ::llvm::StringRef tensor_name();
  ::mlir::StringAttr send_deviceAttr();
  ::llvm::StringRef send_device();
  ::mlir::IntegerAttr send_device_incarnationAttr();
  uint64_t send_device_incarnation();
  ::mlir::StringAttr recv_deviceAttr();
  ::llvm::StringRef recv_device();
  ::mlir::BoolAttr client_terminatedAttr();
  bool client_terminated();
  ::mlir::Type T();
  static bool isDerivedAttribute(::llvm::StringRef name);
  ::mlir::DictionaryAttr materializeDerivedAttributes();
  void tensor_nameAttr(::mlir::StringAttr attr);
  void send_deviceAttr(::mlir::StringAttr attr);
  void send_device_incarnationAttr(::mlir::IntegerAttr attr);
  void recv_deviceAttr(::mlir::StringAttr attr);
  void client_terminatedAttr(::mlir::BoolAttr attr);
  ::mlir::Attribute removeClient_terminatedAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value tensor, ::mlir::StringAttr tensor_name, ::mlir::StringAttr send_device, ::mlir::IntegerAttr send_device_incarnation, ::mlir::StringAttr recv_device, /*optional*/::mlir::BoolAttr client_terminated);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value tensor, ::mlir::StringAttr tensor_name, ::mlir::StringAttr send_device, ::mlir::IntegerAttr send_device_incarnation, ::mlir::StringAttr recv_device, /*optional*/::mlir::BoolAttr client_terminated);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value tensor, ::llvm::StringRef tensor_name, ::llvm::StringRef send_device, uint64_t send_device_incarnation, ::llvm::StringRef recv_device, /*optional*/bool client_terminated = false);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value tensor, ::llvm::StringRef tensor_name, ::llvm::StringRef send_device, uint64_t send_device_incarnation, ::llvm::StringRef recv_device, /*optional*/bool client_terminated = false);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  std::string GetResourceInstanceStr();
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
} // namespace TF
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::TF::_SendOp)

namespace mlir {
namespace TF {

//===----------------------------------------------------------------------===//
// ::mlir::TF::_TPUCompileMlirOp declarations
//===----------------------------------------------------------------------===//

class _TPUCompileMlirOpAdaptor {
public:
  _TPUCompileMlirOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  _TPUCompileMlirOpAdaptor(_TPUCompileMlirOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange dynamic_shapes();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::StringAttr mlir_moduleAttr();
  ::llvm::StringRef mlir_module();
  ::mlir::StringAttr metadataAttr();
  ::llvm::StringRef metadata();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class _TPUCompileMlirOp : public ::mlir::Op<_TPUCompileMlirOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::AtLeastNResults<1>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpAsmOpInterface::Trait, ::mlir::DerivedAttributeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = _TPUCompileMlirOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("NumDynamicShapes"), ::llvm::StringRef("metadata"), ::llvm::StringRef("mlir_module"), ::llvm::StringRef("num_computations")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr NumDynamicShapesAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr NumDynamicShapesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr metadataAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr metadataAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr mlir_moduleAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr mlir_moduleAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr num_computationsAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr num_computationsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tf._TPUCompileMlir");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range dynamic_shapes();
  ::mlir::MutableOperandRange dynamic_shapesMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::TensorType> compilation_status();
  ::mlir::Operation::result_range program();
  ::mlir::StringAttr mlir_moduleAttr();
  ::llvm::StringRef mlir_module();
  ::mlir::StringAttr metadataAttr();
  ::llvm::StringRef metadata();
  size_t NumDynamicShapes();
  size_t num_computations();
  static bool isDerivedAttribute(::llvm::StringRef name);
  ::mlir::DictionaryAttr materializeDerivedAttributes();
  void mlir_moduleAttr(::mlir::StringAttr attr);
  void metadataAttr(::mlir::StringAttr attr);
  ::mlir::Attribute removeMlir_moduleAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type compilation_status, ::mlir::TypeRange program, ::mlir::ValueRange dynamic_shapes, /*optional*/::mlir::StringAttr mlir_module, ::mlir::StringAttr metadata);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange dynamic_shapes, /*optional*/::mlir::StringAttr mlir_module, ::mlir::StringAttr metadata);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type compilation_status, ::mlir::TypeRange program, ::mlir::ValueRange dynamic_shapes, /*optional*/::llvm::StringRef mlir_module, ::llvm::StringRef metadata);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange dynamic_shapes, /*optional*/::llvm::StringRef mlir_module, ::llvm::StringRef metadata);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
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
} // namespace TF
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::TF::_TPUCompileMlirOp)

namespace mlir {
namespace TF {

//===----------------------------------------------------------------------===//
// ::mlir::TF::_TPUCompileMlirPlaceholderProgramKeyOp declarations
//===----------------------------------------------------------------------===//

class _TPUCompileMlirPlaceholderProgramKeyOpAdaptor {
public:
  _TPUCompileMlirPlaceholderProgramKeyOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  _TPUCompileMlirPlaceholderProgramKeyOpAdaptor(_TPUCompileMlirPlaceholderProgramKeyOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class _TPUCompileMlirPlaceholderProgramKeyOp : public ::mlir::Op<_TPUCompileMlirPlaceholderProgramKeyOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = _TPUCompileMlirPlaceholderProgramKeyOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tf._TPUCompileMlirPlaceholderProgramKey");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::TensorType> program();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type program);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace TF
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::TF::_TPUCompileMlirPlaceholderProgramKeyOp)

namespace mlir {
namespace TF {

//===----------------------------------------------------------------------===//
// ::mlir::TF::_TPUDeviceOrdinalPlaceholderOp declarations
//===----------------------------------------------------------------------===//

class _TPUDeviceOrdinalPlaceholderOpAdaptor {
public:
  _TPUDeviceOrdinalPlaceholderOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  _TPUDeviceOrdinalPlaceholderOpAdaptor(_TPUDeviceOrdinalPlaceholderOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class _TPUDeviceOrdinalPlaceholderOp : public ::mlir::Op<_TPUDeviceOrdinalPlaceholderOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = _TPUDeviceOrdinalPlaceholderOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tf._TPUDeviceOrdinalPlaceholder");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::TensorType> device_ordinal();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type device_ordinal);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace TF
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::TF::_TPUDeviceOrdinalPlaceholderOp)

namespace mlir {
namespace TF {

//===----------------------------------------------------------------------===//
// ::mlir::TF::_UnaryOpsCompositionOp declarations
//===----------------------------------------------------------------------===//

class _UnaryOpsCompositionOpAdaptor {
public:
  _UnaryOpsCompositionOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  _UnaryOpsCompositionOpAdaptor(_UnaryOpsCompositionOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value x();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::ArrayAttr op_namesAttr();
  ::mlir::ArrayAttr op_names();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class _UnaryOpsCompositionOp : public ::mlir::Op<_UnaryOpsCompositionOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::InferTypeOpInterface::Trait, ::mlir::InferShapedTypeOpInterface::Trait, ::mlir::OpTrait::InferTensorType, ::mlir::OpTrait::TF::SameOperandsAndResultTypeResolveRef, ::mlir::DerivedAttributeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = _UnaryOpsCompositionOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("T"), ::llvm::StringRef("op_names")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr TAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr TAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr op_namesAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr op_namesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tf._UnaryOpsComposition");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::TensorType> x();
  ::mlir::MutableOperandRange xMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::TensorType> y();
  ::mlir::ArrayAttr op_namesAttr();
  ::mlir::ArrayAttr op_names();
  ::mlir::Type T();
  static bool isDerivedAttribute(::llvm::StringRef name);
  ::mlir::DictionaryAttr materializeDerivedAttributes();
  void op_namesAttr(::mlir::ArrayAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type y, ::mlir::Value x, ::mlir::ArrayAttr op_names);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value x, ::mlir::ArrayAttr op_names);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value x, ::mlir::ArrayAttr op_names);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::LogicalResult inferReturnTypeComponents(::mlir::MLIRContext*context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueShapeRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>&inferredReturnShapes);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 2 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  static bool isCompatibleReturnTypes(TypeRange inferred, TypeRange actual) {
    return ArraysAreCastCompatible(inferred, actual);
  }
};
} // namespace TF
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::TF::_UnaryOpsCompositionOp)

namespace mlir {
namespace TF {

//===----------------------------------------------------------------------===//
// ::mlir::TF::_XlaHostComputeMlirOp declarations
//===----------------------------------------------------------------------===//

class _XlaHostComputeMlirOpAdaptor {
public:
  _XlaHostComputeMlirOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  _XlaHostComputeMlirOpAdaptor(_XlaHostComputeMlirOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::StringAttr send_keyAttr();
  ::llvm::StringRef send_key();
  ::mlir::StringAttr recv_keyAttr();
  ::llvm::StringRef recv_key();
  ::mlir::StringAttr host_mlir_moduleAttr();
  ::llvm::StringRef host_mlir_module();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class _XlaHostComputeMlirOp : public ::mlir::Op<_XlaHostComputeMlirOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::DerivedAttributeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = _XlaHostComputeMlirOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("Tinputs"), ::llvm::StringRef("Toutputs"), ::llvm::StringRef("host_mlir_module"), ::llvm::StringRef("recv_key"), ::llvm::StringRef("send_key")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr TinputsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr TinputsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr ToutputsAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr ToutputsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr host_mlir_moduleAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr host_mlir_moduleAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr recv_keyAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr recv_keyAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr send_keyAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr send_keyAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tf._XlaHostComputeMlir");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::MutableOperandRange inputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range outputs();
  ::mlir::StringAttr send_keyAttr();
  ::llvm::StringRef send_key();
  ::mlir::StringAttr recv_keyAttr();
  ::llvm::StringRef recv_key();
  ::mlir::StringAttr host_mlir_moduleAttr();
  ::llvm::StringRef host_mlir_module();
  mlir::OperandElementTypeRange Tinputs();
  mlir::ResultElementTypeRange Toutputs();
  static bool isDerivedAttribute(::llvm::StringRef name);
  ::mlir::DictionaryAttr materializeDerivedAttributes();
  void send_keyAttr(::mlir::StringAttr attr);
  void recv_keyAttr(::mlir::StringAttr attr);
  void host_mlir_moduleAttr(::mlir::StringAttr attr);
  ::mlir::Attribute removeHost_mlir_moduleAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outputs, ::mlir::ValueRange inputs, ::mlir::StringAttr send_key, ::mlir::StringAttr recv_key, /*optional*/::mlir::StringAttr host_mlir_module);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outputs, ::mlir::ValueRange inputs, ::llvm::StringRef send_key, ::llvm::StringRef recv_key, /*optional*/::llvm::StringRef host_mlir_module = "");
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
    assert(index < 5 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  func::FuncOp GetHostFunc(mlir::OwningOpRef<mlir::ModuleOp>* mlir_module);
};
} // namespace TF
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::TF::_XlaHostComputeMlirOp)

namespace mlir {
namespace TF {

//===----------------------------------------------------------------------===//
// ::mlir::TF::_XlaRecvAtHostOp declarations
//===----------------------------------------------------------------------===//

class _XlaRecvAtHostOpAdaptor {
public:
  _XlaRecvAtHostOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  _XlaRecvAtHostOpAdaptor(_XlaRecvAtHostOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value dynamic_key();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::StringAttr keyAttr();
  ::llvm::StringRef key();
  ::mlir::IntegerAttr device_ordinalAttr();
  uint64_t device_ordinal();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class _XlaRecvAtHostOp : public ::mlir::Op<_XlaRecvAtHostOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, GetResourceInstanceInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::DerivedAttributeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = _XlaRecvAtHostOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("Toutputs"), ::llvm::StringRef("device_ordinal"), ::llvm::StringRef("key")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr ToutputsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr ToutputsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr device_ordinalAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr device_ordinalAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr keyAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr keyAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tf._XlaRecvAtHost");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::TensorType> dynamic_key();
  ::mlir::MutableOperandRange dynamic_keyMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range outputs();
  ::mlir::StringAttr keyAttr();
  ::llvm::StringRef key();
  ::mlir::IntegerAttr device_ordinalAttr();
  uint64_t device_ordinal();
  mlir::ResultElementTypeRange Toutputs();
  static bool isDerivedAttribute(::llvm::StringRef name);
  ::mlir::DictionaryAttr materializeDerivedAttributes();
  void keyAttr(::mlir::StringAttr attr);
  void device_ordinalAttr(::mlir::IntegerAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outputs, ::mlir::Value dynamic_key, ::mlir::StringAttr key, ::mlir::IntegerAttr device_ordinal);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outputs, ::mlir::Value dynamic_key, ::llvm::StringRef key, uint64_t device_ordinal);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  std::string GetResourceInstanceStr();
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 3 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
};
} // namespace TF
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::TF::_XlaRecvAtHostOp)

namespace mlir {
namespace TF {

//===----------------------------------------------------------------------===//
// ::mlir::TF::_XlaRecvAtHostV2Op declarations
//===----------------------------------------------------------------------===//

class _XlaRecvAtHostV2OpAdaptor {
public:
  _XlaRecvAtHostV2OpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  _XlaRecvAtHostV2OpAdaptor(_XlaRecvAtHostV2Op op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value dynamic_key();
  ::mlir::Value device_ordinal();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::StringAttr keyAttr();
  ::llvm::StringRef key();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class _XlaRecvAtHostV2Op : public ::mlir::Op<_XlaRecvAtHostV2Op, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, GetResourceInstanceInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::DerivedAttributeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = _XlaRecvAtHostV2OpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("Toutputs"), ::llvm::StringRef("key")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr ToutputsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr ToutputsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr keyAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr keyAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tf._XlaRecvAtHostV2");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::TensorType> dynamic_key();
  ::mlir::TypedValue<::mlir::TensorType> device_ordinal();
  ::mlir::MutableOperandRange dynamic_keyMutable();
  ::mlir::MutableOperandRange device_ordinalMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range outputs();
  ::mlir::StringAttr keyAttr();
  ::llvm::StringRef key();
  mlir::ResultElementTypeRange Toutputs();
  static bool isDerivedAttribute(::llvm::StringRef name);
  ::mlir::DictionaryAttr materializeDerivedAttributes();
  void keyAttr(::mlir::StringAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outputs, ::mlir::Value dynamic_key, ::mlir::Value device_ordinal, ::mlir::StringAttr key);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outputs, ::mlir::Value dynamic_key, ::mlir::Value device_ordinal, ::llvm::StringRef key);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  std::string GetResourceInstanceStr();
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 2 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
};
} // namespace TF
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::TF::_XlaRecvAtHostV2Op)

namespace mlir {
namespace TF {

//===----------------------------------------------------------------------===//
// ::mlir::TF::_XlaSendFromHostOp declarations
//===----------------------------------------------------------------------===//

class _XlaSendFromHostOpAdaptor {
public:
  _XlaSendFromHostOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  _XlaSendFromHostOpAdaptor(_XlaSendFromHostOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::Value dynamic_key();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::StringAttr keyAttr();
  ::llvm::StringRef key();
  ::mlir::IntegerAttr device_ordinalAttr();
  uint64_t device_ordinal();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class _XlaSendFromHostOp : public ::mlir::Op<_XlaSendFromHostOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::OpInvariants, GetResourceInstanceInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::DerivedAttributeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = _XlaSendFromHostOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("Tinputs"), ::llvm::StringRef("device_ordinal"), ::llvm::StringRef("key")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr TinputsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr TinputsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr device_ordinalAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr device_ordinalAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr keyAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr keyAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tf._XlaSendFromHost");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::TypedValue<::mlir::TensorType> dynamic_key();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange dynamic_keyMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::StringAttr keyAttr();
  ::llvm::StringRef key();
  ::mlir::IntegerAttr device_ordinalAttr();
  uint64_t device_ordinal();
  mlir::OperandElementTypeRange Tinputs();
  static bool isDerivedAttribute(::llvm::StringRef name);
  ::mlir::DictionaryAttr materializeDerivedAttributes();
  void keyAttr(::mlir::StringAttr attr);
  void device_ordinalAttr(::mlir::IntegerAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange inputs, ::mlir::Value dynamic_key, ::mlir::StringAttr key, ::mlir::IntegerAttr device_ordinal);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange inputs, ::mlir::Value dynamic_key, ::mlir::StringAttr key, ::mlir::IntegerAttr device_ordinal);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange inputs, ::mlir::Value dynamic_key, ::llvm::StringRef key, uint64_t device_ordinal);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange inputs, ::mlir::Value dynamic_key, ::llvm::StringRef key, uint64_t device_ordinal);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  std::string GetResourceInstanceStr();
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 3 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
};
} // namespace TF
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::TF::_XlaSendFromHostOp)

namespace mlir {
namespace TF {

//===----------------------------------------------------------------------===//
// ::mlir::TF::_XlaSendFromHostV2Op declarations
//===----------------------------------------------------------------------===//

class _XlaSendFromHostV2OpAdaptor {
public:
  _XlaSendFromHostV2OpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  _XlaSendFromHostV2OpAdaptor(_XlaSendFromHostV2Op op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::Value dynamic_key();
  ::mlir::Value device_ordinal();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::StringAttr keyAttr();
  ::llvm::StringRef key();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class _XlaSendFromHostV2Op : public ::mlir::Op<_XlaSendFromHostV2Op, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, GetResourceInstanceInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::DerivedAttributeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = _XlaSendFromHostV2OpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("Tinputs"), ::llvm::StringRef("key")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr TinputsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr TinputsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr keyAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr keyAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tf._XlaSendFromHostV2");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::TypedValue<::mlir::TensorType> dynamic_key();
  ::mlir::TypedValue<::mlir::TensorType> device_ordinal();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange dynamic_keyMutable();
  ::mlir::MutableOperandRange device_ordinalMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::StringAttr keyAttr();
  ::llvm::StringRef key();
  mlir::OperandElementTypeRange Tinputs();
  static bool isDerivedAttribute(::llvm::StringRef name);
  ::mlir::DictionaryAttr materializeDerivedAttributes();
  void keyAttr(::mlir::StringAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange inputs, ::mlir::Value dynamic_key, ::mlir::Value device_ordinal, ::mlir::StringAttr key);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange inputs, ::mlir::Value dynamic_key, ::mlir::Value device_ordinal, ::mlir::StringAttr key);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange inputs, ::mlir::Value dynamic_key, ::mlir::Value device_ordinal, ::llvm::StringRef key);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange inputs, ::mlir::Value dynamic_key, ::mlir::Value device_ordinal, ::llvm::StringRef key);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  std::string GetResourceInstanceStr();
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 2 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
};
} // namespace TF
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::TF::_XlaSendFromHostV2Op)


#endif  // GET_OP_CLASSES

