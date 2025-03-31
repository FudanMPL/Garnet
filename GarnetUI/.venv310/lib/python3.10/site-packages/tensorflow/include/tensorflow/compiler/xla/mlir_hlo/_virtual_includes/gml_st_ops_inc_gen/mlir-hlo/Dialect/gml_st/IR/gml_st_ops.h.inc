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
namespace gml_st {
class ForOp;
} // namespace gml_st
} // namespace mlir
namespace mlir {
namespace gml_st {
class LoopOp;
} // namespace gml_st
} // namespace mlir
namespace mlir {
namespace gml_st {
class MaterializeOp;
} // namespace gml_st
} // namespace mlir
namespace mlir {
namespace gml_st {
class OffsetOp;
} // namespace gml_st
} // namespace mlir
namespace mlir {
namespace gml_st {
class ParallelOp;
} // namespace gml_st
} // namespace mlir
namespace mlir {
namespace gml_st {
class SetYieldOp;
} // namespace gml_st
} // namespace mlir
namespace mlir {
namespace gml_st {
class SizeOp;
} // namespace gml_st
} // namespace mlir
namespace mlir {
namespace gml_st {
class SpaceOp;
} // namespace gml_st
} // namespace mlir
namespace mlir {
namespace gml_st {
class StrideOp;
} // namespace gml_st
} // namespace mlir
namespace mlir {
namespace gml_st {
class TileOp;
} // namespace gml_st
} // namespace mlir
namespace mlir {
namespace gml_st {
class YieldOp;
} // namespace gml_st
} // namespace mlir
#endif

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace gml_st {

//===----------------------------------------------------------------------===//
// ::mlir::gml_st::ForOp declarations
//===----------------------------------------------------------------------===//

class ForOpAdaptor {
public:
  ForOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  ForOpAdaptor(ForOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange getLowerBound();
  ::mlir::ValueRange getUpperBound();
  ::mlir::ValueRange getStep();
  ::mlir::ValueRange getOutputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &getRegion();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class ForOp : public ::mlir::Op<ForOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::SingleBlockImplicitTerminator<gml_st::SetYieldOp>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::LoopLikeOpInterface::Trait, ::mlir::OpTrait::HasRecursiveMemoryEffects> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ForOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gml_st.for");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getLowerBound();
  ::mlir::Operation::operand_range getUpperBound();
  ::mlir::Operation::operand_range getStep();
  ::mlir::Operation::operand_range getOutputs();
  ::mlir::MutableOperandRange getLowerBoundMutable();
  ::mlir::MutableOperandRange getUpperBoundMutable();
  ::mlir::MutableOperandRange getStepMutable();
  ::mlir::MutableOperandRange getOutputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range getResults();
  ::mlir::Region &getRegion();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTypes, ValueRange lowerBounds, ValueRange upperBounds, ValueRange steps, ValueRange outputs, function_ref<void (OpBuilder &, Location, /*ivs=*/ValueRange,/*outputs=*/ValueRange)> bodyBuilderFn = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::mlir::ValueRange lowerBound, ::mlir::ValueRange upperBound, ::mlir::ValueRange step, ::mlir::ValueRange outputs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  ::mlir::Region &getLoopBody();
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
  /// Number of loops
  unsigned getNumLoops() { return getStep().size(); }

  /// Number of operands controlling the loop: lbs, ubs, steps
  unsigned getNumControlOperands() { return 3 * getNumLoops(); }

  ValueRange getInductionVars() {
    return getBody()->getArguments().take_front(getNumLoops());
  }

  /// Return whether the op has no output tensors.
  bool hasBufferSemantics() {
    return this->getOperation()->getNumResults() == 0;
  }

  /// Return terminator of the loop body.
  SetYieldOp getTerminator();

  /// Number of output operands
  unsigned getNumOutputs() { return getOutputs().size(); }

  /// Get the region output args.
  Block::BlockArgListType getRegionOutputArgs() {
    return getBody()->getArguments().take_back(getNumOutputs());
  }

  /// Get the region output arg that corresponds to an OpOperand.
  BlockArgument getRegionOutputArgForOpOperand(OpOperand &opOperand) {
    assert(opOperand.getOperandNumber() >= getNumControlOperands() &&
           "expected an output args operand");
    assert(opOperand.getOwner() == getOperation() &&
           "opOperand does not belong to this gml_st::ForOp operation");
    return getBody()->getArgument(opOperand.getOperandNumber() -
                                  getNumControlOperands() + getNumLoops());
  }

  /// Get the OpOperand& that corresponds to a region output arg.
  OpOperand &getOpOperandForRegionOutputArg(BlockArgument bbArg) {
    assert(bbArg.getArgNumber() >= getNumLoops() &&
           "expected a bbArg that is not an induction variable");
    assert(bbArg.getOwner()->getParentOp() == getOperation() &&
           "bbArg does not belong to the gml_st::ForOp body");
    return getOperation()->getOpOperand(
      getNumControlOperands() + bbArg.getArgNumber() - getNumLoops());
  }

  /// Get the OpResult that corresponds to an OpOperand.
  OpResult getResultForOpOperand(OpOperand &opOperand) {
    assert(opOperand.getOperandNumber() >= getNumControlOperands() &&
           "expected an output args operand");
    assert(opOperand.getOwner() == getOperation() &&
           "opOperand does not belong to this gml_st::ForOp operation");
    return getOperation()->getResult(
      opOperand.getOperandNumber() - getNumControlOperands());
  }

  /// Get the OpOperand& that corresponds to an OpResultOpOperand.
  OpOperand &getOpOperandForResult(OpResult opResult) {
    assert(opResult.getDefiningOp() == getOperation() &&
           "opResult does not belong to the gml_st::ForOp operation");
    return getOperation()->getOpOperand(
      getNumControlOperands() + opResult.getResultNumber());
  }
};
} // namespace gml_st
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::gml_st::ForOp)

namespace mlir {
namespace gml_st {

//===----------------------------------------------------------------------===//
// ::mlir::gml_st::LoopOp declarations
//===----------------------------------------------------------------------===//

class LoopOpAdaptor {
public:
  LoopOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  LoopOpAdaptor(LoopOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange getLowerBound();
  ::mlir::ValueRange getUpperBound();
  ::mlir::ValueRange getStep();
  ::mlir::ValueRange getInputs();
  ::mlir::ValueRange getOutputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::ArrayAttr getIteratorTypesAttr();
  ::mlir::ArrayAttr getIteratorTypes();
  ::mlir::ArrayAttr getDistributionTypesAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getDistributionTypes();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &getRegion();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class LoopOp : public ::mlir::Op<LoopOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::SingleBlockImplicitTerminator<gml_st::YieldOp>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::LoopLikeOpInterface::Trait, ::mlir::OpTrait::HasRecursiveMemoryEffects> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = LoopOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("distribution_types"), ::llvm::StringRef("iterator_types"), ::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getDistributionTypesAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getDistributionTypesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIteratorTypesAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIteratorTypesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gml_st.loop");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getLowerBound();
  ::mlir::Operation::operand_range getUpperBound();
  ::mlir::Operation::operand_range getStep();
  ::mlir::Operation::operand_range getInputs();
  ::mlir::Operation::operand_range getOutputs();
  ::mlir::MutableOperandRange getLowerBoundMutable();
  ::mlir::MutableOperandRange getUpperBoundMutable();
  ::mlir::MutableOperandRange getStepMutable();
  ::mlir::MutableOperandRange getInputsMutable();
  ::mlir::MutableOperandRange getOutputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range getResults();
  ::mlir::Region &getRegion();
  ::mlir::ArrayAttr getIteratorTypesAttr();
  ::mlir::ArrayAttr getIteratorTypes();
  ::mlir::ArrayAttr getDistributionTypesAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getDistributionTypes();
  void setIteratorTypesAttr(::mlir::ArrayAttr attr);
  void setDistributionTypesAttr(::mlir::ArrayAttr attr);
  ::mlir::Attribute removeDistribution_typesAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange lowerBounds, ValueRange upperBounds, ValueRange steps, ValueRange inputs, ValueRange outputs, ArrayAttr iteratorTypes, Optional<ArrayAttr> distributionTypes, function_ref<void (OpBuilder &, Location, /*ivs=*/ValueRange,/*inputs=*/ValueRange, /*outputs=*/ValueRange)> bodyBuilderFn = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange lowerBounds, ValueRange upperBounds, ValueRange steps, ValueRange inputs, ValueRange outputs, ArrayAttr iteratorTypes, function_ref<void (OpBuilder &, Location, /*ivs=*/ValueRange,/*inputs=*/ValueRange, /*outputs=*/ValueRange)> bodyBuilderFn = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::mlir::ValueRange lowerBound, ::mlir::ValueRange upperBound, ::mlir::ValueRange step, ::mlir::ValueRange inputs, ::mlir::ValueRange outputs, ::mlir::ArrayAttr iterator_types, /*optional*/::mlir::ArrayAttr distribution_types);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  ::mlir::Region &getLoopBody();
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
   /// Number of loops
   unsigned getNumLoops() { return getStep().size(); }

   /// Number of input operands
   unsigned getNumInputs() { return getInputs().size(); }

   /// Number of output operands
   unsigned getNumOutputs() { return getOutputs().size(); }

   /// Number of operands controlling the loop: lbs, ubs, steps
   unsigned getNumControlOperands() { return 3 * getNumLoops(); }

   ValueRange getInductionVars() {
     return getBody()->getArguments().take_front(getNumLoops());
   }
   ValueRange getRegionInputArgs() {
     return getBody()->getArguments().slice(getNumLoops(), getInputs().size());
   }
   ValueRange getRegionOutputArgs() {
     return getBody()->getArguments().take_back(getOutputs().size());
   }

   void setDistributionTypes(Builder& b, ArrayRef<StringRef> types) {
     assert(types.size() == getNumLoops() &&
            "expected distribution type for every dimension");
     setDistributionTypesAttr(b.getStrArrayAttr(types));
   }

   void setLowerBounds(ValueRange lowerBounds) {
     unsigned numLoops = getNumLoops();
     assert(lowerBounds.size() == numLoops &&
            "expected lower bounds for every loop dimension");
     for (unsigned i = 0; i < numLoops; ++i)
       setOperand(i, lowerBounds[i]);
   }

   void setUpperBounds(ValueRange upperBounds) {
     unsigned numLoops = getNumLoops();
     assert(upperBounds.size() == numLoops &&
            "expected upper bounds for every loop dimension");
     for (unsigned i = 0, pos = numLoops; i < numLoops; ++i, ++pos)
       setOperand(pos, upperBounds[i]);
   }

   void setSteps(ValueRange steps) {
     unsigned numLoops = getNumLoops();
     assert(steps.size() == numLoops &&
            "expected upper bounds for every loop dimension");
     for (unsigned i = 0, pos = 2 * numLoops; i < numLoops; ++i, ++pos)
       setOperand(pos, steps[i]);
   }

   /// Operand that corresponds to the `bbArg` block argument.
   OpOperand& getTiedOperand(BlockArgument& bbArg) {
     return getOperation()->getOpOperand(getNumControlOperands() +
                                         bbArg.getArgNumber() - getNumLoops());
   }

   /// Block argument that corresponds to the `input` or `output` operand.
   BlockArgument getTiedBlockArgument(OpOperand& operand) {
     auto operandIndex = operand.getOperandNumber();
     assert(
         operandIndex >= getNumControlOperands() &&
         operandIndex < getNumOperands() &&
         "tied block arg is defined only for `input` and `output` arguments");
     return getBody()->getArgument(operandIndex - 2 * getNumLoops());
   }

  /// Result that corresponds to the `outputs` argument of tensor type.
  OpResult getTiedOpResult(OpOperand& opOperand) {
     // No result can correspond to a memref argument.
     if (opOperand.get().getType().isa<MemRefType>()) return OpResult();

     // Check whether the operand index is in bounds of `outputs()` arg.
     int operandIndex = opOperand.getOperandNumber();
     int outputIndexStart =
         getNumControlOperands() + getInputs().size();
     int outputIndexEnd = outputIndexStart + getOutputs().size();
     if (operandIndex < outputIndexStart || operandIndex >= outputIndexEnd)
       return OpResult();

     // Count tensor arguments in `outputs` to compute the result index.
     int tensorId = -1;
     for (int i = outputIndexStart; i <= operandIndex; ++i)
       tensorId += getOperand(i).getType().isa<RankedTensorType>();
     return getOperation()->getResult(tensorId);
   }

   /// Append `operand` to the `input` arguments.
   OpOperand& appendInputOperand(OpBuilder& builder, Value operand) {
     int numLoops = getNumLoops();
     int numInputs = getNumInputs();
     int numOutputs = getNumOutputs();

     getOperation()->insertOperands(getNumControlOperands() + numInputs,
                                    operand);
     getBody()->insertArgument(numLoops + numInputs, operand.getType(),
                               getLoc());
     getOperation()->setAttr(
         LoopOp::getOperandSegmentSizeAttr(),
         builder.getDenseI32ArrayAttr(
             {numLoops, numLoops, numLoops, numInputs + 1, numOutputs}));
     return getOperation()->getOpOperand(getNumControlOperands() + numInputs);
   }

   /// Append `operand` to the `output` arguments.
   OpOperand& appendOutputOperand(OpBuilder& builder, Value operand) {
     int numLoops = getNumLoops();
     int numInputs = getNumInputs();
     int numOutputs = getNumOutputs();

     getOperation()->insertOperands(
         getNumControlOperands() + numInputs + numOutputs, operand);
     getBody()->insertArgument(numLoops + numInputs + numOutputs,
                               operand.getType(), getLoc());
     getOperation()->setAttr(
         LoopOp::getOperandSegmentSizeAttr(),
         builder.getDenseI32ArrayAttr(
             {numLoops, numLoops, numLoops, numInputs, numOutputs + 1}));
     return getOperation()->getOpOperand(getNumControlOperands() + numInputs +
                                         numOutputs);
   }

   /// Erase `operand` from the `input` or `output` arguments.
   void eraseOperand(OpBuilder& builder, OpOperand& operand) {
     int numInputs = getNumInputs();
     int numLoops = getNumLoops();
     int numOutputs = getNumOutputs();
     int numControlOperands = getNumControlOperands();

     int operandIndex = operand.getOperandNumber();
     assert(operandIndex >= numControlOperands &&
            operandIndex < static_cast<int>(getNumOperands()) &&
            "Can erase only `input` or `output` operand");

     if (operandIndex >= numControlOperands + numInputs)
       --numOutputs;
     else
       --numInputs;

     getOperation()->eraseOperand(operandIndex);
     getBody()->eraseArgument(operandIndex - 2 * numLoops);
     getOperation()->setAttr(
         LoopOp::getOperandSegmentSizeAttr(),
         builder.getDenseI32ArrayAttr(
             {numLoops, numLoops, numLoops, numInputs, numOutputs}));
   }

   OpOperand* findInputOperand(Value value) {
     OperandRange::iterator it = llvm::find(getInputs(), value);
     if (it == getInputs().end()) return nullptr;
     return it.getBase();
   }

   OpOperand* findOutputOperand(Value value) {
     OperandRange::iterator it = llvm::find(getOutputs(), value);
     if (it == getOutputs().end()) return nullptr;
     return it.getBase();
   }

   /// Return whether the op has only MemRef input and outputs.
   bool hasBufferSemantics() {
     Operation* op = this->getOperation();
     return op->getNumResults() == 0 &&
            llvm::all_of(op->getOpOperands(), [&](OpOperand & operand) {
              return !operand.get().getType().template isa<ShapedType>() ||
                     operand.get().getType().template isa<MemRefType>();
            });
   }

   static constexpr StringRef getDistributionTypesAttrStrName() {
     return "distribution_types";
   }
   static constexpr StringRef getIteratorTypesAttrStrName() {
     return "iterator_types";
   }


   /// Return whether the loop dimension is parallel or not.
   bool isParallelDimension(unsigned dim) {
     IteratorTypeAttr attr =
         this->getIteratorTypes()[dim].cast<IteratorTypeAttr>();
     return attr.getValue() == utils::IteratorType::parallel;
   }
};
} // namespace gml_st
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::gml_st::LoopOp)

namespace mlir {
namespace gml_st {

//===----------------------------------------------------------------------===//
// ::mlir::gml_st::MaterializeOp declarations
//===----------------------------------------------------------------------===//

class MaterializeOpAdaptor {
public:
  MaterializeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  MaterializeOpAdaptor(MaterializeOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getSource();
  ::mlir::Value getSet();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class MaterializeOp : public ::mlir::Op<MaterializeOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = MaterializeOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gml_st.materialize");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::ShapedType> getSource();
  ::mlir::Value getSet();
  ::mlir::MutableOperandRange getSourceMutable();
  ::mlir::MutableOperandRange getSetMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value getResult();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value source, Value set);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value source, ::mlir::Value set);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::Value set);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace gml_st
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::gml_st::MaterializeOp)

namespace mlir {
namespace gml_st {

//===----------------------------------------------------------------------===//
// ::mlir::gml_st::OffsetOp declarations
//===----------------------------------------------------------------------===//

class OffsetOpAdaptor {
public:
  OffsetOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  OffsetOpAdaptor(OffsetOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getSubset();
  ::mlir::Value getDim();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class OffsetOp : public ::mlir::Op<OffsetOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::IndexType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::InferTypeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = OffsetOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gml_st.offset");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value getSubset();
  ::mlir::TypedValue<::mlir::IndexType> getDim();
  ::mlir::MutableOperandRange getSubsetMutable();
  ::mlir::MutableOperandRange getDimMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::IndexType> getResult();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value subset, ::mlir::Value dim);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value subset, ::mlir::Value dim);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value subset, ::mlir::Value dim);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::OpFoldResult fold(::llvm::ArrayRef<::mlir::Attribute> operands);
  static ::mlir::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace gml_st
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::gml_st::OffsetOp)

namespace mlir {
namespace gml_st {

//===----------------------------------------------------------------------===//
// ::mlir::gml_st::ParallelOp declarations
//===----------------------------------------------------------------------===//

class ParallelOpAdaptor {
public:
  ParallelOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  ParallelOpAdaptor(ParallelOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange getLowerBound();
  ::mlir::ValueRange getUpperBound();
  ::mlir::ValueRange getStep();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::StringAttr getDistributionTypeAttr();
  ::llvm::Optional< ::llvm::StringRef > getDistributionType();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &getRegion();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class ParallelOp : public ::mlir::Op<ParallelOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::SingleBlockImplicitTerminator<gml_st::SetYieldOp>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::LoopLikeOpInterface::Trait, ::mlir::OpTrait::HasRecursiveMemoryEffects> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ParallelOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("distributionType"), ::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getDistributionTypeAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getDistributionTypeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gml_st.parallel");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getLowerBound();
  ::mlir::Operation::operand_range getUpperBound();
  ::mlir::Operation::operand_range getStep();
  ::mlir::MutableOperandRange getLowerBoundMutable();
  ::mlir::MutableOperandRange getUpperBoundMutable();
  ::mlir::MutableOperandRange getStepMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range getResults();
  ::mlir::Region &getRegion();
  ::mlir::StringAttr getDistributionTypeAttr();
  ::llvm::Optional< ::llvm::StringRef > getDistributionType();
  void setDistributionTypeAttr(::mlir::StringAttr attr);
  ::mlir::Attribute removeDistributionTypeAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTypes, ValueRange lowerBounds, ValueRange upperBounds, ValueRange steps, Optional<StringAttr> distributionType = llvm::None, function_ref<void (OpBuilder &, Location, /*ivs=*/ValueRange)> bodyBuilderFn = nullptr);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  ::mlir::Region &getLoopBody();
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
  /// Number of loops
  unsigned getNumLoops() { return getStep().size(); }

  /// Number of operands controlling the loop: lbs, ubs, steps
  unsigned getNumControlOperands() { return 3 * getNumLoops(); }

  ValueRange getInductionVars() {
    return getBody()->getArguments().take_front(getNumLoops());
  }

  /// Return whether the op has no output tensors.
  bool hasBufferSemantics() {
    return this->getOperation()->getNumResults() == 0;
  }

  /// Return terminator of the loop body.
  SetYieldOp getTerminator();
};
} // namespace gml_st
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::gml_st::ParallelOp)

namespace mlir {
namespace gml_st {

//===----------------------------------------------------------------------===//
// ::mlir::gml_st::SetYieldOp declarations
//===----------------------------------------------------------------------===//

class SetYieldOpAdaptor {
public:
  SetYieldOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  SetYieldOpAdaptor(SetYieldOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange getSrcs();
  ::mlir::ValueRange getDsts();
  ::mlir::ValueRange getSets();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::ArrayAttr getAccumulatorFlagsAttr();
  ::mlir::ArrayAttr getAccumulatorFlags();
  ::mlir::RegionRange getRegions();
  ::mlir::RegionRange getAccumulators();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class SetYieldOp : public ::mlir::Op<SetYieldOp, ::mlir::OpTrait::VariadicRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::ReturnLike, ::mlir::OpTrait::IsTerminator> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = SetYieldOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("accumulatorFlags")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getAccumulatorFlagsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getAccumulatorFlagsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gml_st.set_yield");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getSrcs();
  ::mlir::Operation::operand_range getDsts();
  ::mlir::Operation::operand_range getSets();
  ::mlir::MutableOperandRange getSrcsMutable();
  ::mlir::MutableOperandRange getDstsMutable();
  ::mlir::MutableOperandRange getSetsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::MutableArrayRef<::mlir::Region> getAccumulators();
  ::mlir::ArrayAttr getAccumulatorFlagsAttr();
  ::mlir::ArrayAttr getAccumulatorFlags();
  void setAccumulatorFlagsAttr(::mlir::ArrayAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange srcs, ValueRange dsts, ValueRange sets);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange srcs, ValueRange dsts, ValueRange sets, ArrayAttr accumulatorFlags, ArrayRef<function_ref<void(OpBuilder &, Location, Value, Value)>> combiners);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
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
  unsigned getNumUpdates() { return getSrcs().size(); }

  // Methods for `dst` arguments.
  OpOperand* getDstOperand(unsigned i) {
    return &getOperation()->getOpOperand(getNumUpdates() + i);
  }

  FailureOr<OpResult> getTiedOpResult(OpOperand &opOperand) {
    if (!isDstOperand(opOperand)) return failure();

    auto parent = getOperation()->getBlock()->getParentOp();
    if (isa<ParallelOp>(parent) || isa<ForOp>(parent)) {
      return parent->getResult(opOperand.getOperandNumber() -
                               getNumUpdates());
    }
    return failure();
  }

  bool isDstOperand(OpOperand& operand) {
    return operand.getOperandNumber() >= getNumUpdates() &&
           operand.getOperandNumber() < getNumUpdates() * 2;
  }
};
} // namespace gml_st
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::gml_st::SetYieldOp)

namespace mlir {
namespace gml_st {

//===----------------------------------------------------------------------===//
// ::mlir::gml_st::SizeOp declarations
//===----------------------------------------------------------------------===//

class SizeOpAdaptor {
public:
  SizeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  SizeOpAdaptor(SizeOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getTile();
  ::mlir::Value getDim();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class SizeOp : public ::mlir::Op<SizeOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::IndexType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::InferTypeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = SizeOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gml_st.size");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::gml_st::TileType> getTile();
  ::mlir::TypedValue<::mlir::IndexType> getDim();
  ::mlir::MutableOperandRange getTileMutable();
  ::mlir::MutableOperandRange getDimMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::IndexType> getResult();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value tile, ::mlir::Value dim);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value tile, ::mlir::Value dim);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value tile, ::mlir::Value dim);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::OpFoldResult fold(::llvm::ArrayRef<::mlir::Attribute> operands);
  static ::mlir::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace gml_st
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::gml_st::SizeOp)

namespace mlir {
namespace gml_st {

//===----------------------------------------------------------------------===//
// ::mlir::gml_st::SpaceOp declarations
//===----------------------------------------------------------------------===//

class SpaceOpAdaptor {
public:
  SpaceOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  SpaceOpAdaptor(SpaceOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange getDynamicSizes();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::ArrayAttr getStaticSizesAttr();
  ::mlir::ArrayAttr getStaticSizes();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class SpaceOp : public ::mlir::Op<SpaceOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::gml_st::TileType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::InferTypeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = SpaceOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("static_sizes")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getStaticSizesAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getStaticSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gml_st.space");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getDynamicSizes();
  ::mlir::MutableOperandRange getDynamicSizesMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::gml_st::TileType> getResult();
  ::mlir::ArrayAttr getStaticSizesAttr();
  ::mlir::ArrayAttr getStaticSizes();
  void setStaticSizesAttr(::mlir::ArrayAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ArrayRef<OpFoldResult> sizes, ArrayRef<NamedAttribute> attrs = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::ValueRange dynamic_sizes, ::mlir::ArrayAttr static_sizes);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange dynamic_sizes, ::mlir::ArrayAttr static_sizes);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange dynamic_sizes, ::mlir::ArrayAttr static_sizes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static ::mlir::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
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
  unsigned getNumDynamicEntriesUpToIdx(unsigned idx);
  mlir::Value getDynamicSize(unsigned idx);
};
} // namespace gml_st
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::gml_st::SpaceOp)

namespace mlir {
namespace gml_st {

//===----------------------------------------------------------------------===//
// ::mlir::gml_st::StrideOp declarations
//===----------------------------------------------------------------------===//

class StrideOpAdaptor {
public:
  StrideOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  StrideOpAdaptor(StrideOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getTile();
  ::mlir::Value getDim();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class StrideOp : public ::mlir::Op<StrideOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::IndexType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::InferTypeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = StrideOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gml_st.stride");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::gml_st::TileType> getTile();
  ::mlir::TypedValue<::mlir::IndexType> getDim();
  ::mlir::MutableOperandRange getTileMutable();
  ::mlir::MutableOperandRange getDimMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::IndexType> getResult();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value tile, ::mlir::Value dim);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value tile, ::mlir::Value dim);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value tile, ::mlir::Value dim);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::OpFoldResult fold(::llvm::ArrayRef<::mlir::Attribute> operands);
  static ::mlir::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace gml_st
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::gml_st::StrideOp)

namespace mlir {
namespace gml_st {

//===----------------------------------------------------------------------===//
// ::mlir::gml_st::TileOp declarations
//===----------------------------------------------------------------------===//

class TileOpAdaptor {
public:
  TileOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  TileOpAdaptor(TileOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getSuperset();
  ::mlir::ValueRange getOffsets();
  ::mlir::ValueRange getSizes();
  ::mlir::ValueRange getStrides();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::ArrayAttr getStaticOffsetsAttr();
  ::mlir::ArrayAttr getStaticOffsets();
  ::mlir::ArrayAttr getStaticSizesAttr();
  ::mlir::ArrayAttr getStaticSizes();
  ::mlir::ArrayAttr getStaticStridesAttr();
  ::mlir::ArrayAttr getStaticStrides();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class TileOp : public ::mlir::Op<TileOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::gml_st::TileType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OffsetSizeAndStrideOpInterface::Trait, ::mlir::InferTypeOpInterface::Trait, ::mlir::gml_st::ComposeSetInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = TileOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("static_offsets"), ::llvm::StringRef("static_sizes"), ::llvm::StringRef("static_strides")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getStaticOffsetsAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getStaticOffsetsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getStaticSizesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getStaticSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getStaticStridesAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getStaticStridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gml_st.tile");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::gml_st::TileType> getSuperset();
  ::mlir::Operation::operand_range getOffsets();
  ::mlir::Operation::operand_range getSizes();
  ::mlir::Operation::operand_range getStrides();
  ::mlir::MutableOperandRange getSupersetMutable();
  ::mlir::MutableOperandRange getOffsetsMutable();
  ::mlir::MutableOperandRange getSizesMutable();
  ::mlir::MutableOperandRange getStridesMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::gml_st::TileType> getResult();
  ::mlir::ArrayAttr getStaticOffsetsAttr();
  ::mlir::ArrayAttr getStaticOffsets();
  ::mlir::ArrayAttr getStaticSizesAttr();
  ::mlir::ArrayAttr getStaticSizes();
  ::mlir::ArrayAttr getStaticStridesAttr();
  ::mlir::ArrayAttr getStaticStrides();
  void setStaticOffsetsAttr(::mlir::ArrayAttr attr);
  void setStaticSizesAttr(::mlir::ArrayAttr attr);
  void setStaticStridesAttr(::mlir::ArrayAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value superset, ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes, ArrayRef<OpFoldResult> strides, ArrayRef<NamedAttribute> attrs = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value superset, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::mlir::ArrayAttr static_offsets, ::mlir::ArrayAttr static_sizes, ::mlir::ArrayAttr static_strides);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value superset, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::mlir::ArrayAttr static_offsets, ::mlir::ArrayAttr static_sizes, ::mlir::ArrayAttr static_strides);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value superset, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::mlir::ArrayAttr static_offsets, ::mlir::ArrayAttr static_sizes, ::mlir::ArrayAttr static_strides);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static ::mlir::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  ::mlir::Value compose(OpBuilder&builder);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
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
  /// Return the expected rank of each of the`static_offsets`, `static_sizes`
  /// and `static_strides` attributes.
  std::array<unsigned, 3> getArrayAttrMaxRanks() {
    unsigned rank = getSuperset().getType().cast<TileType>().getRank();
    return {rank, rank, rank};
  }
  /// Return the number of leading operands before the `offsets`, `sizes` and
  /// and `strides` operands.
  static unsigned getOffsetSizeAndStrideStartOperandIndex() { return 1; }
};
} // namespace gml_st
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::gml_st::TileOp)

namespace mlir {
namespace gml_st {

//===----------------------------------------------------------------------===//
// ::mlir::gml_st::YieldOp declarations
//===----------------------------------------------------------------------===//

class YieldOpAdaptor {
public:
  YieldOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  YieldOpAdaptor(YieldOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange getValues();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class YieldOp : public ::mlir::Op<YieldOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::HasParent<::mlir::gml_st::LoopOp, ::mlir::gml_st::SetYieldOp>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::ReturnLike, ::mlir::OpTrait::IsTerminator> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = YieldOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gml_st.yield");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getValues();
  ::mlir::MutableOperandRange getValuesMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange values);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace gml_st
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::gml_st::YieldOp)


#endif  // GET_OP_CLASSES

