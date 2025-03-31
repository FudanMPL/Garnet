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
namespace thlo {
class ConcatenateOp;
} // namespace thlo
} // namespace mlir
namespace mlir {
namespace thlo {
class DynamicBroadcastInDimOp;
} // namespace thlo
} // namespace mlir
namespace mlir {
namespace thlo {
class GatherOp;
} // namespace thlo
} // namespace mlir
namespace mlir {
namespace thlo {
class MapOp;
} // namespace thlo
} // namespace mlir
namespace mlir {
namespace thlo {
class ReductionOp;
} // namespace thlo
} // namespace mlir
namespace mlir {
namespace thlo {
class ScatterOp;
} // namespace thlo
} // namespace mlir
namespace mlir {
namespace thlo {
class SortOp;
} // namespace thlo
} // namespace mlir
namespace mlir {
namespace thlo {
class TransposeOp;
} // namespace thlo
} // namespace mlir
namespace mlir {
namespace thlo {
class YieldOp;
} // namespace thlo
} // namespace mlir
#endif

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace thlo {

//===----------------------------------------------------------------------===//
// ::mlir::thlo::ConcatenateOp declarations
//===----------------------------------------------------------------------===//

class ConcatenateOpAdaptor {
public:
  ConcatenateOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  ConcatenateOpAdaptor(ConcatenateOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange getInputs();
  ::mlir::Value getInit();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::IntegerAttr getDimensionAttr();
  uint64_t getDimension();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class ConcatenateOp : public ::mlir::Op<ConcatenateOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::linalg::DestinationStyleOpInterface::Trait, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::gml_st::TilingInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ConcatenateOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("dimension")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getDimensionAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getDimensionAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("thlo.concatenate");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getInputs();
  ::mlir::TypedValue<::mlir::ShapedType> getInit();
  ::mlir::MutableOperandRange getInputsMutable();
  ::mlir::MutableOperandRange getInitMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range getResult();
  ::mlir::IntegerAttr getDimensionAttr();
  uint64_t getDimension();
  void setDimensionAttr(::mlir::IntegerAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange result, ::mlir::ValueRange inputs, ::mlir::Value init, ::mlir::IntegerAttr dimension);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange result, ::mlir::ValueRange inputs, ::mlir::Value init, uint64_t dimension);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  SmallVector<Value> getDestinationOperands(OpBuilder &b);
  SmallVector<utils::IteratorType> getLoopIteratorTypes();
  SmallVector<Range> getIterationDomain(OpBuilder &b);
  mlir::gml_st::TilingInterface getTiledImplementation(OpBuilder &b, ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes);
  FailureOr<Value> generateResultTileValue(OpBuilder &b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes);
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
  unsigned getNumInputs() {
    return this->getOperation()->getNumOperands() - getNumOutputs();
  };
  unsigned getNumOutputs() { return 1; };

  // Implement functions necessary for DestinationStyleOpInterface.
  mlir::ValueRange inputs() { return getInputs(); }
  mlir::ValueRange getOutputs() { return getOperands().take_back(1); }
};
} // namespace thlo
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::thlo::ConcatenateOp)

namespace mlir {
namespace thlo {

//===----------------------------------------------------------------------===//
// ::mlir::thlo::DynamicBroadcastInDimOp declarations
//===----------------------------------------------------------------------===//

class DynamicBroadcastInDimOpAdaptor {
public:
  DynamicBroadcastInDimOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  DynamicBroadcastInDimOpAdaptor(DynamicBroadcastInDimOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getOperand();
  ::mlir::Value getInit();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseI64ArrayAttr getBroadcastDimensionsAttr();
  ::llvm::ArrayRef<int64_t> getBroadcastDimensions();
  ::mlir::DenseI64ArrayAttr getKnownExpandingDimensionsAttr();
  ::llvm::Optional<::llvm::ArrayRef<int64_t>> getKnownExpandingDimensions();
  ::mlir::DenseI64ArrayAttr getKnownNonexpandingDimensionsAttr();
  ::llvm::Optional<::llvm::ArrayRef<int64_t>> getKnownNonexpandingDimensions();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class DynamicBroadcastInDimOp : public ::mlir::Op<DynamicBroadcastInDimOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::linalg::DestinationStyleOpInterface::Trait, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::gml_st::TilingInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = DynamicBroadcastInDimOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("broadcast_dimensions"), ::llvm::StringRef("known_expanding_dimensions"), ::llvm::StringRef("known_nonexpanding_dimensions")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getBroadcastDimensionsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getBroadcastDimensionsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getKnownExpandingDimensionsAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getKnownExpandingDimensionsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getKnownNonexpandingDimensionsAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getKnownNonexpandingDimensionsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("thlo.dynamic_broadcast_in_dim");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::ShapedType> getOperand();
  ::mlir::TypedValue<::mlir::ShapedType> getInit();
  ::mlir::MutableOperandRange getOperandMutable();
  ::mlir::MutableOperandRange getInitMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range getResult();
  ::mlir::DenseI64ArrayAttr getBroadcastDimensionsAttr();
  ::llvm::ArrayRef<int64_t> getBroadcastDimensions();
  ::mlir::DenseI64ArrayAttr getKnownExpandingDimensionsAttr();
  ::llvm::Optional<::llvm::ArrayRef<int64_t>> getKnownExpandingDimensions();
  ::mlir::DenseI64ArrayAttr getKnownNonexpandingDimensionsAttr();
  ::llvm::Optional<::llvm::ArrayRef<int64_t>> getKnownNonexpandingDimensions();
  void setBroadcastDimensionsAttr(::mlir::DenseI64ArrayAttr attr);
  void setKnownExpandingDimensionsAttr(::mlir::DenseI64ArrayAttr attr);
  void setKnownNonexpandingDimensionsAttr(::mlir::DenseI64ArrayAttr attr);
  ::mlir::Attribute removeKnown_expanding_dimensionsAttr();
  ::mlir::Attribute removeKnown_nonexpanding_dimensionsAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange result, ::mlir::Value operand, ::mlir::Value init, ::mlir::DenseI64ArrayAttr broadcast_dimensions, /*optional*/::mlir::DenseI64ArrayAttr known_expanding_dimensions, /*optional*/::mlir::DenseI64ArrayAttr known_nonexpanding_dimensions);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange result, ::mlir::Value operand, ::mlir::Value init, ::llvm::ArrayRef<int64_t> broadcast_dimensions, /*optional*/::mlir::DenseI64ArrayAttr known_expanding_dimensions, /*optional*/::mlir::DenseI64ArrayAttr known_nonexpanding_dimensions);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  SmallVector<Value> getDestinationOperands(OpBuilder &b);
  SmallVector<utils::IteratorType> getLoopIteratorTypes();
  SmallVector<Range> getIterationDomain(OpBuilder &b);
  mlir::gml_st::TilingInterface getTiledImplementation(OpBuilder &b, ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes);
  FailureOr<Value> generateResultTileValue(OpBuilder &b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes);
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
  unsigned getNumInputs() {
    return this->getOperation()->getNumOperands() - getNumOutputs();
  };
  unsigned getNumOutputs() { return 1; };

  // Implement functions necessary for DestinationStyleOpInterface.
  mlir::ValueRange inputs() { return getOperands().take_front(1); }
  mlir::ValueRange getInputs() { return inputs(); }
  mlir::ValueRange getOutputs() { return getOperands().take_back(1); }
};
} // namespace thlo
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::thlo::DynamicBroadcastInDimOp)

namespace mlir {
namespace thlo {

//===----------------------------------------------------------------------===//
// ::mlir::thlo::GatherOp declarations
//===----------------------------------------------------------------------===//

class GatherOpAdaptor {
public:
  GatherOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  GatherOpAdaptor(GatherOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getOperand();
  ::mlir::Value getStartIndices();
  ::mlir::Value getInit();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class GatherOp : public ::mlir::Op<GatherOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<3>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::linalg::DestinationStyleOpInterface::Trait, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::gml_st::TilingInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = GatherOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("thlo.gather");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::ShapedType> getOperand();
  ::mlir::TypedValue<::mlir::ShapedType> getStartIndices();
  ::mlir::TypedValue<::mlir::ShapedType> getInit();
  ::mlir::MutableOperandRange getOperandMutable();
  ::mlir::MutableOperandRange getStartIndicesMutable();
  ::mlir::MutableOperandRange getInitMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range getResult();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange result, ::mlir::Value operand, ::mlir::Value start_indices, ::mlir::Value init);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  SmallVector<Value> getDestinationOperands(OpBuilder &b);
  SmallVector<utils::IteratorType> getLoopIteratorTypes();
  SmallVector<Range> getIterationDomain(OpBuilder &b);
  mlir::gml_st::TilingInterface getTiledImplementation(OpBuilder &b, ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes);
  FailureOr<Value> generateResultTileValue(OpBuilder &b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
  unsigned getNumInputs() {
    return this->getOperation()->getNumOperands() - getNumOutputs();
  };
  unsigned getNumOutputs() { return 1; };

  // Implement functions necessary for DestinationStyleOpInterface.
  mlir::ValueRange inputs() {
    return getOperands().drop_back(getNumOutputs());
  }
  mlir::ValueRange getInputs() { return inputs(); }
  mlir::ValueRange getOutputs() { return getOperands().take_back(1); }
};
} // namespace thlo
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::thlo::GatherOp)

namespace mlir {
namespace thlo {

//===----------------------------------------------------------------------===//
// ::mlir::thlo::MapOp declarations
//===----------------------------------------------------------------------===//

class MapOpAdaptor {
public:
  MapOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  MapOpAdaptor(MapOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange getInputs();
  ::mlir::Value getInit();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &getMapper();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class MapOp : public ::mlir::Op<MapOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::linalg::DestinationStyleOpInterface::Trait, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::linalg::LinalgOp::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = MapOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("thlo.map");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getInputs();
  ::mlir::TypedValue<::mlir::ShapedType> getInit();
  ::mlir::MutableOperandRange getInputsMutable();
  ::mlir::MutableOperandRange getInitMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range getResult();
  ::mlir::Region &getMapper();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange result, ::mlir::ValueRange inputs, ::mlir::Value init);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  SmallVector<StringRef> getIteratorTypesArray();
  ArrayAttr getIndexingMaps();
  bool hasIndexSemantics();
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
  unsigned getNumInputs() {
    return this->getOperation()->getNumOperands() - getNumOutputs();
  };
  unsigned getNumOutputs() { return 1; };

  // Implement functions necessary for DestinationStyleOpInterface.
  mlir::ValueRange inputs() { return getInputs(); }
  mlir::ValueRange getOutputs() { return getOperands().take_back(1); }
  linalg::OpOperandVector getOpOperandsMatchingBBargs() { return getInputOperands(); }

  static std::function<void(mlir::ImplicitLocOpBuilder &, mlir::Block &,
                            mlir::ArrayRef<mlir::NamedAttribute>)>
  getRegionBuilder() {
    return nullptr;
  }

};
} // namespace thlo
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::thlo::MapOp)

namespace mlir {
namespace thlo {

//===----------------------------------------------------------------------===//
// ::mlir::thlo::ReductionOp declarations
//===----------------------------------------------------------------------===//

class ReductionOpAdaptor {
public:
  ReductionOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  ReductionOpAdaptor(ReductionOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange getInputs();
  ::mlir::ValueRange getInits();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseI64ArrayAttr getDimensionsAttr();
  ::llvm::ArrayRef<int64_t> getDimensions();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &getCombiner();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class ReductionOp : public ::mlir::Op<ReductionOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::linalg::DestinationStyleOpInterface::Trait, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::linalg::LinalgOp::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ReductionOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("dimensions")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getDimensionsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getDimensionsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("thlo.reduction");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getInputs();
  ::mlir::Operation::operand_range getInits();
  ::mlir::MutableOperandRange getInputsMutable();
  ::mlir::MutableOperandRange getInitsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &getCombiner();
  ::mlir::DenseI64ArrayAttr getDimensionsAttr();
  ::llvm::ArrayRef<int64_t> getDimensions();
  void setDimensionsAttr(::mlir::DenseI64ArrayAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultType0, ::mlir::ValueRange inputs, ::mlir::ValueRange inits, ::mlir::DenseI64ArrayAttr dimensions);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultType0, ::mlir::ValueRange inputs, ::mlir::ValueRange inits, ::llvm::ArrayRef<int64_t> dimensions);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  SmallVector<StringRef> getIteratorTypesArray();
  ArrayAttr getIndexingMaps();
  bool hasIndexSemantics();
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
  // Implement functions necessary for DestinationStyleOpInterface.
  mlir::ValueRange inputs() { return getInputs(); }
  mlir::ValueRange getOutputs() { return getInits(); }
  unsigned getNumInputs() { return getInputs().size(); };
  unsigned getNumOutputs() { return getInits().size(); };
  static std::function<void(mlir::ImplicitLocOpBuilder &, mlir::Block &,
                            mlir::ArrayRef<mlir::NamedAttribute>)>
  getRegionBuilder() {
    return nullptr;
  }
};
} // namespace thlo
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::thlo::ReductionOp)

namespace mlir {
namespace thlo {

//===----------------------------------------------------------------------===//
// ::mlir::thlo::ScatterOp declarations
//===----------------------------------------------------------------------===//

class ScatterOpAdaptor {
public:
  ScatterOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  ScatterOpAdaptor(ScatterOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getIndices();
  ::mlir::Value getUpdates();
  ::mlir::Value getInit();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &getUpdateComputation();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class ScatterOp : public ::mlir::Op<ScatterOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<3>::Impl, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::linalg::DestinationStyleOpInterface::Trait, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::gml_st::TilingInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ScatterOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("thlo.scatter");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::ShapedType> getIndices();
  ::mlir::TypedValue<::mlir::ShapedType> getUpdates();
  ::mlir::TypedValue<::mlir::ShapedType> getInit();
  ::mlir::MutableOperandRange getIndicesMutable();
  ::mlir::MutableOperandRange getUpdatesMutable();
  ::mlir::MutableOperandRange getInitMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range getResult();
  ::mlir::Region &getUpdateComputation();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange result, ::mlir::Value indices, ::mlir::Value updates, ::mlir::Value init);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  SmallVector<Value> getDestinationOperands(OpBuilder &b);
  SmallVector<utils::IteratorType> getLoopIteratorTypes();
  SmallVector<Range> getIterationDomain(OpBuilder &b);
  mlir::gml_st::TilingInterface getTiledImplementation(OpBuilder &b, ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes);
  FailureOr<Value> generateResultTileValue(OpBuilder &b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
  unsigned getNumInputs() {
    return this->getOperation()->getNumOperands() - getNumOutputs();
  };
  unsigned getNumOutputs() { return 1; };

  // Implement functions necessary for DestinationStyleOpInterface.
  mlir::ValueRange inputs() {
    return getOperands().drop_back(getNumOutputs());
  }
  mlir::ValueRange getInputs() { return inputs(); }
  mlir::ValueRange getOutputs() { return getOperands().take_back(1); }

  // Returns index vector dimension size, which is always statically-known.
  int64_t getIndexVectorDimSize() {
    return getIndices().getType().getDimSize(1);
  }

  // Returns the number of indices, i.e. number of scalar/tensor updates.
  int64_t getIndicesCount() { return getIndices().getType().getDimSize(0); }
};
} // namespace thlo
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::thlo::ScatterOp)

namespace mlir {
namespace thlo {

//===----------------------------------------------------------------------===//
// ::mlir::thlo::SortOp declarations
//===----------------------------------------------------------------------===//

class SortOpAdaptor {
public:
  SortOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  SortOpAdaptor(SortOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange getInputs();
  ::mlir::ValueRange getInits();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::IntegerAttr getDimensionAttr();
  uint64_t getDimension();
  ::mlir::BoolAttr getIsStableAttr();
  bool getIsStable();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &getComparator();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class SortOp : public ::mlir::Op<SortOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::linalg::DestinationStyleOpInterface::Trait, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::HasRecursiveMemoryEffects, ::mlir::gml_st::TilingInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = SortOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("dimension"), ::llvm::StringRef("is_stable")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getDimensionAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getDimensionAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIsStableAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIsStableAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("thlo.sort");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getInputs();
  ::mlir::Operation::operand_range getInits();
  ::mlir::MutableOperandRange getInputsMutable();
  ::mlir::MutableOperandRange getInitsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range getResult();
  ::mlir::Region &getComparator();
  ::mlir::IntegerAttr getDimensionAttr();
  uint64_t getDimension();
  ::mlir::BoolAttr getIsStableAttr();
  bool getIsStable();
  void setDimensionAttr(::mlir::IntegerAttr attr);
  void setIsStableAttr(::mlir::BoolAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange result, ::mlir::ValueRange inputs, ::mlir::ValueRange inits, ::mlir::IntegerAttr dimension, ::mlir::BoolAttr is_stable);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange result, ::mlir::ValueRange inputs, ::mlir::ValueRange inits, uint64_t dimension, bool is_stable);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  SmallVector<Value> getDestinationOperands(OpBuilder &b);
  SmallVector<utils::IteratorType> getLoopIteratorTypes();
  SmallVector<Range> getIterationDomain(OpBuilder &b);
  mlir::gml_st::TilingInterface getTiledImplementation(OpBuilder &b, ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes);
  FailureOr<Value> generateResultTileValue(OpBuilder &b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes);
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
  // Implement functions necessary for DestinationStyleOpInterface.
  mlir::ValueRange getOutputs() { return getInits(); }
  unsigned getNumInputs() { return getInputs().size(); };
  unsigned getNumOutputs() { return getInits().size(); };
};
} // namespace thlo
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::thlo::SortOp)

namespace mlir {
namespace thlo {

//===----------------------------------------------------------------------===//
// ::mlir::thlo::TransposeOp declarations
//===----------------------------------------------------------------------===//

class TransposeOpAdaptor {
public:
  TransposeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  TransposeOpAdaptor(TransposeOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getInput();
  ::mlir::Value getInit();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseI64ArrayAttr getPermutationAttr();
  ::llvm::ArrayRef<int64_t> getPermutation();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &getRegion();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class TransposeOp : public ::mlir::Op<TransposeOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::linalg::DestinationStyleOpInterface::Trait, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::linalg::LinalgOp::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = TransposeOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("permutation")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getPermutationAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getPermutationAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("thlo.transpose");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::ShapedType> getInput();
  ::mlir::TypedValue<::mlir::ShapedType> getInit();
  ::mlir::MutableOperandRange getInputMutable();
  ::mlir::MutableOperandRange getInitMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range getResult();
  ::mlir::Region &getRegion();
  ::mlir::DenseI64ArrayAttr getPermutationAttr();
  ::llvm::ArrayRef<int64_t> getPermutation();
  void setPermutationAttr(::mlir::DenseI64ArrayAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, Value input, Value init, DenseI64ArrayAttr permutation, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, Value input, Value init, ArrayRef<int64_t> permutation, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  SmallVector<StringRef> getIteratorTypesArray();
  ArrayAttr getIndexingMaps();
  bool hasIndexSemantics();
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
  unsigned getNumInputs() {
    return this->getOperation()->getNumOperands() - getNumOutputs();
  };
  unsigned getNumOutputs() { return 1; };

  // Implement functions necessary for DestinationStyleOpInterface.
  mlir::ValueRange inputs() { return getOperands().take_front(1); }
  mlir::ValueRange getInputs() { return inputs(); }
  mlir::ValueRange getOutputs() { return getOperands().take_back(1); }

  static std::function<void(mlir::ImplicitLocOpBuilder &, mlir::Block &,
      mlir::ArrayRef<mlir::NamedAttribute>)>
    getRegionBuilder();

  static void createRegion(::mlir::OpBuilder &opBuilder,
                           ::mlir::OperationState & odsState);
};
} // namespace thlo
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::thlo::TransposeOp)

namespace mlir {
namespace thlo {

//===----------------------------------------------------------------------===//
// ::mlir::thlo::YieldOp declarations
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
class YieldOp : public ::mlir::Op<YieldOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::HasParent<MapOp, ReductionOp, ScatterOp, SortOp, TransposeOp>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::ReturnLike, ::mlir::OpTrait::IsTerminator> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = YieldOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("thlo.yield");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getValues();
  ::mlir::MutableOperandRange getValuesMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
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
} // namespace thlo
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::thlo::YieldOp)


#endif  // GET_OP_CLASSES

