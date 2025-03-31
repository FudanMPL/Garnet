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
namespace acc {
class DataOp;
} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {
class EnterDataOp;
} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {
class ExitDataOp;
} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {
class InitOp;
} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {
class LoopOp;
} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {
class ParallelOp;
} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {
class ShutdownOp;
} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {
class TerminatorOp;
} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {
class UpdateOp;
} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {
class WaitOp;
} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {
class YieldOp;
} // namespace acc
} // namespace mlir
#endif

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::DataOp declarations
//===----------------------------------------------------------------------===//

class DataOpAdaptor {
public:
  DataOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  DataOpAdaptor(DataOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getIfCond();
  ::mlir::ValueRange getCopyOperands();
  ::mlir::ValueRange getCopyinOperands();
  ::mlir::ValueRange getCopyinReadonlyOperands();
  ::mlir::ValueRange getCopyoutOperands();
  ::mlir::ValueRange getCopyoutZeroOperands();
  ::mlir::ValueRange getCreateOperands();
  ::mlir::ValueRange getCreateZeroOperands();
  ::mlir::ValueRange getNoCreateOperands();
  ::mlir::ValueRange getPresentOperands();
  ::mlir::ValueRange getDeviceptrOperands();
  ::mlir::ValueRange getAttachOperands();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::acc::ClauseDefaultValueAttr getDefaultAttrAttr();
  ::llvm::Optional<::mlir::acc::ClauseDefaultValue> getDefaultAttr();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &getRegion();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class DataOp : public ::mlir::Op<DataOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = DataOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("defaultAttr"), ::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getDefaultAttrAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getDefaultAttrAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("acc.data");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::IntegerType> getIfCond();
  ::mlir::Operation::operand_range getCopyOperands();
  ::mlir::Operation::operand_range getCopyinOperands();
  ::mlir::Operation::operand_range getCopyinReadonlyOperands();
  ::mlir::Operation::operand_range getCopyoutOperands();
  ::mlir::Operation::operand_range getCopyoutZeroOperands();
  ::mlir::Operation::operand_range getCreateOperands();
  ::mlir::Operation::operand_range getCreateZeroOperands();
  ::mlir::Operation::operand_range getNoCreateOperands();
  ::mlir::Operation::operand_range getPresentOperands();
  ::mlir::Operation::operand_range getDeviceptrOperands();
  ::mlir::Operation::operand_range getAttachOperands();
  ::mlir::MutableOperandRange getIfCondMutable();
  ::mlir::MutableOperandRange getCopyOperandsMutable();
  ::mlir::MutableOperandRange getCopyinOperandsMutable();
  ::mlir::MutableOperandRange getCopyinReadonlyOperandsMutable();
  ::mlir::MutableOperandRange getCopyoutOperandsMutable();
  ::mlir::MutableOperandRange getCopyoutZeroOperandsMutable();
  ::mlir::MutableOperandRange getCreateOperandsMutable();
  ::mlir::MutableOperandRange getCreateZeroOperandsMutable();
  ::mlir::MutableOperandRange getNoCreateOperandsMutable();
  ::mlir::MutableOperandRange getPresentOperandsMutable();
  ::mlir::MutableOperandRange getDeviceptrOperandsMutable();
  ::mlir::MutableOperandRange getAttachOperandsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &getRegion();
  ::mlir::acc::ClauseDefaultValueAttr getDefaultAttrAttr();
  ::llvm::Optional<::mlir::acc::ClauseDefaultValue> getDefaultAttr();
  void setDefaultAttrAttr(::mlir::acc::ClauseDefaultValueAttr attr);
  ::mlir::Attribute removeDefaultAttrAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value ifCond, ::mlir::ValueRange copyOperands, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange copyinReadonlyOperands, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange copyoutZeroOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange noCreateOperands, ::mlir::ValueRange presentOperands, ::mlir::ValueRange deviceptrOperands, ::mlir::ValueRange attachOperands, /*optional*/::mlir::acc::ClauseDefaultValueAttr defaultAttr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value ifCond, ::mlir::ValueRange copyOperands, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange copyinReadonlyOperands, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange copyoutZeroOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange noCreateOperands, ::mlir::ValueRange presentOperands, ::mlir::ValueRange deviceptrOperands, ::mlir::ValueRange attachOperands, /*optional*/::mlir::acc::ClauseDefaultValueAttr defaultAttr);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
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
  /// The number of data operands.
  unsigned getNumDataOperands();

  /// The i-th data operand passed.
  Value getDataOperand(unsigned i);
};
} // namespace acc
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::acc::DataOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::EnterDataOp declarations
//===----------------------------------------------------------------------===//

class EnterDataOpAdaptor {
public:
  EnterDataOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  EnterDataOpAdaptor(EnterDataOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getIfCond();
  ::mlir::Value getAsyncOperand();
  ::mlir::Value getWaitDevnum();
  ::mlir::ValueRange getWaitOperands();
  ::mlir::ValueRange getCopyinOperands();
  ::mlir::ValueRange getCreateOperands();
  ::mlir::ValueRange getCreateZeroOperands();
  ::mlir::ValueRange getAttachOperands();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::UnitAttr getAsyncAttr();
  bool getAsync();
  ::mlir::UnitAttr getWaitAttr();
  bool getWait();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class EnterDataOp : public ::mlir::Op<EnterDataOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = EnterDataOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("async"), ::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("wait")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getAsyncAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getAsyncAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getWaitAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getWaitAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("acc.enter_data");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::IntegerType> getIfCond();
  ::mlir::Value getAsyncOperand();
  ::mlir::Value getWaitDevnum();
  ::mlir::Operation::operand_range getWaitOperands();
  ::mlir::Operation::operand_range getCopyinOperands();
  ::mlir::Operation::operand_range getCreateOperands();
  ::mlir::Operation::operand_range getCreateZeroOperands();
  ::mlir::Operation::operand_range getAttachOperands();
  ::mlir::MutableOperandRange getIfCondMutable();
  ::mlir::MutableOperandRange getAsyncOperandMutable();
  ::mlir::MutableOperandRange getWaitDevnumMutable();
  ::mlir::MutableOperandRange getWaitOperandsMutable();
  ::mlir::MutableOperandRange getCopyinOperandsMutable();
  ::mlir::MutableOperandRange getCreateOperandsMutable();
  ::mlir::MutableOperandRange getCreateZeroOperandsMutable();
  ::mlir::MutableOperandRange getAttachOperandsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::UnitAttr getAsyncAttr();
  bool getAsync();
  ::mlir::UnitAttr getWaitAttr();
  bool getWait();
  void setAsyncAttr(::mlir::UnitAttr attr);
  void setWaitAttr(::mlir::UnitAttr attr);
  ::mlir::Attribute removeAsyncAttr();
  ::mlir::Attribute removeWaitAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::UnitAttr async, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/::mlir::UnitAttr wait, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange attachOperands);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::UnitAttr async, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/::mlir::UnitAttr wait, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange attachOperands);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/bool async, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/bool wait, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange attachOperands);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/bool async, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/bool wait, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange attachOperands);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
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
  /// The number of data operands.
  unsigned getNumDataOperands();

  /// The i-th data operand passed.
  Value getDataOperand(unsigned i);
};
} // namespace acc
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::acc::EnterDataOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::ExitDataOp declarations
//===----------------------------------------------------------------------===//

class ExitDataOpAdaptor {
public:
  ExitDataOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  ExitDataOpAdaptor(ExitDataOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getIfCond();
  ::mlir::Value getAsyncOperand();
  ::mlir::Value getWaitDevnum();
  ::mlir::ValueRange getWaitOperands();
  ::mlir::ValueRange getCopyoutOperands();
  ::mlir::ValueRange getDeleteOperands();
  ::mlir::ValueRange getDetachOperands();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::UnitAttr getAsyncAttr();
  bool getAsync();
  ::mlir::UnitAttr getWaitAttr();
  bool getWait();
  ::mlir::UnitAttr getFinalizeAttr();
  bool getFinalize();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class ExitDataOp : public ::mlir::Op<ExitDataOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ExitDataOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("async"), ::llvm::StringRef("finalize"), ::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("wait")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getAsyncAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getAsyncAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getFinalizeAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getFinalizeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getWaitAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getWaitAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("acc.exit_data");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::IntegerType> getIfCond();
  ::mlir::Value getAsyncOperand();
  ::mlir::Value getWaitDevnum();
  ::mlir::Operation::operand_range getWaitOperands();
  ::mlir::Operation::operand_range getCopyoutOperands();
  ::mlir::Operation::operand_range getDeleteOperands();
  ::mlir::Operation::operand_range getDetachOperands();
  ::mlir::MutableOperandRange getIfCondMutable();
  ::mlir::MutableOperandRange getAsyncOperandMutable();
  ::mlir::MutableOperandRange getWaitDevnumMutable();
  ::mlir::MutableOperandRange getWaitOperandsMutable();
  ::mlir::MutableOperandRange getCopyoutOperandsMutable();
  ::mlir::MutableOperandRange getDeleteOperandsMutable();
  ::mlir::MutableOperandRange getDetachOperandsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::UnitAttr getAsyncAttr();
  bool getAsync();
  ::mlir::UnitAttr getWaitAttr();
  bool getWait();
  ::mlir::UnitAttr getFinalizeAttr();
  bool getFinalize();
  void setAsyncAttr(::mlir::UnitAttr attr);
  void setWaitAttr(::mlir::UnitAttr attr);
  void setFinalizeAttr(::mlir::UnitAttr attr);
  ::mlir::Attribute removeAsyncAttr();
  ::mlir::Attribute removeWaitAttr();
  ::mlir::Attribute removeFinalizeAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::UnitAttr async, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/::mlir::UnitAttr wait, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange deleteOperands, ::mlir::ValueRange detachOperands, /*optional*/::mlir::UnitAttr finalize);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::UnitAttr async, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/::mlir::UnitAttr wait, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange deleteOperands, ::mlir::ValueRange detachOperands, /*optional*/::mlir::UnitAttr finalize);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/bool async, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/bool wait, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange deleteOperands, ::mlir::ValueRange detachOperands, /*optional*/bool finalize = false);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/bool async, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/bool wait, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange deleteOperands, ::mlir::ValueRange detachOperands, /*optional*/bool finalize = false);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
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
  /// The number of data operands.
  unsigned getNumDataOperands();

  /// The i-th data operand passed.
  Value getDataOperand(unsigned i);
};
} // namespace acc
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::acc::ExitDataOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::InitOp declarations
//===----------------------------------------------------------------------===//

class InitOpAdaptor {
public:
  InitOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  InitOpAdaptor(InitOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange getDeviceTypeOperands();
  ::mlir::Value getDeviceNumOperand();
  ::mlir::Value getIfCond();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class InitOp : public ::mlir::Op<InitOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = InitOpAdaptor;
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
    return ::llvm::StringLiteral("acc.init");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getDeviceTypeOperands();
  ::mlir::Value getDeviceNumOperand();
  ::mlir::TypedValue<::mlir::IntegerType> getIfCond();
  ::mlir::MutableOperandRange getDeviceTypeOperandsMutable();
  ::mlir::MutableOperandRange getDeviceNumOperandMutable();
  ::mlir::MutableOperandRange getIfCondMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange deviceTypeOperands, /*optional*/::mlir::Value deviceNumOperand, /*optional*/::mlir::Value ifCond);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange deviceTypeOperands, /*optional*/::mlir::Value deviceNumOperand, /*optional*/::mlir::Value ifCond);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
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
} // namespace acc
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::acc::InitOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::LoopOp declarations
//===----------------------------------------------------------------------===//

class LoopOpAdaptor {
public:
  LoopOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  LoopOpAdaptor(LoopOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getGangNum();
  ::mlir::Value getGangStatic();
  ::mlir::Value getWorkerNum();
  ::mlir::Value getVectorLength();
  ::mlir::ValueRange getTileOperands();
  ::mlir::ValueRange getPrivateOperands();
  ::mlir::ValueRange getReductionOperands();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::IntegerAttr getCollapseAttr();
  ::llvm::Optional<uint64_t> getCollapse();
  ::mlir::UnitAttr getSeqAttr();
  bool getSeq();
  ::mlir::UnitAttr getIndependentAttr();
  bool getIndependent();
  ::mlir::UnitAttr getAuto_Attr();
  bool getAuto_();
  ::mlir::acc::ReductionOpAttr getReductionOpAttr();
  ::llvm::Optional<::mlir::acc::ReductionOp> getReductionOp();
  ::mlir::IntegerAttr getExecMappingAttr();
  uint64_t getExecMapping();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &getRegion();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class LoopOp : public ::mlir::Op<LoopOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::SingleBlockImplicitTerminator<acc::YieldOp>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = LoopOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("auto_"), ::llvm::StringRef("collapse"), ::llvm::StringRef("exec_mapping"), ::llvm::StringRef("independent"), ::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("reductionOp"), ::llvm::StringRef("seq")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getAuto_AttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getAuto_AttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getCollapseAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getCollapseAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getExecMappingAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getExecMappingAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getIndependentAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getIndependentAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr getReductionOpAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr getReductionOpAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  ::mlir::StringAttr getSeqAttrName() {
    return getAttributeNameForIndex(6);
  }

  static ::mlir::StringAttr getSeqAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 6);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("acc.loop");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value getGangNum();
  ::mlir::Value getGangStatic();
  ::mlir::Value getWorkerNum();
  ::mlir::Value getVectorLength();
  ::mlir::Operation::operand_range getTileOperands();
  ::mlir::Operation::operand_range getPrivateOperands();
  ::mlir::Operation::operand_range getReductionOperands();
  ::mlir::MutableOperandRange getGangNumMutable();
  ::mlir::MutableOperandRange getGangStaticMutable();
  ::mlir::MutableOperandRange getWorkerNumMutable();
  ::mlir::MutableOperandRange getVectorLengthMutable();
  ::mlir::MutableOperandRange getTileOperandsMutable();
  ::mlir::MutableOperandRange getPrivateOperandsMutable();
  ::mlir::MutableOperandRange getReductionOperandsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range getResults();
  ::mlir::Region &getRegion();
  ::mlir::IntegerAttr getCollapseAttr();
  ::llvm::Optional<uint64_t> getCollapse();
  ::mlir::UnitAttr getSeqAttr();
  bool getSeq();
  ::mlir::UnitAttr getIndependentAttr();
  bool getIndependent();
  ::mlir::UnitAttr getAuto_Attr();
  bool getAuto_();
  ::mlir::acc::ReductionOpAttr getReductionOpAttr();
  ::llvm::Optional<::mlir::acc::ReductionOp> getReductionOp();
  ::mlir::IntegerAttr getExecMappingAttr();
  uint64_t getExecMapping();
  void setCollapseAttr(::mlir::IntegerAttr attr);
  void setSeqAttr(::mlir::UnitAttr attr);
  void setIndependentAttr(::mlir::UnitAttr attr);
  void setAuto_Attr(::mlir::UnitAttr attr);
  void setReductionOpAttr(::mlir::acc::ReductionOpAttr attr);
  void setExecMappingAttr(::mlir::IntegerAttr attr);
  ::mlir::Attribute removeCollapseAttr();
  ::mlir::Attribute removeSeqAttr();
  ::mlir::Attribute removeIndependentAttr();
  ::mlir::Attribute removeAuto_Attr();
  ::mlir::Attribute removeReductionOpAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, /*optional*/::mlir::IntegerAttr collapse, /*optional*/::mlir::Value gangNum, /*optional*/::mlir::Value gangStatic, /*optional*/::mlir::Value workerNum, /*optional*/::mlir::Value vectorLength, /*optional*/::mlir::UnitAttr seq, /*optional*/::mlir::UnitAttr independent, /*optional*/::mlir::UnitAttr auto_, ::mlir::ValueRange tileOperands, ::mlir::ValueRange privateOperands, /*optional*/::mlir::acc::ReductionOpAttr reductionOp, ::mlir::ValueRange reductionOperands, ::mlir::IntegerAttr exec_mapping);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, /*optional*/::mlir::IntegerAttr collapse, /*optional*/::mlir::Value gangNum, /*optional*/::mlir::Value gangStatic, /*optional*/::mlir::Value workerNum, /*optional*/::mlir::Value vectorLength, /*optional*/bool seq, /*optional*/bool independent, /*optional*/bool auto_, ::mlir::ValueRange tileOperands, ::mlir::ValueRange privateOperands, /*optional*/::mlir::acc::ReductionOpAttr reductionOp, ::mlir::ValueRange reductionOperands, uint64_t exec_mapping = 0);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::RegisteredOperationName &opName, ::mlir::NamedAttrList &attributes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
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
  static StringRef getCollapseAttrStrName() { return "collapse"; }
  static StringRef getSeqAttrStrName() { return "seq"; }
  static StringRef getIndependentAttrStrName() { return "independent"; }
  static StringRef getAutoAttrStrName() { return "auto"; }
  static StringRef getExecutionMappingAttrStrName() { return "exec_mapping"; }
  static StringRef getGangKeyword() { return "gang"; }
  static StringRef getGangNumKeyword() { return "num"; }
  static StringRef getGangStaticKeyword() { return "static"; }
  static StringRef getVectorKeyword() { return "vector"; }
  static StringRef getWorkerKeyword() { return "worker"; }
  static StringRef getTileKeyword() { return "tile"; }
  static StringRef getPrivateKeyword() { return "private"; }
  static StringRef getReductionKeyword() { return "reduction"; }
};
} // namespace acc
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::acc::LoopOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::ParallelOp declarations
//===----------------------------------------------------------------------===//

class ParallelOpAdaptor {
public:
  ParallelOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  ParallelOpAdaptor(ParallelOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getAsync();
  ::mlir::ValueRange getWaitOperands();
  ::mlir::Value getNumGangs();
  ::mlir::Value getNumWorkers();
  ::mlir::Value getVectorLength();
  ::mlir::Value getIfCond();
  ::mlir::Value getSelfCond();
  ::mlir::ValueRange getReductionOperands();
  ::mlir::ValueRange getCopyOperands();
  ::mlir::ValueRange getCopyinOperands();
  ::mlir::ValueRange getCopyinReadonlyOperands();
  ::mlir::ValueRange getCopyoutOperands();
  ::mlir::ValueRange getCopyoutZeroOperands();
  ::mlir::ValueRange getCreateOperands();
  ::mlir::ValueRange getCreateZeroOperands();
  ::mlir::ValueRange getNoCreateOperands();
  ::mlir::ValueRange getPresentOperands();
  ::mlir::ValueRange getDevicePtrOperands();
  ::mlir::ValueRange getAttachOperands();
  ::mlir::ValueRange getGangPrivateOperands();
  ::mlir::ValueRange getGangFirstPrivateOperands();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::UnitAttr getAsyncAttrAttr();
  bool getAsyncAttr();
  ::mlir::UnitAttr getWaitAttrAttr();
  bool getWaitAttr();
  ::mlir::UnitAttr getSelfAttrAttr();
  bool getSelfAttr();
  ::mlir::acc::ReductionOpAttr getReductionOpAttr();
  ::llvm::Optional<::mlir::acc::ReductionOp> getReductionOp();
  ::mlir::acc::ClauseDefaultValueAttr getDefaultAttrAttr();
  ::llvm::Optional<::mlir::acc::ClauseDefaultValue> getDefaultAttr();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &getRegion();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class ParallelOp : public ::mlir::Op<ParallelOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ParallelOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("asyncAttr"), ::llvm::StringRef("defaultAttr"), ::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("reductionOp"), ::llvm::StringRef("selfAttr"), ::llvm::StringRef("waitAttr")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getAsyncAttrAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getAsyncAttrAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getDefaultAttrAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getDefaultAttrAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getReductionOpAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getReductionOpAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getSelfAttrAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getSelfAttrAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr getWaitAttrAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr getWaitAttrAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("acc.parallel");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value getAsync();
  ::mlir::Operation::operand_range getWaitOperands();
  ::mlir::Value getNumGangs();
  ::mlir::Value getNumWorkers();
  ::mlir::Value getVectorLength();
  ::mlir::TypedValue<::mlir::IntegerType> getIfCond();
  ::mlir::TypedValue<::mlir::IntegerType> getSelfCond();
  ::mlir::Operation::operand_range getReductionOperands();
  ::mlir::Operation::operand_range getCopyOperands();
  ::mlir::Operation::operand_range getCopyinOperands();
  ::mlir::Operation::operand_range getCopyinReadonlyOperands();
  ::mlir::Operation::operand_range getCopyoutOperands();
  ::mlir::Operation::operand_range getCopyoutZeroOperands();
  ::mlir::Operation::operand_range getCreateOperands();
  ::mlir::Operation::operand_range getCreateZeroOperands();
  ::mlir::Operation::operand_range getNoCreateOperands();
  ::mlir::Operation::operand_range getPresentOperands();
  ::mlir::Operation::operand_range getDevicePtrOperands();
  ::mlir::Operation::operand_range getAttachOperands();
  ::mlir::Operation::operand_range getGangPrivateOperands();
  ::mlir::Operation::operand_range getGangFirstPrivateOperands();
  ::mlir::MutableOperandRange getAsyncMutable();
  ::mlir::MutableOperandRange getWaitOperandsMutable();
  ::mlir::MutableOperandRange getNumGangsMutable();
  ::mlir::MutableOperandRange getNumWorkersMutable();
  ::mlir::MutableOperandRange getVectorLengthMutable();
  ::mlir::MutableOperandRange getIfCondMutable();
  ::mlir::MutableOperandRange getSelfCondMutable();
  ::mlir::MutableOperandRange getReductionOperandsMutable();
  ::mlir::MutableOperandRange getCopyOperandsMutable();
  ::mlir::MutableOperandRange getCopyinOperandsMutable();
  ::mlir::MutableOperandRange getCopyinReadonlyOperandsMutable();
  ::mlir::MutableOperandRange getCopyoutOperandsMutable();
  ::mlir::MutableOperandRange getCopyoutZeroOperandsMutable();
  ::mlir::MutableOperandRange getCreateOperandsMutable();
  ::mlir::MutableOperandRange getCreateZeroOperandsMutable();
  ::mlir::MutableOperandRange getNoCreateOperandsMutable();
  ::mlir::MutableOperandRange getPresentOperandsMutable();
  ::mlir::MutableOperandRange getDevicePtrOperandsMutable();
  ::mlir::MutableOperandRange getAttachOperandsMutable();
  ::mlir::MutableOperandRange getGangPrivateOperandsMutable();
  ::mlir::MutableOperandRange getGangFirstPrivateOperandsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &getRegion();
  ::mlir::UnitAttr getAsyncAttrAttr();
  bool getAsyncAttr();
  ::mlir::UnitAttr getWaitAttrAttr();
  bool getWaitAttr();
  ::mlir::UnitAttr getSelfAttrAttr();
  bool getSelfAttr();
  ::mlir::acc::ReductionOpAttr getReductionOpAttr();
  ::llvm::Optional<::mlir::acc::ReductionOp> getReductionOp();
  ::mlir::acc::ClauseDefaultValueAttr getDefaultAttrAttr();
  ::llvm::Optional<::mlir::acc::ClauseDefaultValue> getDefaultAttr();
  void setAsyncAttrAttr(::mlir::UnitAttr attr);
  void setWaitAttrAttr(::mlir::UnitAttr attr);
  void setSelfAttrAttr(::mlir::UnitAttr attr);
  void setReductionOpAttr(::mlir::acc::ReductionOpAttr attr);
  void setDefaultAttrAttr(::mlir::acc::ClauseDefaultValueAttr attr);
  ::mlir::Attribute removeAsyncAttrAttr();
  ::mlir::Attribute removeWaitAttrAttr();
  ::mlir::Attribute removeSelfAttrAttr();
  ::mlir::Attribute removeReductionOpAttr();
  ::mlir::Attribute removeDefaultAttrAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value async, /*optional*/::mlir::UnitAttr asyncAttr, ::mlir::ValueRange waitOperands, /*optional*/::mlir::UnitAttr waitAttr, /*optional*/::mlir::Value numGangs, /*optional*/::mlir::Value numWorkers, /*optional*/::mlir::Value vectorLength, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value selfCond, /*optional*/::mlir::UnitAttr selfAttr, /*optional*/::mlir::acc::ReductionOpAttr reductionOp, ::mlir::ValueRange reductionOperands, ::mlir::ValueRange copyOperands, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange copyinReadonlyOperands, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange copyoutZeroOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange noCreateOperands, ::mlir::ValueRange presentOperands, ::mlir::ValueRange devicePtrOperands, ::mlir::ValueRange attachOperands, ::mlir::ValueRange gangPrivateOperands, ::mlir::ValueRange gangFirstPrivateOperands, /*optional*/::mlir::acc::ClauseDefaultValueAttr defaultAttr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value async, /*optional*/::mlir::UnitAttr asyncAttr, ::mlir::ValueRange waitOperands, /*optional*/::mlir::UnitAttr waitAttr, /*optional*/::mlir::Value numGangs, /*optional*/::mlir::Value numWorkers, /*optional*/::mlir::Value vectorLength, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value selfCond, /*optional*/::mlir::UnitAttr selfAttr, /*optional*/::mlir::acc::ReductionOpAttr reductionOp, ::mlir::ValueRange reductionOperands, ::mlir::ValueRange copyOperands, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange copyinReadonlyOperands, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange copyoutZeroOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange noCreateOperands, ::mlir::ValueRange presentOperands, ::mlir::ValueRange devicePtrOperands, ::mlir::ValueRange attachOperands, ::mlir::ValueRange gangPrivateOperands, ::mlir::ValueRange gangFirstPrivateOperands, /*optional*/::mlir::acc::ClauseDefaultValueAttr defaultAttr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value async, /*optional*/bool asyncAttr, ::mlir::ValueRange waitOperands, /*optional*/bool waitAttr, /*optional*/::mlir::Value numGangs, /*optional*/::mlir::Value numWorkers, /*optional*/::mlir::Value vectorLength, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value selfCond, /*optional*/bool selfAttr, /*optional*/::mlir::acc::ReductionOpAttr reductionOp, ::mlir::ValueRange reductionOperands, ::mlir::ValueRange copyOperands, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange copyinReadonlyOperands, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange copyoutZeroOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange noCreateOperands, ::mlir::ValueRange presentOperands, ::mlir::ValueRange devicePtrOperands, ::mlir::ValueRange attachOperands, ::mlir::ValueRange gangPrivateOperands, ::mlir::ValueRange gangFirstPrivateOperands, /*optional*/::mlir::acc::ClauseDefaultValueAttr defaultAttr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value async, /*optional*/bool asyncAttr, ::mlir::ValueRange waitOperands, /*optional*/bool waitAttr, /*optional*/::mlir::Value numGangs, /*optional*/::mlir::Value numWorkers, /*optional*/::mlir::Value vectorLength, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value selfCond, /*optional*/bool selfAttr, /*optional*/::mlir::acc::ReductionOpAttr reductionOp, ::mlir::ValueRange reductionOperands, ::mlir::ValueRange copyOperands, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange copyinReadonlyOperands, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange copyoutZeroOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange noCreateOperands, ::mlir::ValueRange presentOperands, ::mlir::ValueRange devicePtrOperands, ::mlir::ValueRange attachOperands, ::mlir::ValueRange gangPrivateOperands, ::mlir::ValueRange gangFirstPrivateOperands, /*optional*/::mlir::acc::ClauseDefaultValueAttr defaultAttr);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
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
  static StringRef getAsyncKeyword() { return "async"; }
  static StringRef getAsyncAttrName() { return "asyncAttr"; }
  static StringRef getWaitKeyword() { return "wait"; }
  static StringRef getWaitAttrName() { return "waitAttr"; }
  static StringRef getNumGangsKeyword() { return "num_gangs"; }
  static StringRef getNumWorkersKeyword() { return "num_workers"; }
  static StringRef getVectorLengthKeyword() { return "vector_length"; }
  static StringRef getIfKeyword() { return "if"; }
  static StringRef getSelfKeyword() { return "self"; }
  static StringRef getSelfAttrName() { return "selfAttr"; }
  static StringRef getReductionKeyword() { return "reduction"; }
  static StringRef getCopyKeyword() { return "copy"; }
  static StringRef getCopyinKeyword() { return "copyin"; }
  static StringRef getCopyinReadonlyKeyword() { return "copyin_readonly"; }
  static StringRef getCopyoutKeyword() { return "copyout"; }
  static StringRef getCopyoutZeroKeyword() { return "copyout_zero"; }
  static StringRef getCreateKeyword() { return "create"; }
  static StringRef getCreateZeroKeyword() { return "create_zero"; }
  static StringRef getNoCreateKeyword() { return "no_create"; }
  static StringRef getPresentKeyword() { return "present"; }
  static StringRef getDevicePtrKeyword() { return "deviceptr"; }
  static StringRef getAttachKeyword() { return "attach"; }
  static StringRef getPrivateKeyword() { return "private"; }
  static StringRef getFirstPrivateKeyword() { return "firstprivate"; }

  /// The number of data operands.
  unsigned getNumDataOperands();

  /// The i-th data operand passed.
  Value getDataOperand(unsigned i);
};
} // namespace acc
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::acc::ParallelOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::ShutdownOp declarations
//===----------------------------------------------------------------------===//

class ShutdownOpAdaptor {
public:
  ShutdownOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  ShutdownOpAdaptor(ShutdownOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange getDeviceTypeOperands();
  ::mlir::Value getDeviceNumOperand();
  ::mlir::Value getIfCond();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class ShutdownOp : public ::mlir::Op<ShutdownOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ShutdownOpAdaptor;
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
    return ::llvm::StringLiteral("acc.shutdown");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getDeviceTypeOperands();
  ::mlir::Value getDeviceNumOperand();
  ::mlir::TypedValue<::mlir::IntegerType> getIfCond();
  ::mlir::MutableOperandRange getDeviceTypeOperandsMutable();
  ::mlir::MutableOperandRange getDeviceNumOperandMutable();
  ::mlir::MutableOperandRange getIfCondMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange deviceTypeOperands, /*optional*/::mlir::Value deviceNumOperand, /*optional*/::mlir::Value ifCond);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange deviceTypeOperands, /*optional*/::mlir::Value deviceNumOperand, /*optional*/::mlir::Value ifCond);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
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
} // namespace acc
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::acc::ShutdownOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::TerminatorOp declarations
//===----------------------------------------------------------------------===//

class TerminatorOpAdaptor {
public:
  TerminatorOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  TerminatorOpAdaptor(TerminatorOp op);

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
class TerminatorOp : public ::mlir::Op<TerminatorOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::OpTrait::IsTerminator> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = TerminatorOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("acc.terminator");
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
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
};
} // namespace acc
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::acc::TerminatorOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::UpdateOp declarations
//===----------------------------------------------------------------------===//

class UpdateOpAdaptor {
public:
  UpdateOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  UpdateOpAdaptor(UpdateOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getIfCond();
  ::mlir::Value getAsyncOperand();
  ::mlir::Value getWaitDevnum();
  ::mlir::ValueRange getWaitOperands();
  ::mlir::ValueRange getDeviceTypeOperands();
  ::mlir::ValueRange getHostOperands();
  ::mlir::ValueRange getDeviceOperands();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::UnitAttr getAsyncAttr();
  bool getAsync();
  ::mlir::UnitAttr getWaitAttr();
  bool getWait();
  ::mlir::UnitAttr getIfPresentAttr();
  bool getIfPresent();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class UpdateOp : public ::mlir::Op<UpdateOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = UpdateOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("async"), ::llvm::StringRef("ifPresent"), ::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("wait")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getAsyncAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getAsyncAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIfPresentAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIfPresentAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getWaitAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getWaitAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("acc.update");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::IntegerType> getIfCond();
  ::mlir::Value getAsyncOperand();
  ::mlir::Value getWaitDevnum();
  ::mlir::Operation::operand_range getWaitOperands();
  ::mlir::Operation::operand_range getDeviceTypeOperands();
  ::mlir::Operation::operand_range getHostOperands();
  ::mlir::Operation::operand_range getDeviceOperands();
  ::mlir::MutableOperandRange getIfCondMutable();
  ::mlir::MutableOperandRange getAsyncOperandMutable();
  ::mlir::MutableOperandRange getWaitDevnumMutable();
  ::mlir::MutableOperandRange getWaitOperandsMutable();
  ::mlir::MutableOperandRange getDeviceTypeOperandsMutable();
  ::mlir::MutableOperandRange getHostOperandsMutable();
  ::mlir::MutableOperandRange getDeviceOperandsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::UnitAttr getAsyncAttr();
  bool getAsync();
  ::mlir::UnitAttr getWaitAttr();
  bool getWait();
  ::mlir::UnitAttr getIfPresentAttr();
  bool getIfPresent();
  void setAsyncAttr(::mlir::UnitAttr attr);
  void setWaitAttr(::mlir::UnitAttr attr);
  void setIfPresentAttr(::mlir::UnitAttr attr);
  ::mlir::Attribute removeAsyncAttr();
  ::mlir::Attribute removeWaitAttr();
  ::mlir::Attribute removeIfPresentAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/::mlir::UnitAttr async, /*optional*/::mlir::UnitAttr wait, ::mlir::ValueRange deviceTypeOperands, ::mlir::ValueRange hostOperands, ::mlir::ValueRange deviceOperands, /*optional*/::mlir::UnitAttr ifPresent);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/::mlir::UnitAttr async, /*optional*/::mlir::UnitAttr wait, ::mlir::ValueRange deviceTypeOperands, ::mlir::ValueRange hostOperands, ::mlir::ValueRange deviceOperands, /*optional*/::mlir::UnitAttr ifPresent);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/bool async, /*optional*/bool wait, ::mlir::ValueRange deviceTypeOperands, ::mlir::ValueRange hostOperands, ::mlir::ValueRange deviceOperands, /*optional*/bool ifPresent = false);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/bool async, /*optional*/bool wait, ::mlir::ValueRange deviceTypeOperands, ::mlir::ValueRange hostOperands, ::mlir::ValueRange deviceOperands, /*optional*/bool ifPresent = false);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
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
  /// The number of data operands.
  unsigned getNumDataOperands();

  /// The i-th data operand passed.
  Value getDataOperand(unsigned i);
};
} // namespace acc
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::acc::UpdateOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::WaitOp declarations
//===----------------------------------------------------------------------===//

class WaitOpAdaptor {
public:
  WaitOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  WaitOpAdaptor(WaitOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange getWaitOperands();
  ::mlir::Value getAsyncOperand();
  ::mlir::Value getWaitDevnum();
  ::mlir::Value getIfCond();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::UnitAttr getAsyncAttr();
  bool getAsync();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class WaitOp : public ::mlir::Op<WaitOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = WaitOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("async"), ::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getAsyncAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getAsyncAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("acc.wait");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getWaitOperands();
  ::mlir::Value getAsyncOperand();
  ::mlir::Value getWaitDevnum();
  ::mlir::TypedValue<::mlir::IntegerType> getIfCond();
  ::mlir::MutableOperandRange getWaitOperandsMutable();
  ::mlir::MutableOperandRange getAsyncOperandMutable();
  ::mlir::MutableOperandRange getWaitDevnumMutable();
  ::mlir::MutableOperandRange getIfCondMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::UnitAttr getAsyncAttr();
  bool getAsync();
  void setAsyncAttr(::mlir::UnitAttr attr);
  ::mlir::Attribute removeAsyncAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange waitOperands, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::Value waitDevnum, /*optional*/::mlir::UnitAttr async, /*optional*/::mlir::Value ifCond);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange waitOperands, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::Value waitDevnum, /*optional*/::mlir::UnitAttr async, /*optional*/::mlir::Value ifCond);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange waitOperands, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::Value waitDevnum, /*optional*/bool async, /*optional*/::mlir::Value ifCond);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange waitOperands, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::Value waitDevnum, /*optional*/bool async, /*optional*/::mlir::Value ifCond);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
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
} // namespace acc
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::acc::WaitOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::YieldOp declarations
//===----------------------------------------------------------------------===//

class YieldOpAdaptor {
public:
  YieldOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  YieldOpAdaptor(YieldOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange operands();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class YieldOp : public ::mlir::Op<YieldOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::HasParent<ParallelOp, LoopOp>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::OpTrait::IsTerminator> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = YieldOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("acc.yield");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range operands();
  ::mlir::MutableOperandRange operandsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
};
} // namespace acc
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::acc::YieldOp)


#endif  // GET_OP_CLASSES

