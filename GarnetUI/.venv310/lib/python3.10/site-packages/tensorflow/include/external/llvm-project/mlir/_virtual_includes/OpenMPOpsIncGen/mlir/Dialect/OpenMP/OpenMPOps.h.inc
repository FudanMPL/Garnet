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
namespace omp {
class AtomicCaptureOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class AtomicReadOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class AtomicUpdateOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class AtomicWriteOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class BarrierOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class CancelOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class CancellationPointOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class CriticalDeclareOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class CriticalOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class FlushOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class MasterOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class OrderedOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class OrderedRegionOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class ParallelOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class ReductionDeclareOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class ReductionOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class SectionOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class SectionsOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class SimdLoopOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class SingleOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class TargetOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class TaskGroupOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class TaskLoopOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class TaskOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class TaskwaitOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class TaskyieldOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class TerminatorOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class ThreadprivateOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class WsLoopOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class YieldOp;
} // namespace omp
} // namespace mlir
#endif

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::AtomicCaptureOp declarations
//===----------------------------------------------------------------------===//

class AtomicCaptureOpAdaptor {
public:
  AtomicCaptureOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  AtomicCaptureOpAdaptor(AtomicCaptureOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::IntegerAttr getHintValAttr();
  uint64_t getHintVal();
  ::mlir::omp::ClauseMemoryOrderKindAttr getMemoryOrderValAttr();
  ::llvm::Optional<::mlir::omp::ClauseMemoryOrderKind> getMemoryOrderVal();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &getRegion();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class AtomicCaptureOp : public ::mlir::Op<AtomicCaptureOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::SingleBlockImplicitTerminator<TerminatorOp>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AtomicCaptureOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("hint_val"), ::llvm::StringRef("memory_order_val")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getHintValAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getHintValAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getMemoryOrderValAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getMemoryOrderValAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.atomic.capture");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &getRegion();
  ::mlir::IntegerAttr getHintValAttr();
  uint64_t getHintVal();
  ::mlir::omp::ClauseMemoryOrderKindAttr getMemoryOrderValAttr();
  ::llvm::Optional<::mlir::omp::ClauseMemoryOrderKind> getMemoryOrderVal();
  void setHintValAttr(::mlir::IntegerAttr attr);
  void setMemoryOrderValAttr(::mlir::omp::ClauseMemoryOrderKindAttr attr);
  ::mlir::Attribute removeHint_valAttr();
  ::mlir::Attribute removeMemory_order_valAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::IntegerAttr hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::IntegerAttr hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/uint64_t hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/uint64_t hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  ::mlir::LogicalResult verifyRegions();
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
  /// Returns the first operation in atomic capture region
  Operation* getFirstOp();

  /// Returns the second operation in atomic capture region
  Operation* getSecondOp();

  /// Returns the `atomic.read` operation inside the region, if any.
  /// Otherwise, it returns nullptr.
  AtomicReadOp getAtomicReadOp();

  /// Returns the `atomic.write` operation inside the region, if any.
  /// Otherwise, it returns nullptr.
  AtomicWriteOp getAtomicWriteOp();

  /// Returns the `atomic.update` operation inside the region, if any.
  /// Otherwise, it returns nullptr.
  AtomicUpdateOp getAtomicUpdateOp();
};
} // namespace omp
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::AtomicCaptureOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::AtomicReadOp declarations
//===----------------------------------------------------------------------===//

class AtomicReadOpAdaptor {
public:
  AtomicReadOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  AtomicReadOpAdaptor(AtomicReadOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getX();
  ::mlir::Value getV();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::IntegerAttr getHintValAttr();
  uint64_t getHintVal();
  ::mlir::omp::ClauseMemoryOrderKindAttr getMemoryOrderValAttr();
  ::llvm::Optional<::mlir::omp::ClauseMemoryOrderKind> getMemoryOrderVal();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class AtomicReadOp : public ::mlir::Op<AtomicReadOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AtomicReadOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("hint_val"), ::llvm::StringRef("memory_order_val")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getHintValAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getHintValAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getMemoryOrderValAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getMemoryOrderValAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.atomic.read");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::omp::PointerLikeType> getX();
  ::mlir::TypedValue<::mlir::omp::PointerLikeType> getV();
  ::mlir::MutableOperandRange getXMutable();
  ::mlir::MutableOperandRange getVMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::IntegerAttr getHintValAttr();
  uint64_t getHintVal();
  ::mlir::omp::ClauseMemoryOrderKindAttr getMemoryOrderValAttr();
  ::llvm::Optional<::mlir::omp::ClauseMemoryOrderKind> getMemoryOrderVal();
  void setHintValAttr(::mlir::IntegerAttr attr);
  void setMemoryOrderValAttr(::mlir::omp::ClauseMemoryOrderKindAttr attr);
  ::mlir::Attribute removeHint_valAttr();
  ::mlir::Attribute removeMemory_order_valAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value x, ::mlir::Value v, /*optional*/::mlir::IntegerAttr hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value x, ::mlir::Value v, /*optional*/::mlir::IntegerAttr hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value x, ::mlir::Value v, /*optional*/uint64_t hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value x, ::mlir::Value v, /*optional*/uint64_t hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val);
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
  /// The number of variable operands.
  unsigned getNumVariableOperands() {
    assert(getX() && "expected 'x' operand");
    assert(getV() && "expected 'v' operand");
    return 2;
  }

  /// The i-th variable operand passed.
  Value getVariableOperand(unsigned i) {
    assert(i < 2 && "invalid index position for an operand");
    return i == 0 ? getX() : getV();
  }
};
} // namespace omp
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::AtomicReadOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::AtomicUpdateOp declarations
//===----------------------------------------------------------------------===//

class AtomicUpdateOpAdaptor {
public:
  AtomicUpdateOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  AtomicUpdateOpAdaptor(AtomicUpdateOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getX();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::IntegerAttr getHintValAttr();
  uint64_t getHintVal();
  ::mlir::omp::ClauseMemoryOrderKindAttr getMemoryOrderValAttr();
  ::llvm::Optional<::mlir::omp::ClauseMemoryOrderKind> getMemoryOrderVal();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &getRegion();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class AtomicUpdateOp : public ::mlir::Op<AtomicUpdateOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::OpTrait::HasRecursiveMemoryEffects, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AtomicUpdateOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("hint_val"), ::llvm::StringRef("memory_order_val")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getHintValAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getHintValAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getMemoryOrderValAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getMemoryOrderValAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.atomic.update");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::omp::PointerLikeType> getX();
  ::mlir::MutableOperandRange getXMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &getRegion();
  ::mlir::IntegerAttr getHintValAttr();
  uint64_t getHintVal();
  ::mlir::omp::ClauseMemoryOrderKindAttr getMemoryOrderValAttr();
  ::llvm::Optional<::mlir::omp::ClauseMemoryOrderKind> getMemoryOrderVal();
  void setHintValAttr(::mlir::IntegerAttr attr);
  void setMemoryOrderValAttr(::mlir::omp::ClauseMemoryOrderKindAttr attr);
  ::mlir::Attribute removeHint_valAttr();
  ::mlir::Attribute removeMemory_order_valAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value x, /*optional*/::mlir::IntegerAttr hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value x, /*optional*/::mlir::IntegerAttr hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value x, /*optional*/uint64_t hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value x, /*optional*/uint64_t hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  ::mlir::LogicalResult verifyRegions();
  static ::mlir::LogicalResult canonicalize(AtomicUpdateOp op, ::mlir::PatternRewriter &rewriter);
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
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
  Operation* getFirstOp() {
    return &getRegion().front().getOperations().front();
  }

  /// Returns true if the new value is same as old value and the operation is
  /// a no-op, false otherwise.
  bool isNoOp();

  /// Returns the new value if the operation is equivalent to just a write
  /// operation. Otherwise, returns nullptr.
  Value getWriteOpVal();
};
} // namespace omp
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::AtomicUpdateOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::AtomicWriteOp declarations
//===----------------------------------------------------------------------===//

class AtomicWriteOpAdaptor {
public:
  AtomicWriteOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  AtomicWriteOpAdaptor(AtomicWriteOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getAddress();
  ::mlir::Value getValue();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::IntegerAttr getHintValAttr();
  uint64_t getHintVal();
  ::mlir::omp::ClauseMemoryOrderKindAttr getMemoryOrderValAttr();
  ::llvm::Optional<::mlir::omp::ClauseMemoryOrderKind> getMemoryOrderVal();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class AtomicWriteOp : public ::mlir::Op<AtomicWriteOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AtomicWriteOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("hint_val"), ::llvm::StringRef("memory_order_val")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getHintValAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getHintValAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getMemoryOrderValAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getMemoryOrderValAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.atomic.write");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::omp::PointerLikeType> getAddress();
  ::mlir::Value getValue();
  ::mlir::MutableOperandRange getAddressMutable();
  ::mlir::MutableOperandRange getValueMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::IntegerAttr getHintValAttr();
  uint64_t getHintVal();
  ::mlir::omp::ClauseMemoryOrderKindAttr getMemoryOrderValAttr();
  ::llvm::Optional<::mlir::omp::ClauseMemoryOrderKind> getMemoryOrderVal();
  void setHintValAttr(::mlir::IntegerAttr attr);
  void setMemoryOrderValAttr(::mlir::omp::ClauseMemoryOrderKindAttr attr);
  ::mlir::Attribute removeHint_valAttr();
  ::mlir::Attribute removeMemory_order_valAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value address, ::mlir::Value value, /*optional*/::mlir::IntegerAttr hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value address, ::mlir::Value value, /*optional*/::mlir::IntegerAttr hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value address, ::mlir::Value value, /*optional*/uint64_t hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value address, ::mlir::Value value, /*optional*/uint64_t hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val);
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
  /// The number of variable operands.
  unsigned getNumVariableOperands() {
    assert(getAddress() && "expected address operand");
    assert(getValue() && "expected value operand");
    return 2;
  }

  /// The i-th variable operand passed.
  Value getVariableOperand(unsigned i) {
    assert(i < 2 && "invalid index position for an operand");
    return i == 0 ? getAddress() : getValue();
  }
};
} // namespace omp
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::AtomicWriteOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::BarrierOp declarations
//===----------------------------------------------------------------------===//

class BarrierOpAdaptor {
public:
  BarrierOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  BarrierOpAdaptor(BarrierOp op);

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
class BarrierOp : public ::mlir::Op<BarrierOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = BarrierOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.barrier");
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
} // namespace omp
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::BarrierOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::CancelOp declarations
//===----------------------------------------------------------------------===//

class CancelOpAdaptor {
public:
  CancelOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  CancelOpAdaptor(CancelOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getIfExpr();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::omp::ClauseCancellationConstructTypeAttr getCancellationConstructTypeValAttr();
  ::mlir::omp::ClauseCancellationConstructType getCancellationConstructTypeVal();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class CancelOp : public ::mlir::Op<CancelOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = CancelOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("cancellation_construct_type_val")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getCancellationConstructTypeValAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getCancellationConstructTypeValAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.cancel");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::IntegerType> getIfExpr();
  ::mlir::MutableOperandRange getIfExprMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::omp::ClauseCancellationConstructTypeAttr getCancellationConstructTypeValAttr();
  ::mlir::omp::ClauseCancellationConstructType getCancellationConstructTypeVal();
  void setCancellationConstructTypeValAttr(::mlir::omp::ClauseCancellationConstructTypeAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::omp::ClauseCancellationConstructTypeAttr cancellation_construct_type_val, /*optional*/::mlir::Value if_expr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::omp::ClauseCancellationConstructTypeAttr cancellation_construct_type_val, /*optional*/::mlir::Value if_expr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::omp::ClauseCancellationConstructType cancellation_construct_type_val, /*optional*/::mlir::Value if_expr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::omp::ClauseCancellationConstructType cancellation_construct_type_val, /*optional*/::mlir::Value if_expr);
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
} // namespace omp
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::CancelOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::CancellationPointOp declarations
//===----------------------------------------------------------------------===//

class CancellationPointOpAdaptor {
public:
  CancellationPointOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  CancellationPointOpAdaptor(CancellationPointOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::omp::ClauseCancellationConstructTypeAttr getCancellationConstructTypeValAttr();
  ::mlir::omp::ClauseCancellationConstructType getCancellationConstructTypeVal();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class CancellationPointOp : public ::mlir::Op<CancellationPointOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = CancellationPointOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("cancellation_construct_type_val")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getCancellationConstructTypeValAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getCancellationConstructTypeValAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.cancellationpoint");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::omp::ClauseCancellationConstructTypeAttr getCancellationConstructTypeValAttr();
  ::mlir::omp::ClauseCancellationConstructType getCancellationConstructTypeVal();
  void setCancellationConstructTypeValAttr(::mlir::omp::ClauseCancellationConstructTypeAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::omp::ClauseCancellationConstructTypeAttr cancellation_construct_type_val);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::omp::ClauseCancellationConstructTypeAttr cancellation_construct_type_val);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::omp::ClauseCancellationConstructType cancellation_construct_type_val);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::omp::ClauseCancellationConstructType cancellation_construct_type_val);
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
} // namespace omp
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::CancellationPointOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::CriticalDeclareOp declarations
//===----------------------------------------------------------------------===//

class CriticalDeclareOpAdaptor {
public:
  CriticalDeclareOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  CriticalDeclareOpAdaptor(CriticalDeclareOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::StringAttr getSymNameAttr();
  ::llvm::StringRef getSymName();
  ::mlir::IntegerAttr getHintValAttr();
  uint64_t getHintVal();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class CriticalDeclareOp : public ::mlir::Op<CriticalDeclareOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::SymbolOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = CriticalDeclareOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("hint_val"), ::llvm::StringRef("sym_name")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getHintValAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getHintValAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getSymNameAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getSymNameAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.critical.declare");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::StringAttr getSymNameAttr();
  ::llvm::StringRef getSymName();
  ::mlir::IntegerAttr getHintValAttr();
  uint64_t getHintVal();
  void setSymNameAttr(::mlir::StringAttr attr);
  void setHintValAttr(::mlir::IntegerAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::StringAttr sym_name, ::mlir::IntegerAttr hint_val);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr sym_name, ::mlir::IntegerAttr hint_val);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::llvm::StringRef sym_name, uint64_t hint_val = 0);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef sym_name, uint64_t hint_val = 0);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::RegisteredOperationName &opName, ::mlir::NamedAttrList &attributes);
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
} // namespace omp
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::CriticalDeclareOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::CriticalOp declarations
//===----------------------------------------------------------------------===//

class CriticalOpAdaptor {
public:
  CriticalOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  CriticalOpAdaptor(CriticalOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::FlatSymbolRefAttr getNameAttr();
  ::llvm::Optional< ::llvm::StringRef > getName();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &getRegion();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class CriticalOp : public ::mlir::Op<CriticalOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::SymbolUserOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = CriticalOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("name")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getNameAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getNameAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.critical");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &getRegion();
  ::mlir::FlatSymbolRefAttr getNameAttr();
  ::llvm::Optional< ::llvm::StringRef > getName();
  void setNameAttr(::mlir::FlatSymbolRefAttr attr);
  ::mlir::Attribute removeNameAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::FlatSymbolRefAttr name);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::FlatSymbolRefAttr name);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verifySymbolUses(::mlir::SymbolTableCollection &symbolTable);
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
} // namespace omp
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::CriticalOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::FlushOp declarations
//===----------------------------------------------------------------------===//

class FlushOpAdaptor {
public:
  FlushOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  FlushOpAdaptor(FlushOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange getVarList();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class FlushOp : public ::mlir::Op<FlushOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = FlushOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.flush");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getVarList();
  ::mlir::MutableOperandRange getVarListMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange varList);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
  /// The number of variable operands.
  unsigned getNumVariableOperands() {
    return getOperation()->getNumOperands();
  }
  /// The i-th variable operand passed.
  Value getVariableOperand(unsigned i) {
    return getOperand(i);
  }
};
} // namespace omp
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::FlushOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::MasterOp declarations
//===----------------------------------------------------------------------===//

class MasterOpAdaptor {
public:
  MasterOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  MasterOpAdaptor(MasterOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
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
class MasterOp : public ::mlir::Op<MasterOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = MasterOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.master");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &getRegion();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
};
} // namespace omp
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::MasterOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::OrderedOp declarations
//===----------------------------------------------------------------------===//

class OrderedOpAdaptor {
public:
  OrderedOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  OrderedOpAdaptor(OrderedOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange getDependVecVars();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::omp::ClauseDependAttr getDependTypeValAttr();
  ::llvm::Optional<::mlir::omp::ClauseDepend> getDependTypeVal();
  ::mlir::IntegerAttr getNumLoopsValAttr();
  ::llvm::Optional<uint64_t> getNumLoopsVal();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class OrderedOp : public ::mlir::Op<OrderedOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = OrderedOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("depend_type_val"), ::llvm::StringRef("num_loops_val")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getDependTypeValAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getDependTypeValAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getNumLoopsValAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getNumLoopsValAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.ordered");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getDependVecVars();
  ::mlir::MutableOperandRange getDependVecVarsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::omp::ClauseDependAttr getDependTypeValAttr();
  ::llvm::Optional<::mlir::omp::ClauseDepend> getDependTypeVal();
  ::mlir::IntegerAttr getNumLoopsValAttr();
  ::llvm::Optional<uint64_t> getNumLoopsVal();
  void setDependTypeValAttr(::mlir::omp::ClauseDependAttr attr);
  void setNumLoopsValAttr(::mlir::IntegerAttr attr);
  ::mlir::Attribute removeDepend_type_valAttr();
  ::mlir::Attribute removeNum_loops_valAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::omp::ClauseDependAttr depend_type_val, /*optional*/::mlir::IntegerAttr num_loops_val, ::mlir::ValueRange depend_vec_vars);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::omp::ClauseDependAttr depend_type_val, /*optional*/::mlir::IntegerAttr num_loops_val, ::mlir::ValueRange depend_vec_vars);
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
} // namespace omp
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::OrderedOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::OrderedRegionOp declarations
//===----------------------------------------------------------------------===//

class OrderedRegionOpAdaptor {
public:
  OrderedRegionOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  OrderedRegionOpAdaptor(OrderedRegionOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::UnitAttr getSimdAttr();
  bool getSimd();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &getRegion();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class OrderedRegionOp : public ::mlir::Op<OrderedRegionOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = OrderedRegionOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("simd")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getSimdAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getSimdAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.ordered_region");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &getRegion();
  ::mlir::UnitAttr getSimdAttr();
  bool getSimd();
  void setSimdAttr(::mlir::UnitAttr attr);
  ::mlir::Attribute removeSimdAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::UnitAttr simd);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::UnitAttr simd);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/bool simd = false);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/bool simd = false);
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
} // namespace omp
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::OrderedRegionOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::ParallelOp declarations
//===----------------------------------------------------------------------===//

class ParallelOpAdaptor {
public:
  ParallelOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  ParallelOpAdaptor(ParallelOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getIfExprVar();
  ::mlir::Value getNumThreadsVar();
  ::mlir::ValueRange getAllocateVars();
  ::mlir::ValueRange getAllocatorsVars();
  ::mlir::ValueRange getReductionVars();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::ArrayAttr getReductionsAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getReductions();
  ::mlir::omp::ClauseProcBindKindAttr getProcBindValAttr();
  ::llvm::Optional<::mlir::omp::ClauseProcBindKind> getProcBindVal();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &getRegion();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class ParallelOp : public ::mlir::Op<ParallelOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::OpTrait::AutomaticAllocationScope, ::mlir::omp::OutlineableOpenMPOpInterface::Trait, ::mlir::OpTrait::HasRecursiveMemoryEffects, ::mlir::omp::ReductionClauseInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ParallelOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("proc_bind_val"), ::llvm::StringRef("reductions")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getProcBindValAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getProcBindValAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getReductionsAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getReductionsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.parallel");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::IntegerType> getIfExprVar();
  ::mlir::Value getNumThreadsVar();
  ::mlir::Operation::operand_range getAllocateVars();
  ::mlir::Operation::operand_range getAllocatorsVars();
  ::mlir::Operation::operand_range getReductionVars();
  ::mlir::MutableOperandRange getIfExprVarMutable();
  ::mlir::MutableOperandRange getNumThreadsVarMutable();
  ::mlir::MutableOperandRange getAllocateVarsMutable();
  ::mlir::MutableOperandRange getAllocatorsVarsMutable();
  ::mlir::MutableOperandRange getReductionVarsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &getRegion();
  ::mlir::ArrayAttr getReductionsAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getReductions();
  ::mlir::omp::ClauseProcBindKindAttr getProcBindValAttr();
  ::llvm::Optional<::mlir::omp::ClauseProcBindKind> getProcBindVal();
  void setReductionsAttr(::mlir::ArrayAttr attr);
  void setProcBindValAttr(::mlir::omp::ClauseProcBindKindAttr attr);
  ::mlir::Attribute removeReductionsAttr();
  ::mlir::Attribute removeProc_bind_valAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value if_expr_var, /*optional*/::mlir::Value num_threads_var, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars, ::mlir::ValueRange reduction_vars, /*optional*/::mlir::ArrayAttr reductions, /*optional*/::mlir::omp::ClauseProcBindKindAttr proc_bind_val);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value if_expr_var, /*optional*/::mlir::Value num_threads_var, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars, ::mlir::ValueRange reduction_vars, /*optional*/::mlir::ArrayAttr reductions, /*optional*/::mlir::omp::ClauseProcBindKindAttr proc_bind_val);
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
    assert(index < 3 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
};
} // namespace omp
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::ParallelOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::ReductionDeclareOp declarations
//===----------------------------------------------------------------------===//

class ReductionDeclareOpAdaptor {
public:
  ReductionDeclareOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  ReductionDeclareOpAdaptor(ReductionDeclareOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::StringAttr getSymNameAttr();
  ::llvm::StringRef getSymName();
  ::mlir::TypeAttr getTypeAttr();
  ::mlir::Type getType();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &getInitializerRegion();
  ::mlir::Region &getReductionRegion();
  ::mlir::Region &getAtomicReductionRegion();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class ReductionDeclareOp : public ::mlir::Op<ReductionDeclareOp, ::mlir::OpTrait::NRegions<3>::Impl, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::SymbolOpInterface::Trait, ::mlir::OpTrait::IsIsolatedFromAbove> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ReductionDeclareOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("sym_name"), ::llvm::StringRef("type")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getSymNameAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getSymNameAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getTypeAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getTypeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.reduction.declare");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &getInitializerRegion();
  ::mlir::Region &getReductionRegion();
  ::mlir::Region &getAtomicReductionRegion();
  ::mlir::StringAttr getSymNameAttr();
  ::llvm::StringRef getSymName();
  ::mlir::TypeAttr getTypeAttr();
  ::mlir::Type getType();
  void setSymNameAttr(::mlir::StringAttr attr);
  void setTypeAttr(::mlir::TypeAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::StringAttr sym_name, ::mlir::TypeAttr type);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr sym_name, ::mlir::TypeAttr type);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::llvm::StringRef sym_name, ::mlir::Type type);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef sym_name, ::mlir::Type type);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verifyRegions();
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
  PointerLikeType getAccumulatorType() {
    if (getAtomicReductionRegion().empty())
      return {};

    return getAtomicReductionRegion().front().getArgument(0).getType();
  }
};
} // namespace omp
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::ReductionDeclareOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::ReductionOp declarations
//===----------------------------------------------------------------------===//

class ReductionOpAdaptor {
public:
  ReductionOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  ReductionOpAdaptor(ReductionOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getOperand();
  ::mlir::Value getAccumulator();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class ReductionOp : public ::mlir::Op<ReductionOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ReductionOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.reduction");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value getOperand();
  ::mlir::TypedValue<::mlir::omp::PointerLikeType> getAccumulator();
  ::mlir::MutableOperandRange getOperandMutable();
  ::mlir::MutableOperandRange getAccumulatorMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand, ::mlir::Value accumulator);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand, ::mlir::Value accumulator);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
};
} // namespace omp
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::ReductionOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::SectionOp declarations
//===----------------------------------------------------------------------===//

class SectionOpAdaptor {
public:
  SectionOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  SectionOpAdaptor(SectionOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
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
class SectionOp : public ::mlir::Op<SectionOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::HasParent<SectionsOp>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = SectionOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.section");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &getRegion();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
};
} // namespace omp
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::SectionOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::SectionsOp declarations
//===----------------------------------------------------------------------===//

class SectionsOpAdaptor {
public:
  SectionsOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  SectionsOpAdaptor(SectionsOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange getReductionVars();
  ::mlir::ValueRange getAllocateVars();
  ::mlir::ValueRange getAllocatorsVars();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::ArrayAttr getReductionsAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getReductions();
  ::mlir::UnitAttr getNowaitAttr();
  bool getNowait();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &getRegion();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class SectionsOp : public ::mlir::Op<SectionsOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::omp::ReductionClauseInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = SectionsOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("nowait"), ::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("reductions")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getNowaitAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getNowaitAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getReductionsAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getReductionsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.sections");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getReductionVars();
  ::mlir::Operation::operand_range getAllocateVars();
  ::mlir::Operation::operand_range getAllocatorsVars();
  ::mlir::MutableOperandRange getReductionVarsMutable();
  ::mlir::MutableOperandRange getAllocateVarsMutable();
  ::mlir::MutableOperandRange getAllocatorsVarsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &getRegion();
  ::mlir::ArrayAttr getReductionsAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getReductions();
  ::mlir::UnitAttr getNowaitAttr();
  bool getNowait();
  void setReductionsAttr(::mlir::ArrayAttr attr);
  void setNowaitAttr(::mlir::UnitAttr attr);
  ::mlir::Attribute removeReductionsAttr();
  ::mlir::Attribute removeNowaitAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange reduction_vars, /*optional*/::mlir::ArrayAttr reductions, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars, /*optional*/::mlir::UnitAttr nowait);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange reduction_vars, /*optional*/::mlir::ArrayAttr reductions, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars, /*optional*/::mlir::UnitAttr nowait);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange reduction_vars, /*optional*/::mlir::ArrayAttr reductions, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars, /*optional*/bool nowait = false);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange reduction_vars, /*optional*/::mlir::ArrayAttr reductions, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars, /*optional*/bool nowait = false);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  ::mlir::LogicalResult verifyRegions();
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
};
} // namespace omp
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::SectionsOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::SimdLoopOp declarations
//===----------------------------------------------------------------------===//

class SimdLoopOpAdaptor {
public:
  SimdLoopOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  SimdLoopOpAdaptor(SimdLoopOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange getLowerBound();
  ::mlir::ValueRange getUpperBound();
  ::mlir::ValueRange getStep();
  ::mlir::Value getIfExpr();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::IntegerAttr getSimdlenAttr();
  ::llvm::Optional<uint64_t> getSimdlen();
  ::mlir::IntegerAttr getSafelenAttr();
  ::llvm::Optional<uint64_t> getSafelen();
  ::mlir::UnitAttr getInclusiveAttr();
  bool getInclusive();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &getRegion();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class SimdLoopOp : public ::mlir::Op<SimdLoopOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = SimdLoopOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("inclusive"), ::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("safelen"), ::llvm::StringRef("simdlen")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getInclusiveAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getInclusiveAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getSafelenAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getSafelenAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getSimdlenAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getSimdlenAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.simdloop");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getLowerBound();
  ::mlir::Operation::operand_range getUpperBound();
  ::mlir::Operation::operand_range getStep();
  ::mlir::TypedValue<::mlir::IntegerType> getIfExpr();
  ::mlir::MutableOperandRange getLowerBoundMutable();
  ::mlir::MutableOperandRange getUpperBoundMutable();
  ::mlir::MutableOperandRange getStepMutable();
  ::mlir::MutableOperandRange getIfExprMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &getRegion();
  ::mlir::IntegerAttr getSimdlenAttr();
  ::llvm::Optional<uint64_t> getSimdlen();
  ::mlir::IntegerAttr getSafelenAttr();
  ::llvm::Optional<uint64_t> getSafelen();
  ::mlir::UnitAttr getInclusiveAttr();
  bool getInclusive();
  void setSimdlenAttr(::mlir::IntegerAttr attr);
  void setSafelenAttr(::mlir::IntegerAttr attr);
  void setInclusiveAttr(::mlir::UnitAttr attr);
  ::mlir::Attribute removeSimdlenAttr();
  ::mlir::Attribute removeSafelenAttr();
  ::mlir::Attribute removeInclusiveAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange lowerBound, ::mlir::ValueRange upperBound, ::mlir::ValueRange step, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::IntegerAttr simdlen, /*optional*/::mlir::IntegerAttr safelen, /*optional*/::mlir::UnitAttr inclusive);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange lowerBound, ::mlir::ValueRange upperBound, ::mlir::ValueRange step, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::IntegerAttr simdlen, /*optional*/::mlir::IntegerAttr safelen, /*optional*/::mlir::UnitAttr inclusive);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange lowerBound, ::mlir::ValueRange upperBound, ::mlir::ValueRange step, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::IntegerAttr simdlen, /*optional*/::mlir::IntegerAttr safelen, /*optional*/bool inclusive = false);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange lowerBound, ::mlir::ValueRange upperBound, ::mlir::ValueRange step, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::IntegerAttr simdlen, /*optional*/::mlir::IntegerAttr safelen, /*optional*/bool inclusive = false);
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
    assert(index < 4 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  /// Returns the number of loops in the simd loop nest.
  unsigned getNumLoops() { return getLowerBound().size(); }

};
} // namespace omp
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::SimdLoopOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::SingleOp declarations
//===----------------------------------------------------------------------===//

class SingleOpAdaptor {
public:
  SingleOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  SingleOpAdaptor(SingleOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange getAllocateVars();
  ::mlir::ValueRange getAllocatorsVars();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::UnitAttr getNowaitAttr();
  bool getNowait();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &getRegion();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class SingleOp : public ::mlir::Op<SingleOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = SingleOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("nowait"), ::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getNowaitAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getNowaitAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.single");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getAllocateVars();
  ::mlir::Operation::operand_range getAllocatorsVars();
  ::mlir::MutableOperandRange getAllocateVarsMutable();
  ::mlir::MutableOperandRange getAllocatorsVarsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &getRegion();
  ::mlir::UnitAttr getNowaitAttr();
  bool getNowait();
  void setNowaitAttr(::mlir::UnitAttr attr);
  ::mlir::Attribute removeNowaitAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars, /*optional*/::mlir::UnitAttr nowait);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars, /*optional*/::mlir::UnitAttr nowait);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars, /*optional*/bool nowait = false);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars, /*optional*/bool nowait = false);
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
} // namespace omp
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::SingleOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::TargetOp declarations
//===----------------------------------------------------------------------===//

class TargetOpAdaptor {
public:
  TargetOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  TargetOpAdaptor(TargetOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getIfExpr();
  ::mlir::Value getDevice();
  ::mlir::Value getThreadLimit();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::UnitAttr getNowaitAttr();
  bool getNowait();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &getRegion();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class TargetOp : public ::mlir::Op<TargetOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = TargetOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("nowait"), ::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getNowaitAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getNowaitAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.target");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::IntegerType> getIfExpr();
  ::mlir::TypedValue<::mlir::IntegerType> getDevice();
  ::mlir::TypedValue<::mlir::IntegerType> getThreadLimit();
  ::mlir::MutableOperandRange getIfExprMutable();
  ::mlir::MutableOperandRange getDeviceMutable();
  ::mlir::MutableOperandRange getThreadLimitMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &getRegion();
  ::mlir::UnitAttr getNowaitAttr();
  bool getNowait();
  void setNowaitAttr(::mlir::UnitAttr attr);
  ::mlir::Attribute removeNowaitAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value device, /*optional*/::mlir::Value thread_limit, /*optional*/::mlir::UnitAttr nowait);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value device, /*optional*/::mlir::Value thread_limit, /*optional*/::mlir::UnitAttr nowait);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value device, /*optional*/::mlir::Value thread_limit, /*optional*/bool nowait = false);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value device, /*optional*/::mlir::Value thread_limit, /*optional*/bool nowait = false);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
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
} // namespace omp
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::TargetOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::TaskGroupOp declarations
//===----------------------------------------------------------------------===//

class TaskGroupOpAdaptor {
public:
  TaskGroupOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  TaskGroupOpAdaptor(TaskGroupOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange getTaskReductionVars();
  ::mlir::ValueRange getAllocateVars();
  ::mlir::ValueRange getAllocatorsVars();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::ArrayAttr getTaskReductionsAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getTaskReductions();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &getRegion();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class TaskGroupOp : public ::mlir::Op<TaskGroupOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::omp::ReductionClauseInterface::Trait, ::mlir::OpTrait::AutomaticAllocationScope> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = TaskGroupOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("task_reductions")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getTaskReductionsAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getTaskReductionsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.taskgroup");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getTaskReductionVars();
  ::mlir::Operation::operand_range getAllocateVars();
  ::mlir::Operation::operand_range getAllocatorsVars();
  ::mlir::MutableOperandRange getTaskReductionVarsMutable();
  ::mlir::MutableOperandRange getAllocateVarsMutable();
  ::mlir::MutableOperandRange getAllocatorsVarsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &getRegion();
  ::mlir::ArrayAttr getTaskReductionsAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getTaskReductions();
  void setTaskReductionsAttr(::mlir::ArrayAttr attr);
  ::mlir::Attribute removeTask_reductionsAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange task_reduction_vars, /*optional*/::mlir::ArrayAttr task_reductions, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange task_reduction_vars, /*optional*/::mlir::ArrayAttr task_reductions, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars);
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
  /// Returns the reduction variables
  operand_range getAllReductionVars() { return getTaskReductionVars(); }
};
} // namespace omp
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::TaskGroupOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::TaskLoopOp declarations
//===----------------------------------------------------------------------===//

class TaskLoopOpAdaptor {
public:
  TaskLoopOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  TaskLoopOpAdaptor(TaskLoopOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange getLowerBound();
  ::mlir::ValueRange getUpperBound();
  ::mlir::ValueRange getStep();
  ::mlir::Value getIfExpr();
  ::mlir::Value getFinalExpr();
  ::mlir::ValueRange getInReductionVars();
  ::mlir::ValueRange getReductionVars();
  ::mlir::Value getPriority();
  ::mlir::ValueRange getAllocateVars();
  ::mlir::ValueRange getAllocatorsVars();
  ::mlir::Value getGrainSize();
  ::mlir::Value getNumTasks();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::UnitAttr getInclusiveAttr();
  bool getInclusive();
  ::mlir::UnitAttr getUntiedAttr();
  bool getUntied();
  ::mlir::UnitAttr getMergeableAttr();
  bool getMergeable();
  ::mlir::ArrayAttr getInReductionsAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getInReductions();
  ::mlir::ArrayAttr getReductionsAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getReductions();
  ::mlir::UnitAttr getNogroupAttr();
  bool getNogroup();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &getRegion();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class TaskLoopOp : public ::mlir::Op<TaskLoopOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::OpTrait::AutomaticAllocationScope, ::mlir::OpTrait::HasRecursiveMemoryEffects, ::mlir::omp::ReductionClauseInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = TaskLoopOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("in_reductions"), ::llvm::StringRef("inclusive"), ::llvm::StringRef("mergeable"), ::llvm::StringRef("nogroup"), ::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("reductions"), ::llvm::StringRef("untied")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getInReductionsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getInReductionsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getInclusiveAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getInclusiveAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getMergeableAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getMergeableAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getNogroupAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getNogroupAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr getReductionsAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr getReductionsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  ::mlir::StringAttr getUntiedAttrName() {
    return getAttributeNameForIndex(6);
  }

  static ::mlir::StringAttr getUntiedAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 6);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.taskloop");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getLowerBound();
  ::mlir::Operation::operand_range getUpperBound();
  ::mlir::Operation::operand_range getStep();
  ::mlir::TypedValue<::mlir::IntegerType> getIfExpr();
  ::mlir::TypedValue<::mlir::IntegerType> getFinalExpr();
  ::mlir::Operation::operand_range getInReductionVars();
  ::mlir::Operation::operand_range getReductionVars();
  ::mlir::Value getPriority();
  ::mlir::Operation::operand_range getAllocateVars();
  ::mlir::Operation::operand_range getAllocatorsVars();
  ::mlir::Value getGrainSize();
  ::mlir::Value getNumTasks();
  ::mlir::MutableOperandRange getLowerBoundMutable();
  ::mlir::MutableOperandRange getUpperBoundMutable();
  ::mlir::MutableOperandRange getStepMutable();
  ::mlir::MutableOperandRange getIfExprMutable();
  ::mlir::MutableOperandRange getFinalExprMutable();
  ::mlir::MutableOperandRange getInReductionVarsMutable();
  ::mlir::MutableOperandRange getReductionVarsMutable();
  ::mlir::MutableOperandRange getPriorityMutable();
  ::mlir::MutableOperandRange getAllocateVarsMutable();
  ::mlir::MutableOperandRange getAllocatorsVarsMutable();
  ::mlir::MutableOperandRange getGrainSizeMutable();
  ::mlir::MutableOperandRange getNumTasksMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &getRegion();
  ::mlir::UnitAttr getInclusiveAttr();
  bool getInclusive();
  ::mlir::UnitAttr getUntiedAttr();
  bool getUntied();
  ::mlir::UnitAttr getMergeableAttr();
  bool getMergeable();
  ::mlir::ArrayAttr getInReductionsAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getInReductions();
  ::mlir::ArrayAttr getReductionsAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getReductions();
  ::mlir::UnitAttr getNogroupAttr();
  bool getNogroup();
  void setInclusiveAttr(::mlir::UnitAttr attr);
  void setUntiedAttr(::mlir::UnitAttr attr);
  void setMergeableAttr(::mlir::UnitAttr attr);
  void setInReductionsAttr(::mlir::ArrayAttr attr);
  void setReductionsAttr(::mlir::ArrayAttr attr);
  void setNogroupAttr(::mlir::UnitAttr attr);
  ::mlir::Attribute removeInclusiveAttr();
  ::mlir::Attribute removeUntiedAttr();
  ::mlir::Attribute removeMergeableAttr();
  ::mlir::Attribute removeIn_reductionsAttr();
  ::mlir::Attribute removeReductionsAttr();
  ::mlir::Attribute removeNogroupAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange lowerBound, ::mlir::ValueRange upperBound, ::mlir::ValueRange step, /*optional*/::mlir::UnitAttr inclusive, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value final_expr, /*optional*/::mlir::UnitAttr untied, /*optional*/::mlir::UnitAttr mergeable, ::mlir::ValueRange in_reduction_vars, /*optional*/::mlir::ArrayAttr in_reductions, ::mlir::ValueRange reduction_vars, /*optional*/::mlir::ArrayAttr reductions, /*optional*/::mlir::Value priority, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars, /*optional*/::mlir::Value grain_size, /*optional*/::mlir::Value num_tasks, /*optional*/::mlir::UnitAttr nogroup);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange lowerBound, ::mlir::ValueRange upperBound, ::mlir::ValueRange step, /*optional*/::mlir::UnitAttr inclusive, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value final_expr, /*optional*/::mlir::UnitAttr untied, /*optional*/::mlir::UnitAttr mergeable, ::mlir::ValueRange in_reduction_vars, /*optional*/::mlir::ArrayAttr in_reductions, ::mlir::ValueRange reduction_vars, /*optional*/::mlir::ArrayAttr reductions, /*optional*/::mlir::Value priority, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars, /*optional*/::mlir::Value grain_size, /*optional*/::mlir::Value num_tasks, /*optional*/::mlir::UnitAttr nogroup);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange lowerBound, ::mlir::ValueRange upperBound, ::mlir::ValueRange step, /*optional*/bool inclusive, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value final_expr, /*optional*/bool untied, /*optional*/bool mergeable, ::mlir::ValueRange in_reduction_vars, /*optional*/::mlir::ArrayAttr in_reductions, ::mlir::ValueRange reduction_vars, /*optional*/::mlir::ArrayAttr reductions, /*optional*/::mlir::Value priority, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars, /*optional*/::mlir::Value grain_size, /*optional*/::mlir::Value num_tasks, /*optional*/bool nogroup = false);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange lowerBound, ::mlir::ValueRange upperBound, ::mlir::ValueRange step, /*optional*/bool inclusive, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value final_expr, /*optional*/bool untied, /*optional*/bool mergeable, ::mlir::ValueRange in_reduction_vars, /*optional*/::mlir::ArrayAttr in_reductions, ::mlir::ValueRange reduction_vars, /*optional*/::mlir::ArrayAttr reductions, /*optional*/::mlir::Value priority, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars, /*optional*/::mlir::Value grain_size, /*optional*/::mlir::Value num_tasks, /*optional*/bool nogroup = false);
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
    assert(index < 7 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  /// Returns the reduction variables
  SmallVector<Value> getAllReductionVars();
  void getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects);
};
} // namespace omp
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::TaskLoopOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::TaskOp declarations
//===----------------------------------------------------------------------===//

class TaskOpAdaptor {
public:
  TaskOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  TaskOpAdaptor(TaskOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getIfExpr();
  ::mlir::Value getFinalExpr();
  ::mlir::ValueRange getInReductionVars();
  ::mlir::Value getPriority();
  ::mlir::ValueRange getAllocateVars();
  ::mlir::ValueRange getAllocatorsVars();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::UnitAttr getUntiedAttr();
  bool getUntied();
  ::mlir::UnitAttr getMergeableAttr();
  bool getMergeable();
  ::mlir::ArrayAttr getInReductionsAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getInReductions();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &getRegion();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class TaskOp : public ::mlir::Op<TaskOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::omp::OutlineableOpenMPOpInterface::Trait, ::mlir::OpTrait::AutomaticAllocationScope, ::mlir::omp::ReductionClauseInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = TaskOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("in_reductions"), ::llvm::StringRef("mergeable"), ::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("untied")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getInReductionsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getInReductionsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getMergeableAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getMergeableAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getUntiedAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getUntiedAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.task");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::IntegerType> getIfExpr();
  ::mlir::TypedValue<::mlir::IntegerType> getFinalExpr();
  ::mlir::Operation::operand_range getInReductionVars();
  ::mlir::TypedValue<::mlir::IntegerType> getPriority();
  ::mlir::Operation::operand_range getAllocateVars();
  ::mlir::Operation::operand_range getAllocatorsVars();
  ::mlir::MutableOperandRange getIfExprMutable();
  ::mlir::MutableOperandRange getFinalExprMutable();
  ::mlir::MutableOperandRange getInReductionVarsMutable();
  ::mlir::MutableOperandRange getPriorityMutable();
  ::mlir::MutableOperandRange getAllocateVarsMutable();
  ::mlir::MutableOperandRange getAllocatorsVarsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &getRegion();
  ::mlir::UnitAttr getUntiedAttr();
  bool getUntied();
  ::mlir::UnitAttr getMergeableAttr();
  bool getMergeable();
  ::mlir::ArrayAttr getInReductionsAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getInReductions();
  void setUntiedAttr(::mlir::UnitAttr attr);
  void setMergeableAttr(::mlir::UnitAttr attr);
  void setInReductionsAttr(::mlir::ArrayAttr attr);
  ::mlir::Attribute removeUntiedAttr();
  ::mlir::Attribute removeMergeableAttr();
  ::mlir::Attribute removeIn_reductionsAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value final_expr, /*optional*/::mlir::UnitAttr untied, /*optional*/::mlir::UnitAttr mergeable, ::mlir::ValueRange in_reduction_vars, /*optional*/::mlir::ArrayAttr in_reductions, /*optional*/::mlir::Value priority, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value final_expr, /*optional*/::mlir::UnitAttr untied, /*optional*/::mlir::UnitAttr mergeable, ::mlir::ValueRange in_reduction_vars, /*optional*/::mlir::ArrayAttr in_reductions, /*optional*/::mlir::Value priority, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value final_expr, /*optional*/bool untied, /*optional*/bool mergeable, ::mlir::ValueRange in_reduction_vars, /*optional*/::mlir::ArrayAttr in_reductions, /*optional*/::mlir::Value priority, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value final_expr, /*optional*/bool untied, /*optional*/bool mergeable, ::mlir::ValueRange in_reduction_vars, /*optional*/::mlir::ArrayAttr in_reductions, /*optional*/::mlir::Value priority, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars);
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
    assert(index < 4 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  /// Returns the reduction variables
  SmallVector<Value> getReductionVars() {
    return SmallVector<Value>(getInReductionVars().begin(),
                              getInReductionVars().end());
  }
};
} // namespace omp
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::TaskOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::TaskwaitOp declarations
//===----------------------------------------------------------------------===//

class TaskwaitOpAdaptor {
public:
  TaskwaitOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  TaskwaitOpAdaptor(TaskwaitOp op);

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
class TaskwaitOp : public ::mlir::Op<TaskwaitOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = TaskwaitOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.taskwait");
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
} // namespace omp
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::TaskwaitOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::TaskyieldOp declarations
//===----------------------------------------------------------------------===//

class TaskyieldOpAdaptor {
public:
  TaskyieldOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  TaskyieldOpAdaptor(TaskyieldOp op);

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
class TaskyieldOp : public ::mlir::Op<TaskyieldOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = TaskyieldOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.taskyield");
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
} // namespace omp
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::TaskyieldOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::TerminatorOp declarations
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
class TerminatorOp : public ::mlir::Op<TerminatorOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::OpTrait::IsTerminator, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = TerminatorOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.terminator");
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
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace omp
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::TerminatorOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::ThreadprivateOp declarations
//===----------------------------------------------------------------------===//

class ThreadprivateOpAdaptor {
public:
  ThreadprivateOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  ThreadprivateOpAdaptor(ThreadprivateOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getSymAddr();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class ThreadprivateOp : public ::mlir::Op<ThreadprivateOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::omp::PointerLikeType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ThreadprivateOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.threadprivate");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::omp::PointerLikeType> getSymAddr();
  ::mlir::MutableOperandRange getSymAddrMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::omp::PointerLikeType> getTlsAddr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type tls_addr, ::mlir::Value sym_addr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value sym_addr);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
  /// The number of variable operands.
  unsigned getNumVariableOperands() {
    assert(getSymAddr() && "expected one variable operand");
    return 1;
  }

  /// The i-th variable operand passed.
  Value getVariableOperand(unsigned i) {
    assert(i == 0 && "invalid index position for an operand");
    return getSymAddr();
  }
};
} // namespace omp
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::ThreadprivateOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::WsLoopOp declarations
//===----------------------------------------------------------------------===//

class WsLoopOpAdaptor {
public:
  WsLoopOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  WsLoopOpAdaptor(WsLoopOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange getLowerBound();
  ::mlir::ValueRange getUpperBound();
  ::mlir::ValueRange getStep();
  ::mlir::ValueRange getLinearVars();
  ::mlir::ValueRange getLinearStepVars();
  ::mlir::ValueRange getReductionVars();
  ::mlir::Value getScheduleChunkVar();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::ArrayAttr getReductionsAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getReductions();
  ::mlir::omp::ClauseScheduleKindAttr getScheduleValAttr();
  ::llvm::Optional<::mlir::omp::ClauseScheduleKind> getScheduleVal();
  ::mlir::omp::ScheduleModifierAttr getScheduleModifierAttr();
  ::llvm::Optional<::mlir::omp::ScheduleModifier> getScheduleModifier();
  ::mlir::UnitAttr getSimdModifierAttr();
  bool getSimdModifier();
  ::mlir::UnitAttr getNowaitAttr();
  bool getNowait();
  ::mlir::IntegerAttr getOrderedValAttr();
  ::llvm::Optional<uint64_t> getOrderedVal();
  ::mlir::omp::ClauseOrderKindAttr getOrderValAttr();
  ::llvm::Optional<::mlir::omp::ClauseOrderKind> getOrderVal();
  ::mlir::UnitAttr getInclusiveAttr();
  bool getInclusive();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &getRegion();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class WsLoopOp : public ::mlir::Op<WsLoopOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::OpTrait::HasRecursiveMemoryEffects, ::mlir::omp::ReductionClauseInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = WsLoopOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("inclusive"), ::llvm::StringRef("nowait"), ::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("order_val"), ::llvm::StringRef("ordered_val"), ::llvm::StringRef("reductions"), ::llvm::StringRef("schedule_modifier"), ::llvm::StringRef("schedule_val"), ::llvm::StringRef("simd_modifier")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getInclusiveAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getInclusiveAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getNowaitAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getNowaitAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getOrderValAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getOrderValAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getOrderedValAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getOrderedValAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr getReductionsAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr getReductionsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  ::mlir::StringAttr getScheduleModifierAttrName() {
    return getAttributeNameForIndex(6);
  }

  static ::mlir::StringAttr getScheduleModifierAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 6);
  }

  ::mlir::StringAttr getScheduleValAttrName() {
    return getAttributeNameForIndex(7);
  }

  static ::mlir::StringAttr getScheduleValAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 7);
  }

  ::mlir::StringAttr getSimdModifierAttrName() {
    return getAttributeNameForIndex(8);
  }

  static ::mlir::StringAttr getSimdModifierAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 8);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.wsloop");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getLowerBound();
  ::mlir::Operation::operand_range getUpperBound();
  ::mlir::Operation::operand_range getStep();
  ::mlir::Operation::operand_range getLinearVars();
  ::mlir::Operation::operand_range getLinearStepVars();
  ::mlir::Operation::operand_range getReductionVars();
  ::mlir::Value getScheduleChunkVar();
  ::mlir::MutableOperandRange getLowerBoundMutable();
  ::mlir::MutableOperandRange getUpperBoundMutable();
  ::mlir::MutableOperandRange getStepMutable();
  ::mlir::MutableOperandRange getLinearVarsMutable();
  ::mlir::MutableOperandRange getLinearStepVarsMutable();
  ::mlir::MutableOperandRange getReductionVarsMutable();
  ::mlir::MutableOperandRange getScheduleChunkVarMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &getRegion();
  ::mlir::ArrayAttr getReductionsAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getReductions();
  ::mlir::omp::ClauseScheduleKindAttr getScheduleValAttr();
  ::llvm::Optional<::mlir::omp::ClauseScheduleKind> getScheduleVal();
  ::mlir::omp::ScheduleModifierAttr getScheduleModifierAttr();
  ::llvm::Optional<::mlir::omp::ScheduleModifier> getScheduleModifier();
  ::mlir::UnitAttr getSimdModifierAttr();
  bool getSimdModifier();
  ::mlir::UnitAttr getNowaitAttr();
  bool getNowait();
  ::mlir::IntegerAttr getOrderedValAttr();
  ::llvm::Optional<uint64_t> getOrderedVal();
  ::mlir::omp::ClauseOrderKindAttr getOrderValAttr();
  ::llvm::Optional<::mlir::omp::ClauseOrderKind> getOrderVal();
  ::mlir::UnitAttr getInclusiveAttr();
  bool getInclusive();
  void setReductionsAttr(::mlir::ArrayAttr attr);
  void setScheduleValAttr(::mlir::omp::ClauseScheduleKindAttr attr);
  void setScheduleModifierAttr(::mlir::omp::ScheduleModifierAttr attr);
  void setSimdModifierAttr(::mlir::UnitAttr attr);
  void setNowaitAttr(::mlir::UnitAttr attr);
  void setOrderedValAttr(::mlir::IntegerAttr attr);
  void setOrderValAttr(::mlir::omp::ClauseOrderKindAttr attr);
  void setInclusiveAttr(::mlir::UnitAttr attr);
  ::mlir::Attribute removeReductionsAttr();
  ::mlir::Attribute removeSchedule_valAttr();
  ::mlir::Attribute removeSchedule_modifierAttr();
  ::mlir::Attribute removeSimd_modifierAttr();
  ::mlir::Attribute removeNowaitAttr();
  ::mlir::Attribute removeOrdered_valAttr();
  ::mlir::Attribute removeOrder_valAttr();
  ::mlir::Attribute removeInclusiveAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange lowerBound, ValueRange upperBound, ValueRange step, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange lowerBound, ::mlir::ValueRange upperBound, ::mlir::ValueRange step, ::mlir::ValueRange linear_vars, ::mlir::ValueRange linear_step_vars, ::mlir::ValueRange reduction_vars, /*optional*/::mlir::ArrayAttr reductions, /*optional*/::mlir::omp::ClauseScheduleKindAttr schedule_val, /*optional*/::mlir::Value schedule_chunk_var, /*optional*/::mlir::omp::ScheduleModifierAttr schedule_modifier, /*optional*/::mlir::UnitAttr simd_modifier, /*optional*/::mlir::UnitAttr nowait, /*optional*/::mlir::IntegerAttr ordered_val, /*optional*/::mlir::omp::ClauseOrderKindAttr order_val, /*optional*/::mlir::UnitAttr inclusive);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange lowerBound, ::mlir::ValueRange upperBound, ::mlir::ValueRange step, ::mlir::ValueRange linear_vars, ::mlir::ValueRange linear_step_vars, ::mlir::ValueRange reduction_vars, /*optional*/::mlir::ArrayAttr reductions, /*optional*/::mlir::omp::ClauseScheduleKindAttr schedule_val, /*optional*/::mlir::Value schedule_chunk_var, /*optional*/::mlir::omp::ScheduleModifierAttr schedule_modifier, /*optional*/::mlir::UnitAttr simd_modifier, /*optional*/::mlir::UnitAttr nowait, /*optional*/::mlir::IntegerAttr ordered_val, /*optional*/::mlir::omp::ClauseOrderKindAttr order_val, /*optional*/::mlir::UnitAttr inclusive);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange lowerBound, ::mlir::ValueRange upperBound, ::mlir::ValueRange step, ::mlir::ValueRange linear_vars, ::mlir::ValueRange linear_step_vars, ::mlir::ValueRange reduction_vars, /*optional*/::mlir::ArrayAttr reductions, /*optional*/::mlir::omp::ClauseScheduleKindAttr schedule_val, /*optional*/::mlir::Value schedule_chunk_var, /*optional*/::mlir::omp::ScheduleModifierAttr schedule_modifier, /*optional*/bool simd_modifier, /*optional*/bool nowait, /*optional*/::mlir::IntegerAttr ordered_val, /*optional*/::mlir::omp::ClauseOrderKindAttr order_val, /*optional*/bool inclusive = false);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange lowerBound, ::mlir::ValueRange upperBound, ::mlir::ValueRange step, ::mlir::ValueRange linear_vars, ::mlir::ValueRange linear_step_vars, ::mlir::ValueRange reduction_vars, /*optional*/::mlir::ArrayAttr reductions, /*optional*/::mlir::omp::ClauseScheduleKindAttr schedule_val, /*optional*/::mlir::Value schedule_chunk_var, /*optional*/::mlir::omp::ScheduleModifierAttr schedule_modifier, /*optional*/bool simd_modifier, /*optional*/bool nowait, /*optional*/::mlir::IntegerAttr ordered_val, /*optional*/::mlir::omp::ClauseOrderKindAttr order_val, /*optional*/bool inclusive = false);
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
    assert(index < 9 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  /// Returns the number of loops in the worksharing-loop nest.
  unsigned getNumLoops() { return getLowerBound().size(); }

  /// Returns the number of reduction variables.
  unsigned getNumReductionVars() { return getReductionVars().size(); }
};
} // namespace omp
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::WsLoopOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::YieldOp declarations
//===----------------------------------------------------------------------===//

class YieldOpAdaptor {
public:
  YieldOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  YieldOpAdaptor(YieldOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange getResults();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class YieldOp : public ::mlir::Op<YieldOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::HasParent<WsLoopOp, ReductionDeclareOp, AtomicUpdateOp, SimdLoopOp>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::ReturnLike, ::mlir::OpTrait::IsTerminator> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = YieldOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.yield");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getResults();
  ::mlir::MutableOperandRange getResultsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange results);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace omp
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::YieldOp)


#endif  // GET_OP_CLASSES

