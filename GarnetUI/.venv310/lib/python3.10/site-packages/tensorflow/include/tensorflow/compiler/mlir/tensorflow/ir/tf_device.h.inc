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
namespace tf_device {
class ClusterFuncOp;
} // namespace tf_device
} // namespace mlir
namespace mlir {
namespace tf_device {
class ClusterOp;
} // namespace tf_device
} // namespace mlir
namespace mlir {
namespace tf_device {
class LaunchFuncOp;
} // namespace tf_device
} // namespace mlir
namespace mlir {
namespace tf_device {
class LaunchOp;
} // namespace tf_device
} // namespace mlir
namespace mlir {
namespace tf_device {
class ParallelExecuteOp;
} // namespace tf_device
} // namespace mlir
namespace mlir {
namespace tf_device {
class ReceiveOp;
} // namespace tf_device
} // namespace mlir
namespace mlir {
namespace tf_device {
class RemoteRunOp;
} // namespace tf_device
} // namespace mlir
namespace mlir {
namespace tf_device {
class ReplicateOp;
} // namespace tf_device
} // namespace mlir
namespace mlir {
namespace tf_device {
class ReturnOp;
} // namespace tf_device
} // namespace mlir
namespace mlir {
namespace tf_device {
class SendOp;
} // namespace tf_device
} // namespace mlir
#endif

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace tf_device {

//===----------------------------------------------------------------------===//
// ::mlir::tf_device::ClusterFuncOp declarations
//===----------------------------------------------------------------------===//

class ClusterFuncOpAdaptor {
public:
  ClusterFuncOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  ClusterFuncOpAdaptor(ClusterFuncOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange operands();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::FlatSymbolRefAttr getFuncAttr();
  ::llvm::StringRef getFunc();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class ClusterFuncOp : public ::mlir::Op<ClusterFuncOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ClusterFuncOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("func")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getFuncAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getFuncAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tf_device.cluster_func");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range operands();
  ::mlir::MutableOperandRange operandsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range getResults();
  ::mlir::FlatSymbolRefAttr getFuncAttr();
  ::llvm::StringRef getFunc();
  void setFuncAttr(::mlir::FlatSymbolRefAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::mlir::FlatSymbolRefAttr func, ::mlir::ValueRange operands);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::llvm::StringRef func, ::mlir::ValueRange operands);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
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
  // returns the function that this operation will launch.
  func::FuncOp getFuncOp() {
    return SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(*this, getFuncAttr());
  }
};
} // namespace tf_device
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tf_device::ClusterFuncOp)

namespace mlir {
namespace tf_device {

//===----------------------------------------------------------------------===//
// ::mlir::tf_device::ClusterOp declarations
//===----------------------------------------------------------------------===//

class ClusterOpAdaptor {
public:
  ClusterOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  ClusterOpAdaptor(ClusterOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::StringAttr getPolicyAttr();
  ::llvm::Optional< ::llvm::StringRef > getPolicy();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &getBody();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class ClusterOp : public ::mlir::Op<ClusterOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::SingleBlockImplicitTerminator<ReturnOp>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ClusterOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("policy")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getPolicyAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getPolicyAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tf_device.cluster");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range getResults();
  ::mlir::Region &getBody();
  ::mlir::StringAttr getPolicyAttr();
  ::llvm::Optional< ::llvm::StringRef > getPolicy();
  void setPolicyAttr(::mlir::StringAttr attr);
  ::mlir::Attribute removePolicyAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTypes);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, /*optional*/::mlir::StringAttr policy);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
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
  Block &GetBody() { return getOperation()->getRegion(0).front(); }
};
} // namespace tf_device
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tf_device::ClusterOp)

namespace mlir {
namespace tf_device {

//===----------------------------------------------------------------------===//
// ::mlir::tf_device::LaunchFuncOp declarations
//===----------------------------------------------------------------------===//

class LaunchFuncOpAdaptor {
public:
  LaunchFuncOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  LaunchFuncOpAdaptor(LaunchFuncOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange operands();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::StringAttr getDeviceAttr();
  ::llvm::StringRef getDevice();
  ::mlir::FlatSymbolRefAttr getFuncAttr();
  ::llvm::StringRef getFunc();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class LaunchFuncOp : public ::mlir::Op<LaunchFuncOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = LaunchFuncOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("device"), ::llvm::StringRef("func")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getDeviceAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getDeviceAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getFuncAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getFuncAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tf_device.launch_func");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range operands();
  ::mlir::MutableOperandRange operandsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range getResults();
  ::mlir::StringAttr getDeviceAttr();
  ::llvm::StringRef getDevice();
  ::mlir::FlatSymbolRefAttr getFuncAttr();
  ::llvm::StringRef getFunc();
  void setDeviceAttr(::mlir::StringAttr attr);
  void setFuncAttr(::mlir::FlatSymbolRefAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::mlir::StringAttr device, ::mlir::FlatSymbolRefAttr func, ::mlir::ValueRange operands);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::llvm::StringRef device, ::llvm::StringRef func, ::mlir::ValueRange operands);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
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
} // namespace tf_device
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tf_device::LaunchFuncOp)

namespace mlir {
namespace tf_device {

//===----------------------------------------------------------------------===//
// ::mlir::tf_device::LaunchOp declarations
//===----------------------------------------------------------------------===//

class LaunchOpAdaptor {
public:
  LaunchOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  LaunchOpAdaptor(LaunchOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::StringAttr getDeviceAttr();
  ::llvm::StringRef getDevice();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &getBody();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class LaunchOp : public ::mlir::Op<LaunchOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::SingleBlockImplicitTerminator<ReturnOp>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = LaunchOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("device")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getDeviceAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getDeviceAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tf_device.launch");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range getResults();
  ::mlir::Region &getBody();
  ::mlir::StringAttr getDeviceAttr();
  ::llvm::StringRef getDevice();
  void setDeviceAttr(::mlir::StringAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, StringAttr device, TypeRange result_types);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::mlir::StringAttr device);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::llvm::StringRef device);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
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
  Block &GetBody() { return getOperation()->getRegion(0).front(); }
  bool WrapsSingleOp();
};
} // namespace tf_device
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tf_device::LaunchOp)

namespace mlir {
namespace tf_device {

//===----------------------------------------------------------------------===//
// ::mlir::tf_device::ParallelExecuteOp declarations
//===----------------------------------------------------------------------===//

class ParallelExecuteOpAdaptor {
public:
  ParallelExecuteOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  ParallelExecuteOpAdaptor(ParallelExecuteOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::RegionRange getRegions();
  ::mlir::RegionRange regions();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class ParallelExecuteOp : public ::mlir::Op<ParallelExecuteOp, ::mlir::OpTrait::VariadicRegions, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::SingleBlockImplicitTerminator<ReturnOp>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ParallelExecuteOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tf_device.parallel_execute");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range getExecuteOutputs();
  ::mlir::MutableArrayRef<::mlir::Region> regions();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, int num_regions, TypeRange output_types);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange execute_outputs, unsigned regionsCount);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes, unsigned numRegions);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
public:
  Block& GetRegionBlockWithIndex(unsigned index);
  Operation::result_range GetRegionOutputs(unsigned region_index);

  // Checks if a tf_device.parallel_execute index'th region block wraps a
  // single operation and the single operation results are perfectly forwarded
  // to the region block's return.
  bool RegionWrapsSingleOp(unsigned index);
};
} // namespace tf_device
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tf_device::ParallelExecuteOp)

namespace mlir {
namespace tf_device {

//===----------------------------------------------------------------------===//
// ::mlir::tf_device::ReceiveOp declarations
//===----------------------------------------------------------------------===//

class ReceiveOpAdaptor {
public:
  ReceiveOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  ReceiveOpAdaptor(ReceiveOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::StringAttr getKeyAttr();
  ::llvm::StringRef getKey();
  ::mlir::StringAttr getSrcHostAttr();
  ::llvm::StringRef getSrcHost();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class ReceiveOp : public ::mlir::Op<ReceiveOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ReceiveOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("key"), ::llvm::StringRef("src_host")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getKeyAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getKeyAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getSrcHostAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getSrcHostAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tf_device.receive");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value getResult();
  ::mlir::StringAttr getKeyAttr();
  ::llvm::StringRef getKey();
  ::mlir::StringAttr getSrcHostAttr();
  ::llvm::StringRef getSrcHost();
  void setKeyAttr(::mlir::StringAttr attr);
  void setSrcHostAttr(::mlir::StringAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::StringAttr key, ::mlir::StringAttr src_host);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr key, ::mlir::StringAttr src_host);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::llvm::StringRef key, ::llvm::StringRef src_host);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef key, ::llvm::StringRef src_host);
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
} // namespace tf_device
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tf_device::ReceiveOp)

namespace mlir {
namespace tf_device {

//===----------------------------------------------------------------------===//
// ::mlir::tf_device::RemoteRunOp declarations
//===----------------------------------------------------------------------===//

class RemoteRunOpAdaptor {
public:
  RemoteRunOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  RemoteRunOpAdaptor(RemoteRunOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange getCalleeArgs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::StringAttr getHostAttr();
  ::llvm::StringRef getHost();
  ::mlir::FlatSymbolRefAttr getCalleeAttr();
  ::llvm::StringRef getCallee();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class RemoteRunOp : public ::mlir::Op<RemoteRunOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::SingleBlockImplicitTerminator<ReturnOp>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = RemoteRunOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("callee"), ::llvm::StringRef("host")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getCalleeAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getCalleeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getHostAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getHostAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tf_device.remote_run");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getCalleeArgs();
  ::mlir::MutableOperandRange getCalleeArgsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range getResults();
  ::mlir::StringAttr getHostAttr();
  ::llvm::StringRef getHost();
  ::mlir::FlatSymbolRefAttr getCalleeAttr();
  ::llvm::StringRef getCallee();
  void setHostAttr(::mlir::StringAttr attr);
  void setCalleeAttr(::mlir::FlatSymbolRefAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::mlir::StringAttr host, ::mlir::FlatSymbolRefAttr callee, ::mlir::ValueRange callee_args);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::llvm::StringRef host, ::llvm::StringRef callee, ::mlir::ValueRange callee_args);
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
} // namespace tf_device
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tf_device::RemoteRunOp)

namespace mlir {
namespace tf_device {

//===----------------------------------------------------------------------===//
// ::mlir::tf_device::ReplicateOp declarations
//===----------------------------------------------------------------------===//

class ReplicateOpAdaptor {
public:
  ReplicateOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  ReplicateOpAdaptor(ReplicateOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange getReplicatedInputs();
  ::mlir::ValueRange getPackedInputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseI32ArrayAttr getOperandSegmentSizesAttr();
  ::llvm::ArrayRef<int32_t> getOperandSegmentSizes();
  ::mlir::IntegerAttr getNAttr();
  uint32_t getN();
  ::mlir::DictionaryAttr getDevicesAttr();
  ::llvm::Optional< ::mlir::DictionaryAttr > getDevices();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &getBody();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class ReplicateOp : public ::mlir::Op<ReplicateOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::SingleBlockImplicitTerminator<ReturnOp>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ReplicateOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("devices"), ::llvm::StringRef("n"), ::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getDevicesAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getDevicesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getNAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getNAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tf_device.replicate");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getReplicatedInputs();
  ::mlir::Operation::operand_range getPackedInputs();
  ::mlir::MutableOperandRange getReplicatedInputsMutable();
  ::mlir::MutableOperandRange getPackedInputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range getReplicatedOutputs();
  ::mlir::Region &getBody();
  ::mlir::DenseI32ArrayAttr getOperandSegmentSizesAttr();
  ::llvm::ArrayRef<int32_t> getOperandSegmentSizes();
  ::mlir::IntegerAttr getNAttr();
  uint32_t getN();
  ::mlir::DictionaryAttr getDevicesAttr();
  ::llvm::Optional< ::mlir::DictionaryAttr > getDevices();
  void setOperandSegmentSizesAttr(::mlir::DenseI32ArrayAttr attr);
  void setNAttr(::mlir::IntegerAttr attr);
  void setDevicesAttr(::mlir::DictionaryAttr attr);
  ::mlir::Attribute removeDevicesAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, int n, const llvm::SmallDenseMap<StringRef, llvm::SmallVector<StringRef, 4>>&devices, llvm::ArrayRef<std::pair<ValueRange, Type>> replicated_inputs, ValueRange packed_inputs, TypeRange replica_output_types);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, int n, llvm::Optional<DictionaryAttr> devices, llvm::ArrayRef<std::pair<ValueRange, Type>> replicated_inputs, ValueRange packed_inputs, TypeRange replica_output_types);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange replicated_outputs, ::mlir::ValueRange replicated_inputs, ::mlir::ValueRange packed_inputs, ::mlir::DenseI32ArrayAttr operand_segment_sizes, ::mlir::IntegerAttr n, /*optional*/::mlir::DictionaryAttr devices);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange replicated_outputs, ::mlir::ValueRange replicated_inputs, ::mlir::ValueRange packed_inputs, ::llvm::ArrayRef<int32_t> operand_segment_sizes, uint32_t n, /*optional*/::mlir::DictionaryAttr devices);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
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
    assert(index < 3 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  Block &GetBody() { return getOperation()->getRegion(0).front(); }
  unsigned GetNumReplicatedBlockArguments();
  unsigned GetNumPackedBlockArguments();
  llvm::ArrayRef<BlockArgument> GetPackedBlockArguments();
  llvm::ArrayRef<BlockArgument> GetReplicatedBlockArguments();
  bool IsReplicatedBlockArgument(BlockArgument block_arg);
  bool IsPackedBlockArgument(BlockArgument block_arg);
  unsigned GetReplicaOperandIndexForBlockArgument(BlockArgument block_arg, unsigned replica);
  Value GetReplicaOperandForBlockArgument(BlockArgument block_arg, unsigned replica);
  MutableArrayRef<OpOperand> GetOperandsForBlockArgument(BlockArgument block_arg);
  bool WrapsSingleOp();
};
} // namespace tf_device
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tf_device::ReplicateOp)

namespace mlir {
namespace tf_device {

//===----------------------------------------------------------------------===//
// ::mlir::tf_device::ReturnOp declarations
//===----------------------------------------------------------------------===//

class ReturnOpAdaptor {
public:
  ReturnOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  ReturnOpAdaptor(ReturnOp op);

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
class ReturnOp : public ::mlir::Op<ReturnOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::ReturnLike, ::mlir::OpTrait::IsTerminator> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ReturnOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tf_device.return");
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
} // namespace tf_device
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tf_device::ReturnOp)

namespace mlir {
namespace tf_device {

//===----------------------------------------------------------------------===//
// ::mlir::tf_device::SendOp declarations
//===----------------------------------------------------------------------===//

class SendOpAdaptor {
public:
  SendOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  SendOpAdaptor(SendOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getValue();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::StringAttr getKeyAttr();
  ::llvm::StringRef getKey();
  ::mlir::StringAttr getDstHostAttr();
  ::llvm::StringRef getDstHost();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class SendOp : public ::mlir::Op<SendOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = SendOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("dst_host"), ::llvm::StringRef("key")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getDstHostAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getDstHostAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getKeyAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getKeyAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tf_device.send");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value getValue();
  ::mlir::MutableOperandRange getValueMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::StringAttr getKeyAttr();
  ::llvm::StringRef getKey();
  ::mlir::StringAttr getDstHostAttr();
  ::llvm::StringRef getDstHost();
  void setKeyAttr(::mlir::StringAttr attr);
  void setDstHostAttr(::mlir::StringAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::StringAttr key, ::mlir::StringAttr dst_host);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::StringAttr key, ::mlir::StringAttr dst_host);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::llvm::StringRef key, ::llvm::StringRef dst_host);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::llvm::StringRef key, ::llvm::StringRef dst_host);
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
} // namespace tf_device
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tf_device::SendOp)


#endif  // GET_OP_CLASSES

