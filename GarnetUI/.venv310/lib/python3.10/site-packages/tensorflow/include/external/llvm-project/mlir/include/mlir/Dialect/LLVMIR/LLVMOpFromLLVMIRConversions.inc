if (inst->getOpcode() == llvm::Instruction::AShr) {

    mapValue(inst) = odsBuilder.create<::mlir::LLVM::AShrOp>(translateLoc(inst->getDebugLoc()), processValue(llvmOperands[0]), processValue(llvmOperands[1]));
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::Add) {

    mapValue(inst) = odsBuilder.create<::mlir::LLVM::AddOp>(translateLoc(inst->getDebugLoc()), processValue(llvmOperands[0]), processValue(llvmOperands[1]));
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::AddrSpaceCast) {

    mapValue(inst) = odsBuilder.create<::mlir::LLVM::AddrSpaceCastOp>(
      translateLoc(inst->getDebugLoc()), convertType(inst->getType()), processValue(llvmOperands[0]));
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::Alloca) {

    auto *allocaInst = cast<llvm::AllocaInst>(inst);
    Type allocatedType = convertType(allocaInst->getAllocatedType());
    unsigned alignment = allocaInst->getAlign().value();
    mapValue(inst) = odsBuilder.create<LLVM::AllocaOp>(
      translateLoc(inst->getDebugLoc()), convertType(inst->getType()), allocatedType, processValue(llvmOperands[0]), alignment);
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::And) {

    mapValue(inst) = odsBuilder.create<::mlir::LLVM::AndOp>(translateLoc(inst->getDebugLoc()), processValue(llvmOperands[0]), processValue(llvmOperands[1]));
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::AtomicCmpXchg) {

    auto *cmpXchgInst = cast<llvm::AtomicCmpXchgInst>(inst);
    mapValue(inst) = odsBuilder.create<LLVM::AtomicCmpXchgOp>(
      translateLoc(inst->getDebugLoc()), convertType(inst->getType()), processValue(llvmOperands[0]), processValue(llvmOperands[1]), processValue(llvmOperands[2]),
      getLLVMAtomicOrdering(cmpXchgInst->getSuccessOrdering()),
      getLLVMAtomicOrdering(cmpXchgInst->getFailureOrdering()));
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::AtomicRMW) {

    auto *atomicInst = cast<llvm::AtomicRMWInst>(inst);
    mapValue(inst) = odsBuilder.create<LLVM::AtomicRMWOp>(translateLoc(inst->getDebugLoc()), convertType(inst->getType()),
        getLLVMAtomicBinOp(atomicInst->getOperation()), processValue(llvmOperands[0]), processValue(llvmOperands[1]),
        getLLVMAtomicOrdering(atomicInst->getOrdering()));
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::BitCast) {

    mapValue(inst) = odsBuilder.create<::mlir::LLVM::BitcastOp>(
      translateLoc(inst->getDebugLoc()), convertType(inst->getType()), processValue(llvmOperands[0]));
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::ExtractElement) {

    mapValue(inst) = odsBuilder.create<LLVM::ExtractElementOp>(
      translateLoc(inst->getDebugLoc()), processValue(llvmOperands[0]), processValue(llvmOperands[1]));
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::ExtractValue) {

    auto *evInst = cast<llvm::ExtractValueInst>(inst);
    mapValue(inst) = odsBuilder.create<LLVM::ExtractValueOp>(translateLoc(inst->getDebugLoc()),
      processValue(llvmOperands[0]), getPositionFromIndices(evInst->getIndices()));
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::FAdd) {

    mapValue(inst) = odsBuilder.create<::mlir::LLVM::FAddOp>(translateLoc(inst->getDebugLoc()), processValue(llvmOperands[0]), processValue(llvmOperands[1]));
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::FCmp) {

    auto *fCmpInst = cast<llvm::FCmpInst>(inst);
    mapValue(inst) = odsBuilder.create<::mlir::LLVM::FCmpOp>(
      translateLoc(inst->getDebugLoc()), getFCmpPredicate(fCmpInst->getPredicate()), processValue(llvmOperands[0]), processValue(llvmOperands[1]));
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::FDiv) {

    mapValue(inst) = odsBuilder.create<::mlir::LLVM::FDivOp>(translateLoc(inst->getDebugLoc()), processValue(llvmOperands[0]), processValue(llvmOperands[1]));
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::FMul) {

    mapValue(inst) = odsBuilder.create<::mlir::LLVM::FMulOp>(translateLoc(inst->getDebugLoc()), processValue(llvmOperands[0]), processValue(llvmOperands[1]));
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::FNeg) {

    mapValue(inst) = odsBuilder.create<::mlir::LLVM::FNegOp>(translateLoc(inst->getDebugLoc()), processValue(llvmOperands[0]));
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::FPExt) {

    mapValue(inst) = odsBuilder.create<::mlir::LLVM::FPExtOp>(
      translateLoc(inst->getDebugLoc()), convertType(inst->getType()), processValue(llvmOperands[0]));
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::FPToSI) {

    mapValue(inst) = odsBuilder.create<::mlir::LLVM::FPToSIOp>(
      translateLoc(inst->getDebugLoc()), convertType(inst->getType()), processValue(llvmOperands[0]));
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::FPToUI) {

    mapValue(inst) = odsBuilder.create<::mlir::LLVM::FPToUIOp>(
      translateLoc(inst->getDebugLoc()), convertType(inst->getType()), processValue(llvmOperands[0]));
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::FPTrunc) {

    mapValue(inst) = odsBuilder.create<::mlir::LLVM::FPTruncOp>(
      translateLoc(inst->getDebugLoc()), convertType(inst->getType()), processValue(llvmOperands[0]));
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::FRem) {

    mapValue(inst) = odsBuilder.create<::mlir::LLVM::FRemOp>(translateLoc(inst->getDebugLoc()), processValue(llvmOperands[0]), processValue(llvmOperands[1]));
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::FSub) {

    mapValue(inst) = odsBuilder.create<::mlir::LLVM::FSubOp>(translateLoc(inst->getDebugLoc()), processValue(llvmOperands[0]), processValue(llvmOperands[1]));
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::Fence) {

    llvm::FenceInst *fenceInst = cast<llvm::FenceInst>(inst);
    odsBuilder.create<LLVM::FenceOp>(
      translateLoc(inst->getDebugLoc()),
      getLLVMAtomicOrdering(fenceInst->getOrdering()),
      getLLVMSyncScope(fenceInst));
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::Freeze) {

    mapValue(inst) = odsBuilder.create<LLVM::FreezeOp>(translateLoc(inst->getDebugLoc()), processValue(llvmOperands[0]));
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::ICmp) {

    auto *iCmpInst = cast<llvm::ICmpInst>(inst);
    mapValue(inst) = odsBuilder.create<::mlir::LLVM::ICmpOp>(
      translateLoc(inst->getDebugLoc()), getICmpPredicate(iCmpInst->getPredicate()), processValue(llvmOperands[0]), processValue(llvmOperands[1]));
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::InsertElement) {

    mapValue(inst) = odsBuilder.create<LLVM::InsertElementOp>(
      translateLoc(inst->getDebugLoc()), processValue(llvmOperands[0]), processValue(llvmOperands[1]), processValue(llvmOperands[2]));
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::InsertValue) {

    auto *ivInst = cast<llvm::InsertValueInst>(inst);
    mapValue(inst) = odsBuilder.create<LLVM::InsertValueOp>(translateLoc(inst->getDebugLoc()),
      processValue(llvmOperands[0]), processValue(llvmOperands[1]), getPositionFromIndices(ivInst->getIndices()));
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::IntToPtr) {

    mapValue(inst) = odsBuilder.create<::mlir::LLVM::IntToPtrOp>(
      translateLoc(inst->getDebugLoc()), convertType(inst->getType()), processValue(llvmOperands[0]));
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::LShr) {

    mapValue(inst) = odsBuilder.create<::mlir::LLVM::LShrOp>(translateLoc(inst->getDebugLoc()), processValue(llvmOperands[0]), processValue(llvmOperands[1]));
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::Load) {

    mapValue(inst) = odsBuilder.create<LLVM::LoadOp>(translateLoc(inst->getDebugLoc()), convertType(inst->getType()), processValue(llvmOperands[0]));
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::Mul) {

    mapValue(inst) = odsBuilder.create<::mlir::LLVM::MulOp>(translateLoc(inst->getDebugLoc()), processValue(llvmOperands[0]), processValue(llvmOperands[1]));
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::Or) {

    mapValue(inst) = odsBuilder.create<::mlir::LLVM::OrOp>(translateLoc(inst->getDebugLoc()), processValue(llvmOperands[0]), processValue(llvmOperands[1]));
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::PtrToInt) {

    mapValue(inst) = odsBuilder.create<::mlir::LLVM::PtrToIntOp>(
      translateLoc(inst->getDebugLoc()), convertType(inst->getType()), processValue(llvmOperands[0]));
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::Resume) {

    odsBuilder.create<LLVM::ResumeOp>(translateLoc(inst->getDebugLoc()), processValue(llvmOperands[0]));
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::Ret) {

    odsBuilder.create<LLVM::ReturnOp>(translateLoc(inst->getDebugLoc()), processValues(llvmOperands));
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::SDiv) {

    mapValue(inst) = odsBuilder.create<::mlir::LLVM::SDivOp>(translateLoc(inst->getDebugLoc()), processValue(llvmOperands[0]), processValue(llvmOperands[1]));
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::SExt) {

    mapValue(inst) = odsBuilder.create<::mlir::LLVM::SExtOp>(
      translateLoc(inst->getDebugLoc()), convertType(inst->getType()), processValue(llvmOperands[0]));
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::SIToFP) {

    mapValue(inst) = odsBuilder.create<::mlir::LLVM::SIToFPOp>(
      translateLoc(inst->getDebugLoc()), convertType(inst->getType()), processValue(llvmOperands[0]));
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::SRem) {

    mapValue(inst) = odsBuilder.create<::mlir::LLVM::SRemOp>(translateLoc(inst->getDebugLoc()), processValue(llvmOperands[0]), processValue(llvmOperands[1]));
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::Select) {

    mapValue(inst) = odsBuilder.create<LLVM::SelectOp>(
      translateLoc(inst->getDebugLoc()), convertType(inst->getType()), processValue(llvmOperands[0]), processValue(llvmOperands[1]), processValue(llvmOperands[2]));
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::Shl) {

    mapValue(inst) = odsBuilder.create<::mlir::LLVM::ShlOp>(translateLoc(inst->getDebugLoc()), processValue(llvmOperands[0]), processValue(llvmOperands[1]));
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::ShuffleVector) {

    auto *svInst = cast<llvm::ShuffleVectorInst>(inst);
    SmallVector<int32_t> mask(svInst->getShuffleMask());
    mapValue(inst) = odsBuilder.create<LLVM::ShuffleVectorOp>(
      translateLoc(inst->getDebugLoc()), processValue(llvmOperands[0]), processValue(llvmOperands[1]), mask);
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::Store) {

    odsBuilder.create<LLVM::StoreOp>(translateLoc(inst->getDebugLoc()), processValue(llvmOperands[0]), processValue(llvmOperands[1]));
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::Sub) {

    mapValue(inst) = odsBuilder.create<::mlir::LLVM::SubOp>(translateLoc(inst->getDebugLoc()), processValue(llvmOperands[0]), processValue(llvmOperands[1]));
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::Trunc) {

    mapValue(inst) = odsBuilder.create<::mlir::LLVM::TruncOp>(
      translateLoc(inst->getDebugLoc()), convertType(inst->getType()), processValue(llvmOperands[0]));
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::UDiv) {

    mapValue(inst) = odsBuilder.create<::mlir::LLVM::UDivOp>(translateLoc(inst->getDebugLoc()), processValue(llvmOperands[0]), processValue(llvmOperands[1]));
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::UIToFP) {

    mapValue(inst) = odsBuilder.create<::mlir::LLVM::UIToFPOp>(
      translateLoc(inst->getDebugLoc()), convertType(inst->getType()), processValue(llvmOperands[0]));
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::URem) {

    mapValue(inst) = odsBuilder.create<::mlir::LLVM::URemOp>(translateLoc(inst->getDebugLoc()), processValue(llvmOperands[0]), processValue(llvmOperands[1]));
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::Unreachable) {

    odsBuilder.create<LLVM::UnreachableOp>(translateLoc(inst->getDebugLoc()));
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::Xor) {

    mapValue(inst) = odsBuilder.create<::mlir::LLVM::XOrOp>(translateLoc(inst->getDebugLoc()), processValue(llvmOperands[0]), processValue(llvmOperands[1]));
  
  return success();
}
if (inst->getOpcode() == llvm::Instruction::ZExt) {

    mapValue(inst) = odsBuilder.create<::mlir::LLVM::ZExtOp>(
      translateLoc(inst->getDebugLoc()), convertType(inst->getType()), processValue(llvmOperands[0]));
  
  return success();
}
