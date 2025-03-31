if (intrinsicID == llvm::Intrinsic::abs) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::AbsOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::bitreverse) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::BitReverseOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::copysign) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::CopySignOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::coro_align) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::CoroAlignOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::coro_begin) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::CoroBeginOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::coro_end) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::CoroEndOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::coro_free) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::CoroFreeOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::coro_id) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::CoroIdOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::coro_resume) {

    SmallVector<Type> resultTypes =
    {};
    Operation *op = odsBuilder.create<::mlir::LLVM::CoroResumeOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    (void)op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::coro_save) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::CoroSaveOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::coro_size) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::CoroSizeOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::coro_suspend) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::CoroSuspendOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::cos) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::CosOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::ctlz) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::CountLeadingZerosOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::cttz) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::CountTrailingZerosOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::ctpop) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::CtPopOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::eh_typeid_for) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::EhTypeidForOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::exp2) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::Exp2Op>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::exp) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::ExpOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::fabs) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::FAbsOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::ceil) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::FCeilOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::floor) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::FFloorOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::fma) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::FMAOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::fmuladd) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::FMulAddOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::trunc) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::FTruncOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::get_active_lane_mask) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::GetActiveLaneMaskOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::lifetime_end) {

    odsBuilder.create<LLVM::LifetimeEndOp>(
      translateLoc(inst->getDebugLoc()), matchIntegerAttr(processValue(llvmOperands[0])), processValue(llvmOperands[1]));
  
  return success();
}
if (intrinsicID == llvm::Intrinsic::lifetime_start) {

    odsBuilder.create<LLVM::LifetimeStartOp>(
      translateLoc(inst->getDebugLoc()), matchIntegerAttr(processValue(llvmOperands[0])), processValue(llvmOperands[1]));
  
  return success();
}
if (intrinsicID == llvm::Intrinsic::log10) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::Log10Op>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::log2) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::Log2Op>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::log) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::LogOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::maxnum) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::MaxNumOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::maximum) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::MaximumOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::memcpy_inline) {

    SmallVector<Type> resultTypes =
    {};
    Operation *op = odsBuilder.create<::mlir::LLVM::MemcpyInlineOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    (void)op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::memcpy) {

    SmallVector<Type> resultTypes =
    {};
    Operation *op = odsBuilder.create<::mlir::LLVM::MemcpyOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    (void)op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::memmove) {

    SmallVector<Type> resultTypes =
    {};
    Operation *op = odsBuilder.create<::mlir::LLVM::MemmoveOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    (void)op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::memset) {

    SmallVector<Type> resultTypes =
    {};
    Operation *op = odsBuilder.create<::mlir::LLVM::MemsetOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    (void)op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::minnum) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::MinNumOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::minimum) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::MinimumOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::powi) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::PowIOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::pow) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::PowOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::prefetch) {

    SmallVector<Type> resultTypes =
    {};
    Operation *op = odsBuilder.create<::mlir::LLVM::Prefetch>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    (void)op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::roundeven) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::RoundEvenOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::round) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::RoundOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::sadd_with_overflow) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::SAddWithOverflowOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::smax) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::SMaxOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::smin) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::SMinOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::smul_with_overflow) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::SMulWithOverflowOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::ssub_with_overflow) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::SSubWithOverflowOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::sin) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::SinOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::sqrt) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::SqrtOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::stackrestore) {

    SmallVector<Type> resultTypes =
    {};
    Operation *op = odsBuilder.create<::mlir::LLVM::StackRestoreOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    (void)op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::stacksave) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::StackSaveOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::experimental_stepvector) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::StepVectorOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::uadd_with_overflow) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::UAddWithOverflowOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::umax) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::UMaxOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::umin) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::UMinOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::umul_with_overflow) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::UMulWithOverflowOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::usub_with_overflow) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::USubWithOverflowOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_ashr) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::VPAShrOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_add) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::VPAddOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_and) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::VPAndOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_fadd) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::VPFAddOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_fdiv) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::VPFDivOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_fmul) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::VPFMulOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_fneg) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::VPFNegOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_fpext) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::VPFPExtOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_fptosi) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::VPFPToSIOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_fptoui) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::VPFPToUIOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_fptrunc) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::VPFPTruncOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_frem) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::VPFRemOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_fsub) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::VPFSubOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_fma) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::VPFmaOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_inttoptr) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::VPIntToPtrOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_lshr) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::VPLShrOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_load) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::VPLoadOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_merge) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::VPMergeMinOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_mul) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::VPMulOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_or) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::VPOrOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_ptrtoint) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::VPPtrToIntOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_reduce_add) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::VPReduceAddOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_reduce_and) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::VPReduceAndOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_reduce_fadd) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::VPReduceFAddOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_reduce_fmax) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::VPReduceFMaxOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_reduce_fmin) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::VPReduceFMinOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_reduce_fmul) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::VPReduceFMulOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_reduce_mul) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::VPReduceMulOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_reduce_or) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::VPReduceOrOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_reduce_smax) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::VPReduceSMaxOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_reduce_smin) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::VPReduceSMinOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_reduce_umax) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::VPReduceUMaxOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_reduce_umin) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::VPReduceUMinOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_reduce_xor) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::VPReduceXorOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_sdiv) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::VPSDivOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_sext) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::VPSExtOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_sitofp) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::VPSIToFPOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_srem) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::VPSRemOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_select) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::VPSelectMinOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_shl) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::VPShlOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_store) {

    SmallVector<Type> resultTypes =
    {};
    Operation *op = odsBuilder.create<::mlir::LLVM::VPStoreOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    (void)op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::experimental_vp_strided_load) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::VPStridedLoadOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::experimental_vp_strided_store) {

    SmallVector<Type> resultTypes =
    {};
    Operation *op = odsBuilder.create<::mlir::LLVM::VPStridedStoreOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    (void)op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_sub) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::VPSubOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_trunc) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::VPTruncOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_udiv) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::VPUDivOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_uitofp) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::VPUIToFPOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_urem) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::VPURemOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_xor) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::VPXorOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::vp_zext) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::VPZExtOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::vacopy) {

    SmallVector<Type> resultTypes =
    {};
    Operation *op = odsBuilder.create<::mlir::LLVM::VaCopyOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    (void)op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vaend) {

    SmallVector<Type> resultTypes =
    {};
    Operation *op = odsBuilder.create<::mlir::LLVM::VaEndOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    (void)op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::vastart) {

    SmallVector<Type> resultTypes =
    {};
    Operation *op = odsBuilder.create<::mlir::LLVM::VaStartOp>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    (void)op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::masked_compressstore) {

    SmallVector<Type> resultTypes =
    {};
    Operation *op = odsBuilder.create<::mlir::LLVM::masked_compressstore>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    (void)op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::masked_expandload) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::masked_expandload>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::vector_reduce_add) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::vector_reduce_add>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::vector_reduce_and) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::vector_reduce_and>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::vector_reduce_fmax) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::vector_reduce_fmax>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::vector_reduce_fmin) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::vector_reduce_fmin>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::vector_reduce_mul) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::vector_reduce_mul>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::vector_reduce_or) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::vector_reduce_or>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::vector_reduce_smax) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::vector_reduce_smax>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::vector_reduce_smin) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::vector_reduce_smin>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::vector_reduce_umax) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::vector_reduce_umax>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::vector_reduce_umin) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::vector_reduce_umin>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::vector_reduce_xor) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::vector_reduce_xor>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
if (intrinsicID == llvm::Intrinsic::vscale) {

    SmallVector<Type> resultTypes =
    {convertType(inst->getType())};
    Operation *op = odsBuilder.create<::mlir::LLVM::vscale>(
      translateLoc(inst->getDebugLoc()),
      resultTypes,
      processValues(llvmOperands));
    mapValue(inst) = op->getResult(0);
  return success();
}
