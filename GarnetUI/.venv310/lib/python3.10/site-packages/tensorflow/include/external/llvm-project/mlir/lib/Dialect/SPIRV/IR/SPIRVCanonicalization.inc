/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Rewriters                                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/SPIRV/IR/SPIRVCanonicalization.td:20
*/
struct ConvertChainedBitcast : public ::mlir::RewritePattern {
  ConvertChainedBitcast(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("spirv.Bitcast", 2, context, {"spirv.Bitcast"}) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range operand(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::spirv::BitcastOp>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::spirv::BitcastOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::spirv::BitcastOp type";
        });
      }
      operand = castedOp1.getODSOperands(0);
      tblgen_ops.push_back(op1);
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::spirv::BitcastOp tblgen_BitcastOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*operand.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_BitcastOp_0 = rewriter.create<::mlir::spirv::BitcastOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_BitcastOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/SPIRV/IR/SPIRVCanonicalization.td:58
*/
struct ConvertComparisonIntoClamp1_SPIRV_FOrdLessThanEqualOp : public ::mlir::RewritePattern {
  ConvertComparisonIntoClamp1_SPIRV_FOrdLessThanEqualOp(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("spirv.Select", 4, context, {"spirv.GL.FClamp"}) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range max(op0->getOperands());
    ::mlir::Operation::operand_range max0(op0->getOperands());
    ::mlir::Operation::operand_range input(op0->getOperands());
    ::mlir::Operation::operand_range input0(op0->getOperands());
    ::mlir::Operation::operand_range middle1(op0->getOperands());
    ::mlir::Operation::operand_range min(op0->getOperands());
    ::mlir::Operation::operand_range min0(op0->getOperands());
    ::mlir::spirv::SelectOp middle0;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::spirv::SelectOp>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::spirv::FOrdLessThanEqualOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::spirv::FOrdLessThanEqualOp type";
        });
      }
      {
        auto *op2 = (*castedOp1.getODSOperands(0).begin()).getDefiningOp();
        if (!(op2)){
          return rewriter.notifyMatchFailure(castedOp1, [&](::mlir::Diagnostic &diag) {
            diag << "There's no operation that defines operand 0 of castedOp1";
          });
        }
        auto castedOp2 = ::llvm::dyn_cast<::mlir::spirv::SelectOp>(op2); (void)castedOp2;
        if (!(castedOp2)){
          return rewriter.notifyMatchFailure(op2, [&](::mlir::Diagnostic &diag) {
            diag << "castedOp2 is not ::mlir::spirv::SelectOp type";
          });
        }
        middle0 = castedOp2;
        {
          auto *op3 = (*castedOp2.getODSOperands(0).begin()).getDefiningOp();
          if (!(op3)){
            return rewriter.notifyMatchFailure(castedOp2, [&](::mlir::Diagnostic &diag) {
              diag << "There's no operation that defines operand 0 of castedOp2";
            });
          }
          auto castedOp3 = ::llvm::dyn_cast<::mlir::spirv::FOrdLessThanEqualOp>(op3); (void)castedOp3;
          if (!(castedOp3)){
            return rewriter.notifyMatchFailure(op3, [&](::mlir::Diagnostic &diag) {
              diag << "castedOp3 is not ::mlir::spirv::FOrdLessThanEqualOp type";
            });
          }
          min0 = castedOp3.getODSOperands(0);
          input0 = castedOp3.getODSOperands(1);
          tblgen_ops.push_back(op3);
        }
        input = castedOp2.getODSOperands(1);
        min = castedOp2.getODSOperands(2);
        tblgen_ops.push_back(op2);
      }
      max0 = castedOp1.getODSOperands(1);
      tblgen_ops.push_back(op1);
    }
    middle1 = castedOp0.getODSOperands(1);
    max = castedOp0.getODSOperands(2);
    if (!(((*middle0.getODSResults(0).begin()) == (*middle1.begin())))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'middle0, middle1' failed to satisfy constraint: ''";
      });
    }
    if (!(*max.begin() == *max0.begin())){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "Operands 'max' and 'max0' must be equal";
      });
    }
    if (!(*input.begin() == *input0.begin())){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "Operands 'input' and 'input0' must be equal";
      });
    }
    if (!(*min.begin() == *min0.begin())){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "Operands 'min' and 'min0' must be equal";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc(), tblgen_ops[2]->getLoc(), tblgen_ops[3]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::spirv::GLFClampOp tblgen_GLFClampOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*input.begin()));
      tblgen_values.push_back((*min.begin()));
      tblgen_values.push_back((*max.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_GLFClampOp_0 = rewriter.create<::mlir::spirv::GLFClampOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_GLFClampOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/SPIRV/IR/SPIRVCanonicalization.td:58
*/
struct ConvertComparisonIntoClamp1_SPIRV_FOrdLessThanOp : public ::mlir::RewritePattern {
  ConvertComparisonIntoClamp1_SPIRV_FOrdLessThanOp(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("spirv.Select", 4, context, {"spirv.GL.FClamp"}) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range max(op0->getOperands());
    ::mlir::Operation::operand_range max0(op0->getOperands());
    ::mlir::Operation::operand_range input(op0->getOperands());
    ::mlir::Operation::operand_range input0(op0->getOperands());
    ::mlir::Operation::operand_range middle1(op0->getOperands());
    ::mlir::Operation::operand_range min(op0->getOperands());
    ::mlir::Operation::operand_range min0(op0->getOperands());
    ::mlir::spirv::SelectOp middle0;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::spirv::SelectOp>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::spirv::FOrdLessThanOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::spirv::FOrdLessThanOp type";
        });
      }
      {
        auto *op2 = (*castedOp1.getODSOperands(0).begin()).getDefiningOp();
        if (!(op2)){
          return rewriter.notifyMatchFailure(castedOp1, [&](::mlir::Diagnostic &diag) {
            diag << "There's no operation that defines operand 0 of castedOp1";
          });
        }
        auto castedOp2 = ::llvm::dyn_cast<::mlir::spirv::SelectOp>(op2); (void)castedOp2;
        if (!(castedOp2)){
          return rewriter.notifyMatchFailure(op2, [&](::mlir::Diagnostic &diag) {
            diag << "castedOp2 is not ::mlir::spirv::SelectOp type";
          });
        }
        middle0 = castedOp2;
        {
          auto *op3 = (*castedOp2.getODSOperands(0).begin()).getDefiningOp();
          if (!(op3)){
            return rewriter.notifyMatchFailure(castedOp2, [&](::mlir::Diagnostic &diag) {
              diag << "There's no operation that defines operand 0 of castedOp2";
            });
          }
          auto castedOp3 = ::llvm::dyn_cast<::mlir::spirv::FOrdLessThanOp>(op3); (void)castedOp3;
          if (!(castedOp3)){
            return rewriter.notifyMatchFailure(op3, [&](::mlir::Diagnostic &diag) {
              diag << "castedOp3 is not ::mlir::spirv::FOrdLessThanOp type";
            });
          }
          min0 = castedOp3.getODSOperands(0);
          input0 = castedOp3.getODSOperands(1);
          tblgen_ops.push_back(op3);
        }
        input = castedOp2.getODSOperands(1);
        min = castedOp2.getODSOperands(2);
        tblgen_ops.push_back(op2);
      }
      max0 = castedOp1.getODSOperands(1);
      tblgen_ops.push_back(op1);
    }
    middle1 = castedOp0.getODSOperands(1);
    max = castedOp0.getODSOperands(2);
    if (!(((*middle0.getODSResults(0).begin()) == (*middle1.begin())))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'middle0, middle1' failed to satisfy constraint: ''";
      });
    }
    if (!(*max.begin() == *max0.begin())){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "Operands 'max' and 'max0' must be equal";
      });
    }
    if (!(*input.begin() == *input0.begin())){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "Operands 'input' and 'input0' must be equal";
      });
    }
    if (!(*min.begin() == *min0.begin())){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "Operands 'min' and 'min0' must be equal";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc(), tblgen_ops[2]->getLoc(), tblgen_ops[3]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::spirv::GLFClampOp tblgen_GLFClampOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*input.begin()));
      tblgen_values.push_back((*min.begin()));
      tblgen_values.push_back((*max.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_GLFClampOp_0 = rewriter.create<::mlir::spirv::GLFClampOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_GLFClampOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/SPIRV/IR/SPIRVCanonicalization.td:58
*/
struct ConvertComparisonIntoClamp1_SPIRV_SLessThanEqualOp : public ::mlir::RewritePattern {
  ConvertComparisonIntoClamp1_SPIRV_SLessThanEqualOp(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("spirv.Select", 4, context, {"spirv.GL.SClamp"}) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range max(op0->getOperands());
    ::mlir::Operation::operand_range max0(op0->getOperands());
    ::mlir::Operation::operand_range input(op0->getOperands());
    ::mlir::Operation::operand_range input0(op0->getOperands());
    ::mlir::Operation::operand_range middle1(op0->getOperands());
    ::mlir::Operation::operand_range min(op0->getOperands());
    ::mlir::Operation::operand_range min0(op0->getOperands());
    ::mlir::spirv::SelectOp middle0;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::spirv::SelectOp>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::spirv::SLessThanEqualOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::spirv::SLessThanEqualOp type";
        });
      }
      {
        auto *op2 = (*castedOp1.getODSOperands(0).begin()).getDefiningOp();
        if (!(op2)){
          return rewriter.notifyMatchFailure(castedOp1, [&](::mlir::Diagnostic &diag) {
            diag << "There's no operation that defines operand 0 of castedOp1";
          });
        }
        auto castedOp2 = ::llvm::dyn_cast<::mlir::spirv::SelectOp>(op2); (void)castedOp2;
        if (!(castedOp2)){
          return rewriter.notifyMatchFailure(op2, [&](::mlir::Diagnostic &diag) {
            diag << "castedOp2 is not ::mlir::spirv::SelectOp type";
          });
        }
        middle0 = castedOp2;
        {
          auto *op3 = (*castedOp2.getODSOperands(0).begin()).getDefiningOp();
          if (!(op3)){
            return rewriter.notifyMatchFailure(castedOp2, [&](::mlir::Diagnostic &diag) {
              diag << "There's no operation that defines operand 0 of castedOp2";
            });
          }
          auto castedOp3 = ::llvm::dyn_cast<::mlir::spirv::SLessThanEqualOp>(op3); (void)castedOp3;
          if (!(castedOp3)){
            return rewriter.notifyMatchFailure(op3, [&](::mlir::Diagnostic &diag) {
              diag << "castedOp3 is not ::mlir::spirv::SLessThanEqualOp type";
            });
          }
          min0 = castedOp3.getODSOperands(0);
          input0 = castedOp3.getODSOperands(1);
          tblgen_ops.push_back(op3);
        }
        input = castedOp2.getODSOperands(1);
        min = castedOp2.getODSOperands(2);
        tblgen_ops.push_back(op2);
      }
      max0 = castedOp1.getODSOperands(1);
      tblgen_ops.push_back(op1);
    }
    middle1 = castedOp0.getODSOperands(1);
    max = castedOp0.getODSOperands(2);
    if (!(((*middle0.getODSResults(0).begin()) == (*middle1.begin())))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'middle0, middle1' failed to satisfy constraint: ''";
      });
    }
    if (!(*max.begin() == *max0.begin())){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "Operands 'max' and 'max0' must be equal";
      });
    }
    if (!(*input.begin() == *input0.begin())){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "Operands 'input' and 'input0' must be equal";
      });
    }
    if (!(*min.begin() == *min0.begin())){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "Operands 'min' and 'min0' must be equal";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc(), tblgen_ops[2]->getLoc(), tblgen_ops[3]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::spirv::GLSClampOp tblgen_GLSClampOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*input.begin()));
      tblgen_values.push_back((*min.begin()));
      tblgen_values.push_back((*max.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_GLSClampOp_0 = rewriter.create<::mlir::spirv::GLSClampOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_GLSClampOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/SPIRV/IR/SPIRVCanonicalization.td:58
*/
struct ConvertComparisonIntoClamp1_SPIRV_SLessThanOp : public ::mlir::RewritePattern {
  ConvertComparisonIntoClamp1_SPIRV_SLessThanOp(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("spirv.Select", 4, context, {"spirv.GL.SClamp"}) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range max(op0->getOperands());
    ::mlir::Operation::operand_range max0(op0->getOperands());
    ::mlir::Operation::operand_range input(op0->getOperands());
    ::mlir::Operation::operand_range input0(op0->getOperands());
    ::mlir::Operation::operand_range middle1(op0->getOperands());
    ::mlir::Operation::operand_range min(op0->getOperands());
    ::mlir::Operation::operand_range min0(op0->getOperands());
    ::mlir::spirv::SelectOp middle0;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::spirv::SelectOp>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::spirv::SLessThanOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::spirv::SLessThanOp type";
        });
      }
      {
        auto *op2 = (*castedOp1.getODSOperands(0).begin()).getDefiningOp();
        if (!(op2)){
          return rewriter.notifyMatchFailure(castedOp1, [&](::mlir::Diagnostic &diag) {
            diag << "There's no operation that defines operand 0 of castedOp1";
          });
        }
        auto castedOp2 = ::llvm::dyn_cast<::mlir::spirv::SelectOp>(op2); (void)castedOp2;
        if (!(castedOp2)){
          return rewriter.notifyMatchFailure(op2, [&](::mlir::Diagnostic &diag) {
            diag << "castedOp2 is not ::mlir::spirv::SelectOp type";
          });
        }
        middle0 = castedOp2;
        {
          auto *op3 = (*castedOp2.getODSOperands(0).begin()).getDefiningOp();
          if (!(op3)){
            return rewriter.notifyMatchFailure(castedOp2, [&](::mlir::Diagnostic &diag) {
              diag << "There's no operation that defines operand 0 of castedOp2";
            });
          }
          auto castedOp3 = ::llvm::dyn_cast<::mlir::spirv::SLessThanOp>(op3); (void)castedOp3;
          if (!(castedOp3)){
            return rewriter.notifyMatchFailure(op3, [&](::mlir::Diagnostic &diag) {
              diag << "castedOp3 is not ::mlir::spirv::SLessThanOp type";
            });
          }
          min0 = castedOp3.getODSOperands(0);
          input0 = castedOp3.getODSOperands(1);
          tblgen_ops.push_back(op3);
        }
        input = castedOp2.getODSOperands(1);
        min = castedOp2.getODSOperands(2);
        tblgen_ops.push_back(op2);
      }
      max0 = castedOp1.getODSOperands(1);
      tblgen_ops.push_back(op1);
    }
    middle1 = castedOp0.getODSOperands(1);
    max = castedOp0.getODSOperands(2);
    if (!(((*middle0.getODSResults(0).begin()) == (*middle1.begin())))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'middle0, middle1' failed to satisfy constraint: ''";
      });
    }
    if (!(*max.begin() == *max0.begin())){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "Operands 'max' and 'max0' must be equal";
      });
    }
    if (!(*input.begin() == *input0.begin())){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "Operands 'input' and 'input0' must be equal";
      });
    }
    if (!(*min.begin() == *min0.begin())){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "Operands 'min' and 'min0' must be equal";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc(), tblgen_ops[2]->getLoc(), tblgen_ops[3]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::spirv::GLSClampOp tblgen_GLSClampOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*input.begin()));
      tblgen_values.push_back((*min.begin()));
      tblgen_values.push_back((*max.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_GLSClampOp_0 = rewriter.create<::mlir::spirv::GLSClampOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_GLSClampOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/SPIRV/IR/SPIRVCanonicalization.td:58
*/
struct ConvertComparisonIntoClamp1_SPIRV_ULessThanEqualOp : public ::mlir::RewritePattern {
  ConvertComparisonIntoClamp1_SPIRV_ULessThanEqualOp(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("spirv.Select", 4, context, {"spirv.GL.UClamp"}) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range max(op0->getOperands());
    ::mlir::Operation::operand_range max0(op0->getOperands());
    ::mlir::Operation::operand_range input(op0->getOperands());
    ::mlir::Operation::operand_range input0(op0->getOperands());
    ::mlir::Operation::operand_range middle1(op0->getOperands());
    ::mlir::Operation::operand_range min(op0->getOperands());
    ::mlir::Operation::operand_range min0(op0->getOperands());
    ::mlir::spirv::SelectOp middle0;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::spirv::SelectOp>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::spirv::ULessThanEqualOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::spirv::ULessThanEqualOp type";
        });
      }
      {
        auto *op2 = (*castedOp1.getODSOperands(0).begin()).getDefiningOp();
        if (!(op2)){
          return rewriter.notifyMatchFailure(castedOp1, [&](::mlir::Diagnostic &diag) {
            diag << "There's no operation that defines operand 0 of castedOp1";
          });
        }
        auto castedOp2 = ::llvm::dyn_cast<::mlir::spirv::SelectOp>(op2); (void)castedOp2;
        if (!(castedOp2)){
          return rewriter.notifyMatchFailure(op2, [&](::mlir::Diagnostic &diag) {
            diag << "castedOp2 is not ::mlir::spirv::SelectOp type";
          });
        }
        middle0 = castedOp2;
        {
          auto *op3 = (*castedOp2.getODSOperands(0).begin()).getDefiningOp();
          if (!(op3)){
            return rewriter.notifyMatchFailure(castedOp2, [&](::mlir::Diagnostic &diag) {
              diag << "There's no operation that defines operand 0 of castedOp2";
            });
          }
          auto castedOp3 = ::llvm::dyn_cast<::mlir::spirv::ULessThanEqualOp>(op3); (void)castedOp3;
          if (!(castedOp3)){
            return rewriter.notifyMatchFailure(op3, [&](::mlir::Diagnostic &diag) {
              diag << "castedOp3 is not ::mlir::spirv::ULessThanEqualOp type";
            });
          }
          min0 = castedOp3.getODSOperands(0);
          input0 = castedOp3.getODSOperands(1);
          tblgen_ops.push_back(op3);
        }
        input = castedOp2.getODSOperands(1);
        min = castedOp2.getODSOperands(2);
        tblgen_ops.push_back(op2);
      }
      max0 = castedOp1.getODSOperands(1);
      tblgen_ops.push_back(op1);
    }
    middle1 = castedOp0.getODSOperands(1);
    max = castedOp0.getODSOperands(2);
    if (!(((*middle0.getODSResults(0).begin()) == (*middle1.begin())))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'middle0, middle1' failed to satisfy constraint: ''";
      });
    }
    if (!(*max.begin() == *max0.begin())){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "Operands 'max' and 'max0' must be equal";
      });
    }
    if (!(*input.begin() == *input0.begin())){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "Operands 'input' and 'input0' must be equal";
      });
    }
    if (!(*min.begin() == *min0.begin())){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "Operands 'min' and 'min0' must be equal";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc(), tblgen_ops[2]->getLoc(), tblgen_ops[3]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::spirv::GLUClampOp tblgen_GLUClampOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*input.begin()));
      tblgen_values.push_back((*min.begin()));
      tblgen_values.push_back((*max.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_GLUClampOp_0 = rewriter.create<::mlir::spirv::GLUClampOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_GLUClampOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/SPIRV/IR/SPIRVCanonicalization.td:58
*/
struct ConvertComparisonIntoClamp1_SPIRV_ULessThanOp : public ::mlir::RewritePattern {
  ConvertComparisonIntoClamp1_SPIRV_ULessThanOp(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("spirv.Select", 4, context, {"spirv.GL.UClamp"}) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range max(op0->getOperands());
    ::mlir::Operation::operand_range max0(op0->getOperands());
    ::mlir::Operation::operand_range input(op0->getOperands());
    ::mlir::Operation::operand_range input0(op0->getOperands());
    ::mlir::Operation::operand_range middle1(op0->getOperands());
    ::mlir::Operation::operand_range min(op0->getOperands());
    ::mlir::Operation::operand_range min0(op0->getOperands());
    ::mlir::spirv::SelectOp middle0;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::spirv::SelectOp>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::spirv::ULessThanOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::spirv::ULessThanOp type";
        });
      }
      {
        auto *op2 = (*castedOp1.getODSOperands(0).begin()).getDefiningOp();
        if (!(op2)){
          return rewriter.notifyMatchFailure(castedOp1, [&](::mlir::Diagnostic &diag) {
            diag << "There's no operation that defines operand 0 of castedOp1";
          });
        }
        auto castedOp2 = ::llvm::dyn_cast<::mlir::spirv::SelectOp>(op2); (void)castedOp2;
        if (!(castedOp2)){
          return rewriter.notifyMatchFailure(op2, [&](::mlir::Diagnostic &diag) {
            diag << "castedOp2 is not ::mlir::spirv::SelectOp type";
          });
        }
        middle0 = castedOp2;
        {
          auto *op3 = (*castedOp2.getODSOperands(0).begin()).getDefiningOp();
          if (!(op3)){
            return rewriter.notifyMatchFailure(castedOp2, [&](::mlir::Diagnostic &diag) {
              diag << "There's no operation that defines operand 0 of castedOp2";
            });
          }
          auto castedOp3 = ::llvm::dyn_cast<::mlir::spirv::ULessThanOp>(op3); (void)castedOp3;
          if (!(castedOp3)){
            return rewriter.notifyMatchFailure(op3, [&](::mlir::Diagnostic &diag) {
              diag << "castedOp3 is not ::mlir::spirv::ULessThanOp type";
            });
          }
          min0 = castedOp3.getODSOperands(0);
          input0 = castedOp3.getODSOperands(1);
          tblgen_ops.push_back(op3);
        }
        input = castedOp2.getODSOperands(1);
        min = castedOp2.getODSOperands(2);
        tblgen_ops.push_back(op2);
      }
      max0 = castedOp1.getODSOperands(1);
      tblgen_ops.push_back(op1);
    }
    middle1 = castedOp0.getODSOperands(1);
    max = castedOp0.getODSOperands(2);
    if (!(((*middle0.getODSResults(0).begin()) == (*middle1.begin())))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'middle0, middle1' failed to satisfy constraint: ''";
      });
    }
    if (!(*max.begin() == *max0.begin())){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "Operands 'max' and 'max0' must be equal";
      });
    }
    if (!(*input.begin() == *input0.begin())){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "Operands 'input' and 'input0' must be equal";
      });
    }
    if (!(*min.begin() == *min0.begin())){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "Operands 'min' and 'min0' must be equal";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc(), tblgen_ops[2]->getLoc(), tblgen_ops[3]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::spirv::GLUClampOp tblgen_GLUClampOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*input.begin()));
      tblgen_values.push_back((*min.begin()));
      tblgen_values.push_back((*max.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_GLUClampOp_0 = rewriter.create<::mlir::spirv::GLUClampOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_GLUClampOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/SPIRV/IR/SPIRVCanonicalization.td:74
*/
struct ConvertComparisonIntoClamp2_SPIRV_FOrdLessThanEqualOp : public ::mlir::RewritePattern {
  ConvertComparisonIntoClamp2_SPIRV_FOrdLessThanEqualOp(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("spirv.Select", 4, context, {"spirv.GL.FClamp"}) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range min(op0->getOperands());
    ::mlir::Operation::operand_range min0(op0->getOperands());
    ::mlir::Operation::operand_range input(op0->getOperands());
    ::mlir::Operation::operand_range input0(op0->getOperands());
    ::mlir::Operation::operand_range input1(op0->getOperands());
    ::mlir::Operation::operand_range max(op0->getOperands());
    ::mlir::Operation::operand_range max0(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::spirv::SelectOp>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::spirv::FOrdLessThanEqualOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::spirv::FOrdLessThanEqualOp type";
        });
      }
      max0 = castedOp1.getODSOperands(0);
      input1 = castedOp1.getODSOperands(1);
      tblgen_ops.push_back(op1);
    }
    max = castedOp0.getODSOperands(1);
    {
      auto *op1 = (*castedOp0.getODSOperands(2).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 2 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::spirv::SelectOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::spirv::SelectOp type";
        });
      }
      {
        auto *op2 = (*castedOp1.getODSOperands(0).begin()).getDefiningOp();
        if (!(op2)){
          return rewriter.notifyMatchFailure(castedOp1, [&](::mlir::Diagnostic &diag) {
            diag << "There's no operation that defines operand 0 of castedOp1";
          });
        }
        auto castedOp2 = ::llvm::dyn_cast<::mlir::spirv::FOrdLessThanEqualOp>(op2); (void)castedOp2;
        if (!(castedOp2)){
          return rewriter.notifyMatchFailure(op2, [&](::mlir::Diagnostic &diag) {
            diag << "castedOp2 is not ::mlir::spirv::FOrdLessThanEqualOp type";
          });
        }
        input0 = castedOp2.getODSOperands(0);
        min0 = castedOp2.getODSOperands(1);
        tblgen_ops.push_back(op2);
      }
      min = castedOp1.getODSOperands(1);
      input = castedOp1.getODSOperands(2);
      tblgen_ops.push_back(op1);
    }
    if (!(*min.begin() == *min0.begin())){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "Operands 'min' and 'min0' must be equal";
      });
    }
    if (!(*input.begin() == *input0.begin())){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "Operands 'input' and 'input0' must be equal";
      });
    }
    if (!(*input.begin() == *input1.begin())){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "Operands 'input' and 'input1' must be equal";
      });
    }
    if (!(*max.begin() == *max0.begin())){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "Operands 'max' and 'max0' must be equal";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc(), tblgen_ops[2]->getLoc(), tblgen_ops[3]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::spirv::GLFClampOp tblgen_GLFClampOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*input.begin()));
      tblgen_values.push_back((*min.begin()));
      tblgen_values.push_back((*max.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_GLFClampOp_0 = rewriter.create<::mlir::spirv::GLFClampOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_GLFClampOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/SPIRV/IR/SPIRVCanonicalization.td:74
*/
struct ConvertComparisonIntoClamp2_SPIRV_FOrdLessThanOp : public ::mlir::RewritePattern {
  ConvertComparisonIntoClamp2_SPIRV_FOrdLessThanOp(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("spirv.Select", 4, context, {"spirv.GL.FClamp"}) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range min(op0->getOperands());
    ::mlir::Operation::operand_range min0(op0->getOperands());
    ::mlir::Operation::operand_range input(op0->getOperands());
    ::mlir::Operation::operand_range input0(op0->getOperands());
    ::mlir::Operation::operand_range input1(op0->getOperands());
    ::mlir::Operation::operand_range max(op0->getOperands());
    ::mlir::Operation::operand_range max0(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::spirv::SelectOp>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::spirv::FOrdLessThanOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::spirv::FOrdLessThanOp type";
        });
      }
      max0 = castedOp1.getODSOperands(0);
      input1 = castedOp1.getODSOperands(1);
      tblgen_ops.push_back(op1);
    }
    max = castedOp0.getODSOperands(1);
    {
      auto *op1 = (*castedOp0.getODSOperands(2).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 2 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::spirv::SelectOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::spirv::SelectOp type";
        });
      }
      {
        auto *op2 = (*castedOp1.getODSOperands(0).begin()).getDefiningOp();
        if (!(op2)){
          return rewriter.notifyMatchFailure(castedOp1, [&](::mlir::Diagnostic &diag) {
            diag << "There's no operation that defines operand 0 of castedOp1";
          });
        }
        auto castedOp2 = ::llvm::dyn_cast<::mlir::spirv::FOrdLessThanOp>(op2); (void)castedOp2;
        if (!(castedOp2)){
          return rewriter.notifyMatchFailure(op2, [&](::mlir::Diagnostic &diag) {
            diag << "castedOp2 is not ::mlir::spirv::FOrdLessThanOp type";
          });
        }
        input0 = castedOp2.getODSOperands(0);
        min0 = castedOp2.getODSOperands(1);
        tblgen_ops.push_back(op2);
      }
      min = castedOp1.getODSOperands(1);
      input = castedOp1.getODSOperands(2);
      tblgen_ops.push_back(op1);
    }
    if (!(*min.begin() == *min0.begin())){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "Operands 'min' and 'min0' must be equal";
      });
    }
    if (!(*input.begin() == *input0.begin())){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "Operands 'input' and 'input0' must be equal";
      });
    }
    if (!(*input.begin() == *input1.begin())){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "Operands 'input' and 'input1' must be equal";
      });
    }
    if (!(*max.begin() == *max0.begin())){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "Operands 'max' and 'max0' must be equal";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc(), tblgen_ops[2]->getLoc(), tblgen_ops[3]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::spirv::GLFClampOp tblgen_GLFClampOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*input.begin()));
      tblgen_values.push_back((*min.begin()));
      tblgen_values.push_back((*max.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_GLFClampOp_0 = rewriter.create<::mlir::spirv::GLFClampOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_GLFClampOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/SPIRV/IR/SPIRVCanonicalization.td:74
*/
struct ConvertComparisonIntoClamp2_SPIRV_SLessThanEqualOp : public ::mlir::RewritePattern {
  ConvertComparisonIntoClamp2_SPIRV_SLessThanEqualOp(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("spirv.Select", 4, context, {"spirv.GL.SClamp"}) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range min(op0->getOperands());
    ::mlir::Operation::operand_range min0(op0->getOperands());
    ::mlir::Operation::operand_range input(op0->getOperands());
    ::mlir::Operation::operand_range input0(op0->getOperands());
    ::mlir::Operation::operand_range input1(op0->getOperands());
    ::mlir::Operation::operand_range max(op0->getOperands());
    ::mlir::Operation::operand_range max0(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::spirv::SelectOp>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::spirv::SLessThanEqualOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::spirv::SLessThanEqualOp type";
        });
      }
      max0 = castedOp1.getODSOperands(0);
      input1 = castedOp1.getODSOperands(1);
      tblgen_ops.push_back(op1);
    }
    max = castedOp0.getODSOperands(1);
    {
      auto *op1 = (*castedOp0.getODSOperands(2).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 2 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::spirv::SelectOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::spirv::SelectOp type";
        });
      }
      {
        auto *op2 = (*castedOp1.getODSOperands(0).begin()).getDefiningOp();
        if (!(op2)){
          return rewriter.notifyMatchFailure(castedOp1, [&](::mlir::Diagnostic &diag) {
            diag << "There's no operation that defines operand 0 of castedOp1";
          });
        }
        auto castedOp2 = ::llvm::dyn_cast<::mlir::spirv::SLessThanEqualOp>(op2); (void)castedOp2;
        if (!(castedOp2)){
          return rewriter.notifyMatchFailure(op2, [&](::mlir::Diagnostic &diag) {
            diag << "castedOp2 is not ::mlir::spirv::SLessThanEqualOp type";
          });
        }
        input0 = castedOp2.getODSOperands(0);
        min0 = castedOp2.getODSOperands(1);
        tblgen_ops.push_back(op2);
      }
      min = castedOp1.getODSOperands(1);
      input = castedOp1.getODSOperands(2);
      tblgen_ops.push_back(op1);
    }
    if (!(*min.begin() == *min0.begin())){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "Operands 'min' and 'min0' must be equal";
      });
    }
    if (!(*input.begin() == *input0.begin())){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "Operands 'input' and 'input0' must be equal";
      });
    }
    if (!(*input.begin() == *input1.begin())){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "Operands 'input' and 'input1' must be equal";
      });
    }
    if (!(*max.begin() == *max0.begin())){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "Operands 'max' and 'max0' must be equal";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc(), tblgen_ops[2]->getLoc(), tblgen_ops[3]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::spirv::GLSClampOp tblgen_GLSClampOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*input.begin()));
      tblgen_values.push_back((*min.begin()));
      tblgen_values.push_back((*max.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_GLSClampOp_0 = rewriter.create<::mlir::spirv::GLSClampOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_GLSClampOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/SPIRV/IR/SPIRVCanonicalization.td:74
*/
struct ConvertComparisonIntoClamp2_SPIRV_SLessThanOp : public ::mlir::RewritePattern {
  ConvertComparisonIntoClamp2_SPIRV_SLessThanOp(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("spirv.Select", 4, context, {"spirv.GL.SClamp"}) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range min(op0->getOperands());
    ::mlir::Operation::operand_range min0(op0->getOperands());
    ::mlir::Operation::operand_range input(op0->getOperands());
    ::mlir::Operation::operand_range input0(op0->getOperands());
    ::mlir::Operation::operand_range input1(op0->getOperands());
    ::mlir::Operation::operand_range max(op0->getOperands());
    ::mlir::Operation::operand_range max0(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::spirv::SelectOp>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::spirv::SLessThanOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::spirv::SLessThanOp type";
        });
      }
      max0 = castedOp1.getODSOperands(0);
      input1 = castedOp1.getODSOperands(1);
      tblgen_ops.push_back(op1);
    }
    max = castedOp0.getODSOperands(1);
    {
      auto *op1 = (*castedOp0.getODSOperands(2).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 2 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::spirv::SelectOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::spirv::SelectOp type";
        });
      }
      {
        auto *op2 = (*castedOp1.getODSOperands(0).begin()).getDefiningOp();
        if (!(op2)){
          return rewriter.notifyMatchFailure(castedOp1, [&](::mlir::Diagnostic &diag) {
            diag << "There's no operation that defines operand 0 of castedOp1";
          });
        }
        auto castedOp2 = ::llvm::dyn_cast<::mlir::spirv::SLessThanOp>(op2); (void)castedOp2;
        if (!(castedOp2)){
          return rewriter.notifyMatchFailure(op2, [&](::mlir::Diagnostic &diag) {
            diag << "castedOp2 is not ::mlir::spirv::SLessThanOp type";
          });
        }
        input0 = castedOp2.getODSOperands(0);
        min0 = castedOp2.getODSOperands(1);
        tblgen_ops.push_back(op2);
      }
      min = castedOp1.getODSOperands(1);
      input = castedOp1.getODSOperands(2);
      tblgen_ops.push_back(op1);
    }
    if (!(*min.begin() == *min0.begin())){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "Operands 'min' and 'min0' must be equal";
      });
    }
    if (!(*input.begin() == *input0.begin())){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "Operands 'input' and 'input0' must be equal";
      });
    }
    if (!(*input.begin() == *input1.begin())){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "Operands 'input' and 'input1' must be equal";
      });
    }
    if (!(*max.begin() == *max0.begin())){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "Operands 'max' and 'max0' must be equal";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc(), tblgen_ops[2]->getLoc(), tblgen_ops[3]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::spirv::GLSClampOp tblgen_GLSClampOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*input.begin()));
      tblgen_values.push_back((*min.begin()));
      tblgen_values.push_back((*max.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_GLSClampOp_0 = rewriter.create<::mlir::spirv::GLSClampOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_GLSClampOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/SPIRV/IR/SPIRVCanonicalization.td:74
*/
struct ConvertComparisonIntoClamp2_SPIRV_ULessThanEqualOp : public ::mlir::RewritePattern {
  ConvertComparisonIntoClamp2_SPIRV_ULessThanEqualOp(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("spirv.Select", 4, context, {"spirv.GL.UClamp"}) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range min(op0->getOperands());
    ::mlir::Operation::operand_range min0(op0->getOperands());
    ::mlir::Operation::operand_range input(op0->getOperands());
    ::mlir::Operation::operand_range input0(op0->getOperands());
    ::mlir::Operation::operand_range input1(op0->getOperands());
    ::mlir::Operation::operand_range max(op0->getOperands());
    ::mlir::Operation::operand_range max0(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::spirv::SelectOp>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::spirv::ULessThanEqualOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::spirv::ULessThanEqualOp type";
        });
      }
      max0 = castedOp1.getODSOperands(0);
      input1 = castedOp1.getODSOperands(1);
      tblgen_ops.push_back(op1);
    }
    max = castedOp0.getODSOperands(1);
    {
      auto *op1 = (*castedOp0.getODSOperands(2).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 2 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::spirv::SelectOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::spirv::SelectOp type";
        });
      }
      {
        auto *op2 = (*castedOp1.getODSOperands(0).begin()).getDefiningOp();
        if (!(op2)){
          return rewriter.notifyMatchFailure(castedOp1, [&](::mlir::Diagnostic &diag) {
            diag << "There's no operation that defines operand 0 of castedOp1";
          });
        }
        auto castedOp2 = ::llvm::dyn_cast<::mlir::spirv::ULessThanEqualOp>(op2); (void)castedOp2;
        if (!(castedOp2)){
          return rewriter.notifyMatchFailure(op2, [&](::mlir::Diagnostic &diag) {
            diag << "castedOp2 is not ::mlir::spirv::ULessThanEqualOp type";
          });
        }
        input0 = castedOp2.getODSOperands(0);
        min0 = castedOp2.getODSOperands(1);
        tblgen_ops.push_back(op2);
      }
      min = castedOp1.getODSOperands(1);
      input = castedOp1.getODSOperands(2);
      tblgen_ops.push_back(op1);
    }
    if (!(*min.begin() == *min0.begin())){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "Operands 'min' and 'min0' must be equal";
      });
    }
    if (!(*input.begin() == *input0.begin())){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "Operands 'input' and 'input0' must be equal";
      });
    }
    if (!(*input.begin() == *input1.begin())){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "Operands 'input' and 'input1' must be equal";
      });
    }
    if (!(*max.begin() == *max0.begin())){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "Operands 'max' and 'max0' must be equal";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc(), tblgen_ops[2]->getLoc(), tblgen_ops[3]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::spirv::GLUClampOp tblgen_GLUClampOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*input.begin()));
      tblgen_values.push_back((*min.begin()));
      tblgen_values.push_back((*max.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_GLUClampOp_0 = rewriter.create<::mlir::spirv::GLUClampOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_GLUClampOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/SPIRV/IR/SPIRVCanonicalization.td:74
*/
struct ConvertComparisonIntoClamp2_SPIRV_ULessThanOp : public ::mlir::RewritePattern {
  ConvertComparisonIntoClamp2_SPIRV_ULessThanOp(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("spirv.Select", 4, context, {"spirv.GL.UClamp"}) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range min(op0->getOperands());
    ::mlir::Operation::operand_range min0(op0->getOperands());
    ::mlir::Operation::operand_range input(op0->getOperands());
    ::mlir::Operation::operand_range input0(op0->getOperands());
    ::mlir::Operation::operand_range input1(op0->getOperands());
    ::mlir::Operation::operand_range max(op0->getOperands());
    ::mlir::Operation::operand_range max0(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::spirv::SelectOp>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::spirv::ULessThanOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::spirv::ULessThanOp type";
        });
      }
      max0 = castedOp1.getODSOperands(0);
      input1 = castedOp1.getODSOperands(1);
      tblgen_ops.push_back(op1);
    }
    max = castedOp0.getODSOperands(1);
    {
      auto *op1 = (*castedOp0.getODSOperands(2).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 2 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::spirv::SelectOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::spirv::SelectOp type";
        });
      }
      {
        auto *op2 = (*castedOp1.getODSOperands(0).begin()).getDefiningOp();
        if (!(op2)){
          return rewriter.notifyMatchFailure(castedOp1, [&](::mlir::Diagnostic &diag) {
            diag << "There's no operation that defines operand 0 of castedOp1";
          });
        }
        auto castedOp2 = ::llvm::dyn_cast<::mlir::spirv::ULessThanOp>(op2); (void)castedOp2;
        if (!(castedOp2)){
          return rewriter.notifyMatchFailure(op2, [&](::mlir::Diagnostic &diag) {
            diag << "castedOp2 is not ::mlir::spirv::ULessThanOp type";
          });
        }
        input0 = castedOp2.getODSOperands(0);
        min0 = castedOp2.getODSOperands(1);
        tblgen_ops.push_back(op2);
      }
      min = castedOp1.getODSOperands(1);
      input = castedOp1.getODSOperands(2);
      tblgen_ops.push_back(op1);
    }
    if (!(*min.begin() == *min0.begin())){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "Operands 'min' and 'min0' must be equal";
      });
    }
    if (!(*input.begin() == *input0.begin())){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "Operands 'input' and 'input0' must be equal";
      });
    }
    if (!(*input.begin() == *input1.begin())){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "Operands 'input' and 'input1' must be equal";
      });
    }
    if (!(*max.begin() == *max0.begin())){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "Operands 'max' and 'max0' must be equal";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc(), tblgen_ops[2]->getLoc(), tblgen_ops[3]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::spirv::GLUClampOp tblgen_GLUClampOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*input.begin()));
      tblgen_values.push_back((*min.begin()));
      tblgen_values.push_back((*max.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_GLUClampOp_0 = rewriter.create<::mlir::spirv::GLUClampOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_GLUClampOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/SPIRV/IR/SPIRVCanonicalization.td:27
*/
struct ConvertLogicalNotOfIEqual : public ::mlir::RewritePattern {
  ConvertLogicalNotOfIEqual(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("spirv.LogicalNot", 2, context, {"spirv.INotEqual"}) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range rhs(op0->getOperands());
    ::mlir::Operation::operand_range lhs(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::spirv::LogicalNotOp>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::spirv::IEqualOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::spirv::IEqualOp type";
        });
      }
      lhs = castedOp1.getODSOperands(0);
      rhs = castedOp1.getODSOperands(1);
      tblgen_ops.push_back(op1);
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::spirv::INotEqualOp tblgen_INotEqualOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*lhs.begin()));
      tblgen_values.push_back((*rhs.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_INotEqualOp_0 = rewriter.create<::mlir::spirv::INotEqualOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_INotEqualOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/SPIRV/IR/SPIRVCanonicalization.td:31
*/
struct ConvertLogicalNotOfINotEqual : public ::mlir::RewritePattern {
  ConvertLogicalNotOfINotEqual(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("spirv.LogicalNot", 2, context, {"spirv.IEqual"}) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range rhs(op0->getOperands());
    ::mlir::Operation::operand_range lhs(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::spirv::LogicalNotOp>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::spirv::INotEqualOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::spirv::INotEqualOp type";
        });
      }
      lhs = castedOp1.getODSOperands(0);
      rhs = castedOp1.getODSOperands(1);
      tblgen_ops.push_back(op1);
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::spirv::IEqualOp tblgen_IEqualOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*lhs.begin()));
      tblgen_values.push_back((*rhs.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_IEqualOp_0 = rewriter.create<::mlir::spirv::IEqualOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_IEqualOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/SPIRV/IR/SPIRVCanonicalization.td:35
*/
struct ConvertLogicalNotOfLogicalEqual : public ::mlir::RewritePattern {
  ConvertLogicalNotOfLogicalEqual(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("spirv.LogicalNot", 2, context, {"spirv.LogicalNotEqual"}) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range rhs(op0->getOperands());
    ::mlir::Operation::operand_range lhs(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::spirv::LogicalNotOp>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::spirv::LogicalEqualOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::spirv::LogicalEqualOp type";
        });
      }
      lhs = castedOp1.getODSOperands(0);
      rhs = castedOp1.getODSOperands(1);
      tblgen_ops.push_back(op1);
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::spirv::LogicalNotEqualOp tblgen_LogicalNotEqualOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*lhs.begin()));
      tblgen_values.push_back((*rhs.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_LogicalNotEqualOp_0 = rewriter.create<::mlir::spirv::LogicalNotEqualOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_LogicalNotEqualOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/SPIRV/IR/SPIRVCanonicalization.td:39
*/
struct ConvertLogicalNotOfLogicalNotEqual : public ::mlir::RewritePattern {
  ConvertLogicalNotOfLogicalNotEqual(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("spirv.LogicalNot", 2, context, {"spirv.LogicalEqual"}) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range rhs(op0->getOperands());
    ::mlir::Operation::operand_range lhs(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::spirv::LogicalNotOp>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::spirv::LogicalNotEqualOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::spirv::LogicalNotEqualOp type";
        });
      }
      lhs = castedOp1.getODSOperands(0);
      rhs = castedOp1.getODSOperands(1);
      tblgen_ops.push_back(op1);
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::spirv::LogicalEqualOp tblgen_LogicalEqualOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*lhs.begin()));
      tblgen_values.push_back((*rhs.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_LogicalEqualOp_0 = rewriter.create<::mlir::spirv::LogicalEqualOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_LogicalEqualOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

void LLVM_ATTRIBUTE_UNUSED populateWithGenerated(::mlir::RewritePatternSet &patterns) {
  patterns.add<ConvertChainedBitcast>(patterns.getContext());
  patterns.add<ConvertComparisonIntoClamp1_SPIRV_FOrdLessThanEqualOp>(patterns.getContext());
  patterns.add<ConvertComparisonIntoClamp1_SPIRV_FOrdLessThanOp>(patterns.getContext());
  patterns.add<ConvertComparisonIntoClamp1_SPIRV_SLessThanEqualOp>(patterns.getContext());
  patterns.add<ConvertComparisonIntoClamp1_SPIRV_SLessThanOp>(patterns.getContext());
  patterns.add<ConvertComparisonIntoClamp1_SPIRV_ULessThanEqualOp>(patterns.getContext());
  patterns.add<ConvertComparisonIntoClamp1_SPIRV_ULessThanOp>(patterns.getContext());
  patterns.add<ConvertComparisonIntoClamp2_SPIRV_FOrdLessThanEqualOp>(patterns.getContext());
  patterns.add<ConvertComparisonIntoClamp2_SPIRV_FOrdLessThanOp>(patterns.getContext());
  patterns.add<ConvertComparisonIntoClamp2_SPIRV_SLessThanEqualOp>(patterns.getContext());
  patterns.add<ConvertComparisonIntoClamp2_SPIRV_SLessThanOp>(patterns.getContext());
  patterns.add<ConvertComparisonIntoClamp2_SPIRV_ULessThanEqualOp>(patterns.getContext());
  patterns.add<ConvertComparisonIntoClamp2_SPIRV_ULessThanOp>(patterns.getContext());
  patterns.add<ConvertLogicalNotOfIEqual>(patterns.getContext());
  patterns.add<ConvertLogicalNotOfINotEqual>(patterns.getContext());
  patterns.add<ConvertLogicalNotOfLogicalEqual>(patterns.getContext());
  patterns.add<ConvertLogicalNotOfLogicalNotEqual>(patterns.getContext());
}
