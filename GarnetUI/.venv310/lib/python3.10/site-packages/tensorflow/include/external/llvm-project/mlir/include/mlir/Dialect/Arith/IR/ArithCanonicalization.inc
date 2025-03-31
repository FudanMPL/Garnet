/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Rewriters                                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/


static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_ArithCanonicalization0(
    ::mlir::PatternRewriter &rewriter, ::mlir::Operation *op, ::mlir::Attribute attr,
    ::llvm::StringRef failureStr) {
  if (!((attr.isa<::mlir::IntegerAttr>()))) {
    return rewriter.notifyMatchFailure(op, [&](::mlir::Diagnostic &diag) {
      diag << failureStr << ": arbitrary integer attribute";
    });
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_ArithCanonicalization1(
    ::mlir::PatternRewriter &rewriter, ::mlir::Operation *op, ::mlir::Attribute attr,
    ::llvm::StringRef failureStr) {
  if (!((attr == rewriter.getIntegerAttr(rewriter.getIntegerType(1), 1)))) {
    return rewriter.notifyMatchFailure(op, [&](::mlir::Diagnostic &diag) {
      diag << failureStr << ": constant attribute 1";
    });
  }
  return ::mlir::success();
}
static ::mlir::LogicalResult static_dag_matcher_0(::mlir::PatternRewriter &rewriter, ::mlir::Operation *op0, ::llvm::SmallVector<::mlir::Operation *, 4> &tblgen_ops, ::mlir::TypedAttr &c0) {
  (void)tblgen_ops;
  auto castedOp1 = ::llvm::dyn_cast<::mlir::arith::ConstantOp>(op0); (void)castedOp1;
  if (!(castedOp1)){
    return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
      diag << "castedOp1 is not ::mlir::arith::ConstantOp type";
    });
  }
  {
    auto tblgen_attr = op0->getAttrOfType<::mlir::TypedAttr>("value");(void)tblgen_attr;
    if (!(tblgen_attr)){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "expected op 'arith.constant' to have attribute 'value' of type '::mlir::TypedAttr'";
      });
    }
    if(::mlir::failed(__mlir_ods_local_attr_constraint_ArithCanonicalization0(rewriter, op0, tblgen_attr, "op 'arith.constant' attribute 'value' failed to satisfy constraint: 'arbitrary integer attribute'"))) {
      return ::mlir::failure();
    }
    c0 = tblgen_attr;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult static_dag_matcher_1(::mlir::PatternRewriter &rewriter, ::mlir::Operation *op0, ::llvm::SmallVector<::mlir::Operation *, 4> &tblgen_ops, ::mlir::TypedAttr &c0, ::mlir::Operation::operand_range &x) {
  (void)tblgen_ops;
  auto castedOp1 = ::llvm::dyn_cast<::mlir::arith::AddIOp>(op0); (void)castedOp1;
  if (!(castedOp1)){
    return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
      diag << "castedOp1 is not ::mlir::arith::AddIOp type";
    });
  }
  x = castedOp1.getODSOperands(0);
  {
    auto *op2 = (*castedOp1.getODSOperands(1).begin()).getDefiningOp();
    if (!(op2)){
      return rewriter.notifyMatchFailure(castedOp1, [&](::mlir::Diagnostic &diag) {
        diag << "There's no operation that defines operand 1 of castedOp1";
      });
    }
    if(::mlir::failed(static_dag_matcher_0(rewriter, op2, tblgen_ops, c0))) {
      return ::mlir::failure();
    }
    tblgen_ops.push_back(op2);
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult static_dag_matcher_2(::mlir::PatternRewriter &rewriter, ::mlir::Operation *op0, ::llvm::SmallVector<::mlir::Operation *, 4> &tblgen_ops, ::mlir::TypedAttr &c1) {
  (void)tblgen_ops;
  auto castedOp1 = ::llvm::dyn_cast<::mlir::arith::ConstantOp>(op0); (void)castedOp1;
  if (!(castedOp1)){
    return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
      diag << "castedOp1 is not ::mlir::arith::ConstantOp type";
    });
  }
  {
    auto tblgen_attr = op0->getAttrOfType<::mlir::TypedAttr>("value");(void)tblgen_attr;
    if (!(tblgen_attr)){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "expected op 'arith.constant' to have attribute 'value' of type '::mlir::TypedAttr'";
      });
    }
    if(::mlir::failed(__mlir_ods_local_attr_constraint_ArithCanonicalization0(rewriter, op0, tblgen_attr, "op 'arith.constant' attribute 'value' failed to satisfy constraint: 'arbitrary integer attribute'"))) {
      return ::mlir::failure();
    }
    c1 = tblgen_attr;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult static_dag_matcher_3(::mlir::PatternRewriter &rewriter, ::mlir::Operation *op0, ::llvm::SmallVector<::mlir::Operation *, 4> &tblgen_ops, ::mlir::Operation::operand_range &x, ::mlir::TypedAttr &c0) {
  (void)tblgen_ops;
  auto castedOp1 = ::llvm::dyn_cast<::mlir::arith::SubIOp>(op0); (void)castedOp1;
  if (!(castedOp1)){
    return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
      diag << "castedOp1 is not ::mlir::arith::SubIOp type";
    });
  }
  {
    auto *op2 = (*castedOp1.getODSOperands(0).begin()).getDefiningOp();
    if (!(op2)){
      return rewriter.notifyMatchFailure(castedOp1, [&](::mlir::Diagnostic &diag) {
        diag << "There's no operation that defines operand 0 of castedOp1";
      });
    }
    if(::mlir::failed(static_dag_matcher_0(rewriter, op2, tblgen_ops, c0))) {
      return ::mlir::failure();
    }
    tblgen_ops.push_back(op2);
  }
  x = castedOp1.getODSOperands(1);
  return ::mlir::success();
}

static ::mlir::LogicalResult static_dag_matcher_4(::mlir::PatternRewriter &rewriter, ::mlir::Operation *op0, ::llvm::SmallVector<::mlir::Operation *, 4> &tblgen_ops, ::mlir::TypedAttr &c0, ::mlir::Operation::operand_range &x) {
  (void)tblgen_ops;
  auto castedOp1 = ::llvm::dyn_cast<::mlir::arith::SubIOp>(op0); (void)castedOp1;
  if (!(castedOp1)){
    return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
      diag << "castedOp1 is not ::mlir::arith::SubIOp type";
    });
  }
  x = castedOp1.getODSOperands(0);
  {
    auto *op2 = (*castedOp1.getODSOperands(1).begin()).getDefiningOp();
    if (!(op2)){
      return rewriter.notifyMatchFailure(castedOp1, [&](::mlir::Diagnostic &diag) {
        diag << "There's no operation that defines operand 1 of castedOp1";
      });
    }
    if(::mlir::failed(static_dag_matcher_0(rewriter, op2, tblgen_ops, c0))) {
      return ::mlir::failure();
    }
    tblgen_ops.push_back(op2);
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult static_dag_matcher_5(::mlir::PatternRewriter &rewriter, ::mlir::Operation *op0, ::llvm::SmallVector<::mlir::Operation *, 4> &tblgen_ops, ::mlir::Operation::operand_range &x) {
  (void)tblgen_ops;
  auto castedOp1 = ::llvm::dyn_cast<::mlir::arith::ExtSIOp>(op0); (void)castedOp1;
  if (!(castedOp1)){
    return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
      diag << "castedOp1 is not ::mlir::arith::ExtSIOp type";
    });
  }
  x = castedOp1.getODSOperands(0);
  return ::mlir::success();
}

static ::mlir::LogicalResult static_dag_matcher_6(::mlir::PatternRewriter &rewriter, ::mlir::Operation *op0, ::llvm::SmallVector<::mlir::Operation *, 4> &tblgen_ops, ::mlir::Operation::operand_range &y) {
  (void)tblgen_ops;
  auto castedOp1 = ::llvm::dyn_cast<::mlir::arith::ExtSIOp>(op0); (void)castedOp1;
  if (!(castedOp1)){
    return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
      diag << "castedOp1 is not ::mlir::arith::ExtSIOp type";
    });
  }
  y = castedOp1.getODSOperands(0);
  return ::mlir::success();
}

static ::mlir::LogicalResult static_dag_matcher_7(::mlir::PatternRewriter &rewriter, ::mlir::Operation *op0, ::llvm::SmallVector<::mlir::Operation *, 4> &tblgen_ops, ::mlir::Operation::operand_range &x) {
  (void)tblgen_ops;
  auto castedOp1 = ::llvm::dyn_cast<::mlir::arith::ExtUIOp>(op0); (void)castedOp1;
  if (!(castedOp1)){
    return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
      diag << "castedOp1 is not ::mlir::arith::ExtUIOp type";
    });
  }
  x = castedOp1.getODSOperands(0);
  return ::mlir::success();
}

static ::mlir::LogicalResult static_dag_matcher_8(::mlir::PatternRewriter &rewriter, ::mlir::Operation *op0, ::llvm::SmallVector<::mlir::Operation *, 4> &tblgen_ops, ::mlir::Operation::operand_range &y) {
  (void)tblgen_ops;
  auto castedOp1 = ::llvm::dyn_cast<::mlir::arith::ExtUIOp>(op0); (void)castedOp1;
  if (!(castedOp1)){
    return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
      diag << "castedOp1 is not ::mlir::arith::ExtUIOp type";
    });
  }
  y = castedOp1.getODSOperands(0);
  return ::mlir::success();
}

static ::mlir::LogicalResult static_dag_matcher_9(::mlir::PatternRewriter &rewriter, ::mlir::Operation *op0, ::llvm::SmallVector<::mlir::Operation *, 4> &tblgen_ops, ::mlir::Operation::operand_range &x) {
  (void)tblgen_ops;
  auto castedOp1 = ::llvm::dyn_cast<::mlir::arith::NegFOp>(op0); (void)castedOp1;
  if (!(castedOp1)){
    return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
      diag << "castedOp1 is not ::mlir::arith::NegFOp type";
    });
  }
  x = castedOp1.getODSOperands(0);
  return ::mlir::success();
}

static ::mlir::LogicalResult static_dag_matcher_10(::mlir::PatternRewriter &rewriter, ::mlir::Operation *op0, ::llvm::SmallVector<::mlir::Operation *, 4> &tblgen_ops, ::mlir::Operation::operand_range &y) {
  (void)tblgen_ops;
  auto castedOp1 = ::llvm::dyn_cast<::mlir::arith::NegFOp>(op0); (void)castedOp1;
  if (!(castedOp1)){
    return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
      diag << "castedOp1 is not ::mlir::arith::NegFOp type";
    });
  }
  y = castedOp1.getODSOperands(0);
  return ::mlir::success();
}

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/Arith/IR/ArithCanonicalization.td:32
*/
struct AddIAddConstant : public ::mlir::RewritePattern {
  AddIAddConstant(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("arith.addi", 4, context, {"arith.addi", "arith.constant"}) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::TypedAttr c1;
    ::mlir::TypedAttr c0;
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::mlir::arith::AddIOp res;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::arith::AddIOp>(op0); (void)castedOp0;
    res = castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_1(rewriter, op1, tblgen_ops, c0, x))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }
    {
      auto *op1 = (*castedOp0.getODSOperands(1).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 1 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_2(rewriter, op1, tblgen_ops, c1))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc(), tblgen_ops[2]->getLoc(), tblgen_ops[3]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = addIntegerAttrs(rewriter, (*res.getODSResults(0).begin()), c0, c1); (void)nativeVar_0;
    ::mlir::arith::ConstantOp tblgen_ConstantOp_1;
    {
      tblgen_ConstantOp_1 = rewriter.create<::mlir::arith::ConstantOp>(odsLoc,
        /*value=*/nativeVar_0
      );
    }
    ::mlir::arith::AddIOp tblgen_AddIOp_2;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*x.begin()));
      tblgen_values.push_back((*tblgen_ConstantOp_1.getODSResults(0).begin()));
      tblgen_AddIOp_2 = rewriter.create<::mlir::arith::AddIOp>(odsLoc, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_AddIOp_2.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/Arith/IR/ArithCanonicalization.td:46
*/
struct AddISubConstantLHS : public ::mlir::RewritePattern {
  AddISubConstantLHS(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("arith.addi", 4, context, {"arith.constant", "arith.subi"}) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::mlir::TypedAttr c1;
    ::mlir::TypedAttr c0;
    ::mlir::arith::AddIOp res;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::arith::AddIOp>(op0); (void)castedOp0;
    res = castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_3(rewriter, op1, tblgen_ops, x, c0))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }
    {
      auto *op1 = (*castedOp0.getODSOperands(1).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 1 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_2(rewriter, op1, tblgen_ops, c1))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc(), tblgen_ops[2]->getLoc(), tblgen_ops[3]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = addIntegerAttrs(rewriter, (*res.getODSResults(0).begin()), c0, c1); (void)nativeVar_0;
    ::mlir::arith::ConstantOp tblgen_ConstantOp_1;
    {
      tblgen_ConstantOp_1 = rewriter.create<::mlir::arith::ConstantOp>(odsLoc,
        /*value=*/nativeVar_0
      );
    }
    ::mlir::arith::SubIOp tblgen_SubIOp_2;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_ConstantOp_1.getODSResults(0).begin()));
      tblgen_values.push_back((*x.begin()));
      tblgen_SubIOp_2 = rewriter.create<::mlir::arith::SubIOp>(odsLoc, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_SubIOp_2.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/Arith/IR/ArithCanonicalization.td:39
*/
struct AddISubConstantRHS : public ::mlir::RewritePattern {
  AddISubConstantRHS(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("arith.addi", 4, context, {"arith.addi", "arith.constant"}) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::TypedAttr c1;
    ::mlir::TypedAttr c0;
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::mlir::arith::AddIOp res;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::arith::AddIOp>(op0); (void)castedOp0;
    res = castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_4(rewriter, op1, tblgen_ops, c0, x))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }
    {
      auto *op1 = (*castedOp0.getODSOperands(1).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 1 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_2(rewriter, op1, tblgen_ops, c1))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc(), tblgen_ops[2]->getLoc(), tblgen_ops[3]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = subIntegerAttrs(rewriter, (*res.getODSResults(0).begin()), c1, c0); (void)nativeVar_0;
    ::mlir::arith::ConstantOp tblgen_ConstantOp_1;
    {
      tblgen_ConstantOp_1 = rewriter.create<::mlir::arith::ConstantOp>(odsLoc,
        /*value=*/nativeVar_0
      );
    }
    ::mlir::arith::AddIOp tblgen_AddIOp_2;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*x.begin()));
      tblgen_values.push_back((*tblgen_ConstantOp_1.getODSResults(0).begin()));
      tblgen_AddIOp_2 = rewriter.create<::mlir::arith::AddIOp>(odsLoc, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_AddIOp_2.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/Arith/IR/ArithCanonicalization.td:195
*/
struct AndOfExtSI : public ::mlir::RewritePattern {
  AndOfExtSI(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("arith.andi", 3, context, {"arith.andi", "arith.extsi"}) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range y(op0->getOperands());
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::arith::AndIOp>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_5(rewriter, op1, tblgen_ops, x))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }
    {
      auto *op1 = (*castedOp0.getODSOperands(1).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 1 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_6(rewriter, op1, tblgen_ops, y))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }
    if (!(((*x.begin()).getType() == (*y.begin()).getType()))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'x, y' failed to satisfy constraint: ''";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc(), tblgen_ops[2]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::arith::AndIOp tblgen_AndIOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*x.begin()));
      tblgen_values.push_back((*y.begin()));
      tblgen_AndIOp_0 = rewriter.create<::mlir::arith::AndIOp>(odsLoc, tblgen_values, tblgen_attrs);
    }
    ::mlir::arith::ExtSIOp tblgen_ExtSIOp_1;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_AndIOp_0.getODSResults(0).begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_ExtSIOp_1 = rewriter.create<::mlir::arith::ExtSIOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_ExtSIOp_1.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/Arith/IR/ArithCanonicalization.td:190
*/
struct AndOfExtUI : public ::mlir::RewritePattern {
  AndOfExtUI(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("arith.andi", 3, context, {"arith.andi", "arith.extui"}) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range y(op0->getOperands());
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::arith::AndIOp>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_7(rewriter, op1, tblgen_ops, x))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }
    {
      auto *op1 = (*castedOp0.getODSOperands(1).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 1 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_8(rewriter, op1, tblgen_ops, y))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }
    if (!(((*x.begin()).getType() == (*y.begin()).getType()))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'x, y' failed to satisfy constraint: ''";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc(), tblgen_ops[2]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::arith::AndIOp tblgen_AndIOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*x.begin()));
      tblgen_values.push_back((*y.begin()));
      tblgen_AndIOp_0 = rewriter.create<::mlir::arith::AndIOp>(odsLoc, tblgen_values, tblgen_attrs);
    }
    ::mlir::arith::ExtUIOp tblgen_ExtUIOp_1;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_AndIOp_0.getODSResults(0).begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_ExtUIOp_1 = rewriter.create<::mlir::arith::ExtUIOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_ExtUIOp_1.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/Arith/IR/ArithCanonicalization.td:174
*/
struct BitcastOfBitcast : public ::mlir::RewritePattern {
  BitcastOfBitcast(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("arith.bitcast", 2, context, {}) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::arith::BitcastOp>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::arith::BitcastOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::arith::BitcastOp type";
        });
      }
      x = castedOp1.getODSOperands(0);
      tblgen_ops.push_back(op1);
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ x }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/Arith/IR/ArithCanonicalization.td:123
*/
struct CmpIExtSI : public ::mlir::RewritePattern {
  CmpIExtSI(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("arith.cmpi", 3, context, {"arith.cmpi"}) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range b(op0->getOperands());
    ::mlir::Operation::operand_range a(op0->getOperands());
    ::mlir::arith::CmpIPredicateAttr pred;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::arith::CmpIOp>(op0); (void)castedOp0;
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::arith::CmpIPredicateAttr>("predicate");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'arith.cmpi' to have attribute 'predicate' of type '::mlir::arith::CmpIPredicateAttr'";
        });
      }
      pred = tblgen_attr;
    }
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::arith::ExtSIOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::arith::ExtSIOp type";
        });
      }
      a = castedOp1.getODSOperands(0);
      tblgen_ops.push_back(op1);
    }
    {
      auto *op1 = (*castedOp0.getODSOperands(1).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 1 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::arith::ExtSIOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::arith::ExtSIOp type";
        });
      }
      b = castedOp1.getODSOperands(0);
      tblgen_ops.push_back(op1);
    }
    if (!(((*a.begin()).getType() == (*b.begin()).getType()))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'a, b' failed to satisfy constraint: ''";
      });
    }
    if (!((pred.getValue() == arith::CmpIPredicate::eq || pred.getValue() == arith::CmpIPredicate::ne))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'pred' failed to satisfy constraint: ''";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc(), tblgen_ops[2]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::arith::CmpIOp tblgen_CmpIOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      if (auto tmpAttr = pred) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("predicate"), tmpAttr);
      }
      tblgen_values.push_back((*a.begin()));
      tblgen_values.push_back((*b.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_CmpIOp_0 = rewriter.create<::mlir::arith::CmpIOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_CmpIOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/Arith/IR/ArithCanonicalization.td:132
*/
struct CmpIExtUI : public ::mlir::RewritePattern {
  CmpIExtUI(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("arith.cmpi", 3, context, {"arith.cmpi"}) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range b(op0->getOperands());
    ::mlir::Operation::operand_range a(op0->getOperands());
    ::mlir::arith::CmpIPredicateAttr pred;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::arith::CmpIOp>(op0); (void)castedOp0;
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::arith::CmpIPredicateAttr>("predicate");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'arith.cmpi' to have attribute 'predicate' of type '::mlir::arith::CmpIPredicateAttr'";
        });
      }
      pred = tblgen_attr;
    }
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::arith::ExtUIOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::arith::ExtUIOp type";
        });
      }
      a = castedOp1.getODSOperands(0);
      tblgen_ops.push_back(op1);
    }
    {
      auto *op1 = (*castedOp0.getODSOperands(1).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 1 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::arith::ExtUIOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::arith::ExtUIOp type";
        });
      }
      b = castedOp1.getODSOperands(0);
      tblgen_ops.push_back(op1);
    }
    if (!(((*a.begin()).getType() == (*b.begin()).getType()))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'a, b' failed to satisfy constraint: ''";
      });
    }
    if (!((pred.getValue() == arith::CmpIPredicate::eq || pred.getValue() == arith::CmpIPredicate::ne))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'pred' failed to satisfy constraint: ''";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc(), tblgen_ops[2]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::arith::CmpIOp tblgen_CmpIOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      if (auto tmpAttr = pred) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("predicate"), tmpAttr);
      }
      tblgen_values.push_back((*a.begin()));
      tblgen_values.push_back((*b.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_CmpIOp_0 = rewriter.create<::mlir::arith::CmpIOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_CmpIOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/Arith/IR/ArithCanonicalization.td:227
*/
struct DivFOfNegF : public ::mlir::RewritePattern {
  DivFOfNegF(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("arith.divf", 3, context, {"arith.divf"}) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range y(op0->getOperands());
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::arith::DivFOp>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_9(rewriter, op1, tblgen_ops, x))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }
    {
      auto *op1 = (*castedOp0.getODSOperands(1).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 1 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_10(rewriter, op1, tblgen_ops, y))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }
    if (!(((*x.begin()).getType() == (*y.begin()).getType()))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'x, y' failed to satisfy constraint: ''";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc(), tblgen_ops[2]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::arith::DivFOp tblgen_DivFOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*x.begin()));
      tblgen_values.push_back((*y.begin()));
      tblgen_DivFOp_0 = rewriter.create<::mlir::arith::DivFOp>(odsLoc, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_DivFOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/Arith/IR/ArithCanonicalization.td:182
*/
struct ExtSIOfExtUI : public ::mlir::RewritePattern {
  ExtSIOfExtUI(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("arith.extsi", 2, context, {"arith.extui"}) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::arith::ExtSIOp>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_7(rewriter, op1, tblgen_ops, x))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::arith::ExtUIOp tblgen_ExtUIOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*x.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_ExtUIOp_0 = rewriter.create<::mlir::arith::ExtUIOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_ExtUIOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/Arith/IR/ArithCanonicalization.td:151
*/
struct IndexCastOfExtSI : public ::mlir::RewritePattern {
  IndexCastOfExtSI(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("arith.index_cast", 2, context, {"arith.index_cast"}) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::arith::IndexCastOp>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_5(rewriter, op1, tblgen_ops, x))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::arith::IndexCastOp tblgen_IndexCastOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*x.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_IndexCastOp_0 = rewriter.create<::mlir::arith::IndexCastOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_IndexCastOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/Arith/IR/ArithCanonicalization.td:145
*/
struct IndexCastOfIndexCast : public ::mlir::RewritePattern {
  IndexCastOfIndexCast(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("arith.index_cast", 2, context, {}) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::mlir::arith::IndexCastOp res;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::arith::IndexCastOp>(op0); (void)castedOp0;
    res = castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::arith::IndexCastOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::arith::IndexCastOp type";
        });
      }
      x = castedOp1.getODSOperands(0);
      tblgen_ops.push_back(op1);
    }
    if (!(((*res.getODSResults(0).begin()).getType() == (*x.begin()).getType()))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'res, x' failed to satisfy constraint: ''";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ x }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/Arith/IR/ArithCanonicalization.td:165
*/
struct IndexCastUIOfExtUI : public ::mlir::RewritePattern {
  IndexCastUIOfExtUI(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("arith.index_castui", 2, context, {"arith.index_castui"}) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::arith::IndexCastUIOp>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_7(rewriter, op1, tblgen_ops, x))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::arith::IndexCastUIOp tblgen_IndexCastUIOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*x.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_IndexCastUIOp_0 = rewriter.create<::mlir::arith::IndexCastUIOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_IndexCastUIOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/Arith/IR/ArithCanonicalization.td:159
*/
struct IndexCastUIOfIndexCastUI : public ::mlir::RewritePattern {
  IndexCastUIOfIndexCastUI(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("arith.index_castui", 2, context, {}) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::mlir::arith::IndexCastUIOp res;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::arith::IndexCastUIOp>(op0); (void)castedOp0;
    res = castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::arith::IndexCastUIOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::arith::IndexCastUIOp type";
        });
      }
      x = castedOp1.getODSOperands(0);
      tblgen_ops.push_back(op1);
    }
    if (!(((*res.getODSResults(0).begin()).getType() == (*x.begin()).getType()))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'res, x' failed to satisfy constraint: ''";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ x }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/Arith/IR/ArithCanonicalization.td:218
*/
struct MulFOfNegF : public ::mlir::RewritePattern {
  MulFOfNegF(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("arith.mulf", 3, context, {"arith.mulf"}) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range y(op0->getOperands());
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::arith::MulFOp>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_9(rewriter, op1, tblgen_ops, x))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }
    {
      auto *op1 = (*castedOp0.getODSOperands(1).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 1 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_10(rewriter, op1, tblgen_ops, y))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }
    if (!(((*x.begin()).getType() == (*y.begin()).getType()))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'x, y' failed to satisfy constraint: ''";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc(), tblgen_ops[2]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::arith::MulFOp tblgen_MulFOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*x.begin()));
      tblgen_values.push_back((*y.begin()));
      tblgen_MulFOp_0 = rewriter.create<::mlir::arith::MulFOp>(odsLoc, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_MulFOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/Arith/IR/ArithCanonicalization.td:209
*/
struct OrOfExtSI : public ::mlir::RewritePattern {
  OrOfExtSI(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("arith.ori", 3, context, {"arith.extsi", "arith.ori"}) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range y(op0->getOperands());
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::arith::OrIOp>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_5(rewriter, op1, tblgen_ops, x))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }
    {
      auto *op1 = (*castedOp0.getODSOperands(1).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 1 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_6(rewriter, op1, tblgen_ops, y))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }
    if (!(((*x.begin()).getType() == (*y.begin()).getType()))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'x, y' failed to satisfy constraint: ''";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc(), tblgen_ops[2]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::arith::OrIOp tblgen_OrIOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*x.begin()));
      tblgen_values.push_back((*y.begin()));
      tblgen_OrIOp_0 = rewriter.create<::mlir::arith::OrIOp>(odsLoc, tblgen_values, tblgen_attrs);
    }
    ::mlir::arith::ExtSIOp tblgen_ExtSIOp_1;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_OrIOp_0.getODSResults(0).begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_ExtSIOp_1 = rewriter.create<::mlir::arith::ExtSIOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_ExtSIOp_1.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/Arith/IR/ArithCanonicalization.td:204
*/
struct OrOfExtUI : public ::mlir::RewritePattern {
  OrOfExtUI(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("arith.ori", 3, context, {"arith.extui", "arith.ori"}) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range y(op0->getOperands());
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::arith::OrIOp>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_7(rewriter, op1, tblgen_ops, x))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }
    {
      auto *op1 = (*castedOp0.getODSOperands(1).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 1 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_8(rewriter, op1, tblgen_ops, y))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }
    if (!(((*x.begin()).getType() == (*y.begin()).getType()))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'x, y' failed to satisfy constraint: ''";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc(), tblgen_ops[2]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::arith::OrIOp tblgen_OrIOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*x.begin()));
      tblgen_values.push_back((*y.begin()));
      tblgen_OrIOp_0 = rewriter.create<::mlir::arith::OrIOp>(odsLoc, tblgen_values, tblgen_attrs);
    }
    ::mlir::arith::ExtUIOp tblgen_ExtUIOp_1;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_OrIOp_0.getODSResults(0).begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_ExtUIOp_1 = rewriter.create<::mlir::arith::ExtUIOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_ExtUIOp_1.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/Arith/IR/ArithCanonicalization.td:64
*/
struct SubILHSAddConstant : public ::mlir::RewritePattern {
  SubILHSAddConstant(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("arith.subi", 4, context, {"arith.constant", "arith.subi"}) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::mlir::TypedAttr c0;
    ::mlir::TypedAttr c1;
    ::mlir::arith::SubIOp res;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::arith::SubIOp>(op0); (void)castedOp0;
    res = castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_2(rewriter, op1, tblgen_ops, c1))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }
    {
      auto *op1 = (*castedOp0.getODSOperands(1).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 1 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_1(rewriter, op1, tblgen_ops, c0, x))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc(), tblgen_ops[2]->getLoc(), tblgen_ops[3]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = subIntegerAttrs(rewriter, (*res.getODSResults(0).begin()), c1, c0); (void)nativeVar_0;
    ::mlir::arith::ConstantOp tblgen_ConstantOp_1;
    {
      tblgen_ConstantOp_1 = rewriter.create<::mlir::arith::ConstantOp>(odsLoc,
        /*value=*/nativeVar_0
      );
    }
    ::mlir::arith::SubIOp tblgen_SubIOp_2;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_ConstantOp_1.getODSResults(0).begin()));
      tblgen_values.push_back((*x.begin()));
      tblgen_SubIOp_2 = rewriter.create<::mlir::arith::SubIOp>(odsLoc, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_SubIOp_2.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/Arith/IR/ArithCanonicalization.td:92
*/
struct SubILHSSubConstantLHS : public ::mlir::RewritePattern {
  SubILHSSubConstantLHS(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("arith.subi", 4, context, {"arith.addi", "arith.constant"}) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::mlir::TypedAttr c0;
    ::mlir::TypedAttr c1;
    ::mlir::arith::SubIOp res;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::arith::SubIOp>(op0); (void)castedOp0;
    res = castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_2(rewriter, op1, tblgen_ops, c1))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }
    {
      auto *op1 = (*castedOp0.getODSOperands(1).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 1 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_3(rewriter, op1, tblgen_ops, x, c0))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc(), tblgen_ops[2]->getLoc(), tblgen_ops[3]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = subIntegerAttrs(rewriter, (*res.getODSResults(0).begin()), c1, c0); (void)nativeVar_0;
    ::mlir::arith::ConstantOp tblgen_ConstantOp_1;
    {
      tblgen_ConstantOp_1 = rewriter.create<::mlir::arith::ConstantOp>(odsLoc,
        /*value=*/nativeVar_0
      );
    }
    ::mlir::arith::AddIOp tblgen_AddIOp_2;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*x.begin()));
      tblgen_values.push_back((*tblgen_ConstantOp_1.getODSResults(0).begin()));
      tblgen_AddIOp_2 = rewriter.create<::mlir::arith::AddIOp>(odsLoc, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_AddIOp_2.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/Arith/IR/ArithCanonicalization.td:85
*/
struct SubILHSSubConstantRHS : public ::mlir::RewritePattern {
  SubILHSSubConstantRHS(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("arith.subi", 4, context, {"arith.constant", "arith.subi"}) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::mlir::TypedAttr c0;
    ::mlir::TypedAttr c1;
    ::mlir::arith::SubIOp res;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::arith::SubIOp>(op0); (void)castedOp0;
    res = castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_2(rewriter, op1, tblgen_ops, c1))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }
    {
      auto *op1 = (*castedOp0.getODSOperands(1).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 1 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_4(rewriter, op1, tblgen_ops, c0, x))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc(), tblgen_ops[2]->getLoc(), tblgen_ops[3]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = addIntegerAttrs(rewriter, (*res.getODSResults(0).begin()), c0, c1); (void)nativeVar_0;
    ::mlir::arith::ConstantOp tblgen_ConstantOp_1;
    {
      tblgen_ConstantOp_1 = rewriter.create<::mlir::arith::ConstantOp>(odsLoc,
        /*value=*/nativeVar_0
      );
    }
    ::mlir::arith::SubIOp tblgen_SubIOp_2;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_ConstantOp_1.getODSResults(0).begin()));
      tblgen_values.push_back((*x.begin()));
      tblgen_SubIOp_2 = rewriter.create<::mlir::arith::SubIOp>(odsLoc, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_SubIOp_2.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/Arith/IR/ArithCanonicalization.td:57
*/
struct SubIRHSAddConstant : public ::mlir::RewritePattern {
  SubIRHSAddConstant(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("arith.subi", 4, context, {"arith.addi", "arith.constant"}) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::TypedAttr c1;
    ::mlir::TypedAttr c0;
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::mlir::arith::SubIOp res;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::arith::SubIOp>(op0); (void)castedOp0;
    res = castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_1(rewriter, op1, tblgen_ops, c0, x))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }
    {
      auto *op1 = (*castedOp0.getODSOperands(1).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 1 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_2(rewriter, op1, tblgen_ops, c1))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc(), tblgen_ops[2]->getLoc(), tblgen_ops[3]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = subIntegerAttrs(rewriter, (*res.getODSResults(0).begin()), c0, c1); (void)nativeVar_0;
    ::mlir::arith::ConstantOp tblgen_ConstantOp_1;
    {
      tblgen_ConstantOp_1 = rewriter.create<::mlir::arith::ConstantOp>(odsLoc,
        /*value=*/nativeVar_0
      );
    }
    ::mlir::arith::AddIOp tblgen_AddIOp_2;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*x.begin()));
      tblgen_values.push_back((*tblgen_ConstantOp_1.getODSResults(0).begin()));
      tblgen_AddIOp_2 = rewriter.create<::mlir::arith::AddIOp>(odsLoc, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_AddIOp_2.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/Arith/IR/ArithCanonicalization.td:78
*/
struct SubIRHSSubConstantLHS : public ::mlir::RewritePattern {
  SubIRHSSubConstantLHS(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("arith.subi", 4, context, {"arith.constant", "arith.subi"}) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::mlir::TypedAttr c1;
    ::mlir::TypedAttr c0;
    ::mlir::arith::SubIOp res;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::arith::SubIOp>(op0); (void)castedOp0;
    res = castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_3(rewriter, op1, tblgen_ops, x, c0))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }
    {
      auto *op1 = (*castedOp0.getODSOperands(1).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 1 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_2(rewriter, op1, tblgen_ops, c1))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc(), tblgen_ops[2]->getLoc(), tblgen_ops[3]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = subIntegerAttrs(rewriter, (*res.getODSResults(0).begin()), c0, c1); (void)nativeVar_0;
    ::mlir::arith::ConstantOp tblgen_ConstantOp_1;
    {
      tblgen_ConstantOp_1 = rewriter.create<::mlir::arith::ConstantOp>(odsLoc,
        /*value=*/nativeVar_0
      );
    }
    ::mlir::arith::SubIOp tblgen_SubIOp_2;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_ConstantOp_1.getODSResults(0).begin()));
      tblgen_values.push_back((*x.begin()));
      tblgen_SubIOp_2 = rewriter.create<::mlir::arith::SubIOp>(odsLoc, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_SubIOp_2.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/Arith/IR/ArithCanonicalization.td:71
*/
struct SubIRHSSubConstantRHS : public ::mlir::RewritePattern {
  SubIRHSSubConstantRHS(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("arith.subi", 4, context, {"arith.constant", "arith.subi"}) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::TypedAttr c1;
    ::mlir::TypedAttr c0;
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::mlir::arith::SubIOp res;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::arith::SubIOp>(op0); (void)castedOp0;
    res = castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_4(rewriter, op1, tblgen_ops, c0, x))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }
    {
      auto *op1 = (*castedOp0.getODSOperands(1).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 1 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_2(rewriter, op1, tblgen_ops, c1))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc(), tblgen_ops[2]->getLoc(), tblgen_ops[3]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = addIntegerAttrs(rewriter, (*res.getODSResults(0).begin()), c0, c1); (void)nativeVar_0;
    ::mlir::arith::ConstantOp tblgen_ConstantOp_1;
    {
      tblgen_ConstantOp_1 = rewriter.create<::mlir::arith::ConstantOp>(odsLoc,
        /*value=*/nativeVar_0
      );
    }
    ::mlir::arith::SubIOp tblgen_SubIOp_2;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*x.begin()));
      tblgen_values.push_back((*tblgen_ConstantOp_1.getODSResults(0).begin()));
      tblgen_SubIOp_2 = rewriter.create<::mlir::arith::SubIOp>(odsLoc, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_SubIOp_2.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/Arith/IR/ArithCanonicalization.td:99
*/
struct SubISubILHSRHSLHS : public ::mlir::RewritePattern {
  SubISubILHSRHSLHS(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("arith.subi", 2, context, {"arith.constant", "arith.subi"}) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::mlir::Operation::operand_range x0(op0->getOperands());
    ::mlir::Operation::operand_range y(op0->getOperands());
    ::mlir::arith::SubIOp res;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::arith::SubIOp>(op0); (void)castedOp0;
    res = castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::arith::SubIOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::arith::SubIOp type";
        });
      }
      x0 = castedOp1.getODSOperands(0);
      y = castedOp1.getODSOperands(1);
      tblgen_ops.push_back(op1);
    }
    x = castedOp0.getODSOperands(1);
    if (!(*x.begin() == *x0.begin())){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "Operands 'x' and 'x0' must be equal";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = rewriter.getZeroAttr((*y.begin()).getType()); (void)nativeVar_0;
    ::mlir::arith::ConstantOp tblgen_ConstantOp_1;
    {
      tblgen_ConstantOp_1 = rewriter.create<::mlir::arith::ConstantOp>(odsLoc,
        /*value=*/nativeVar_0
      );
    }
    ::mlir::arith::SubIOp tblgen_SubIOp_2;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_ConstantOp_1.getODSResults(0).begin()));
      tblgen_values.push_back((*y.begin()));
      tblgen_SubIOp_2 = rewriter.create<::mlir::arith::SubIOp>(odsLoc, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_SubIOp_2.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    external/llvm-project/mlir/lib/Dialect/Arith/IR/ArithCanonicalization.td:112
*/
struct XOrINotCmpI : public ::mlir::RewritePattern {
  XOrINotCmpI(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("arith.xori", 3, context, {"arith.cmpi"}) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range b(op0->getOperands());
    ::mlir::Operation::operand_range a(op0->getOperands());
    ::mlir::arith::CmpIPredicateAttr pred;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::arith::XOrIOp>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::arith::CmpIOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::arith::CmpIOp type";
        });
      }
      {
        auto tblgen_attr = op1->getAttrOfType<::mlir::arith::CmpIPredicateAttr>("predicate");(void)tblgen_attr;
        if (!(tblgen_attr)){
          return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
            diag << "expected op 'arith.cmpi' to have attribute 'predicate' of type '::mlir::arith::CmpIPredicateAttr'";
          });
        }
        pred = tblgen_attr;
      }
      a = castedOp1.getODSOperands(0);
      b = castedOp1.getODSOperands(1);
      tblgen_ops.push_back(op1);
    }
    {
      auto *op1 = (*castedOp0.getODSOperands(1).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 1 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::arith::ConstantOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::arith::ConstantOp type";
        });
      }
      {
        auto tblgen_attr = op1->getAttrOfType<::mlir::TypedAttr>("value");(void)tblgen_attr;
        if (!(tblgen_attr)){
          return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
            diag << "expected op 'arith.constant' to have attribute 'value' of type '::mlir::TypedAttr'";
          });
        }
        if(::mlir::failed(__mlir_ods_local_attr_constraint_ArithCanonicalization1(rewriter, op1, tblgen_attr, "op 'arith.constant' attribute 'value' failed to satisfy constraint: 'constant attribute 1'"))) {
          return ::mlir::failure();
        }
      }
      tblgen_ops.push_back(op1);
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc(), tblgen_ops[2]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = invertPredicate(pred); (void)nativeVar_0;
    ::mlir::arith::CmpIOp tblgen_CmpIOp_1;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      if (auto tmpAttr = nativeVar_0) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("predicate"), tmpAttr);
      }
      tblgen_values.push_back((*a.begin()));
      tblgen_values.push_back((*b.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_CmpIOp_1 = rewriter.create<::mlir::arith::CmpIOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_CmpIOp_1.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

void LLVM_ATTRIBUTE_UNUSED populateWithGenerated(::mlir::RewritePatternSet &patterns) {
  patterns.add<AddIAddConstant>(patterns.getContext());
  patterns.add<AddISubConstantLHS>(patterns.getContext());
  patterns.add<AddISubConstantRHS>(patterns.getContext());
  patterns.add<AndOfExtSI>(patterns.getContext());
  patterns.add<AndOfExtUI>(patterns.getContext());
  patterns.add<BitcastOfBitcast>(patterns.getContext());
  patterns.add<CmpIExtSI>(patterns.getContext());
  patterns.add<CmpIExtUI>(patterns.getContext());
  patterns.add<DivFOfNegF>(patterns.getContext());
  patterns.add<ExtSIOfExtUI>(patterns.getContext());
  patterns.add<IndexCastOfExtSI>(patterns.getContext());
  patterns.add<IndexCastOfIndexCast>(patterns.getContext());
  patterns.add<IndexCastUIOfExtUI>(patterns.getContext());
  patterns.add<IndexCastUIOfIndexCastUI>(patterns.getContext());
  patterns.add<MulFOfNegF>(patterns.getContext());
  patterns.add<OrOfExtSI>(patterns.getContext());
  patterns.add<OrOfExtUI>(patterns.getContext());
  patterns.add<SubILHSAddConstant>(patterns.getContext());
  patterns.add<SubILHSSubConstantLHS>(patterns.getContext());
  patterns.add<SubILHSSubConstantRHS>(patterns.getContext());
  patterns.add<SubIRHSAddConstant>(patterns.getContext());
  patterns.add<SubIRHSSubConstantLHS>(patterns.getContext());
  patterns.add<SubIRHSSubConstantRHS>(patterns.getContext());
  patterns.add<SubISubILHSRHSLHS>(patterns.getContext());
  patterns.add<XOrINotCmpI>(patterns.getContext());
}
