#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir::triton::gpu {
#define GEN_PASS_DEF_CANONICALIZELLVMIR
#include "triton/Conversion/TritonGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton::gpu

namespace {
class SelectConstantConditionPattern : public OpRewritePattern<LLVM::SelectOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::SelectOp op,
                                PatternRewriter &b) const override {
    BoolAttr cond;
    if (!matchPattern(op.getCondition(), m_Constant(&cond)))
      return failure();
    Value val = cond.getValue() ? op.getTrueValue() : op.getFalseValue();
    b.replaceOp(op, ValueRange{val});
    return success();
  }
};

// PTXAS leaves a disjoint three-input integer OR as two instructions, while
// lop3 can represent the same truth table in one instruction.  Keep constants
// out of this form so they can still lower as immediates.
class ThreeInputOrPattern : public OpRewritePattern<LLVM::OrOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::OrOp op,
                                PatternRewriter &b) const override {
    if (!op.getType().isInteger(32))
      return failure();

    auto matchInnerOr = [](Value value) -> LLVM::OrOp {
      auto inner = value.getDefiningOp<LLVM::OrOp>();
      return inner && inner->hasOneUse() ? inner : LLVM::OrOp();
    };

    LLVM::OrOp inner = matchInnerOr(op.getLhs());
    Value third = op.getRhs();
    if (!inner) {
      inner = matchInnerOr(op.getRhs());
      third = op.getLhs();
    }
    if (!inner || !op->hasAttr("isDisjoint") || !inner->hasAttr("isDisjoint"))
      return failure();

    Value first = inner.getLhs();
    Value second = inner.getRhs();
    if (first == second || first == third || second == third)
      return failure();

    APInt unused;
    if (matchPattern(first, m_ConstantInt(&unused)) ||
        matchPattern(second, m_ConstantInt(&unused)) ||
        matchPattern(third, m_ConstantInt(&unused)))
      return failure();

    auto lop3 = LLVM::InlineAsmOp::create(
        b, op.getLoc(), op.getType(), ValueRange{first, second, third},
        /*asm_string=*/"lop3.b32 $0, $1, $2, $3, 0xfe;",
        /*constraints=*/"=r,r,r,r",
        /*has_side_effects=*/false, /*is_align_stack=*/false,
        LLVM::TailCallKind::None,
        LLVM::AsmDialectAttr::get(b.getContext(), LLVM::AsmDialect::AD_ATT),
        /*operand_attrs=*/ArrayAttr());
    b.replaceOp(op, lop3.getRes());
    return success();
  }
};
} // namespace

namespace {
struct CanonicalizeLLVMIR
    : public mlir::triton::gpu::impl::CanonicalizeLLVMIRBase<
          CanonicalizeLLVMIR> {
  void runOnOperation() override {
    LLVM::LLVMFuncOp func = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<SelectConstantConditionPattern, ThreeInputOrPattern>(
        &getContext());

    getContext()
        .getLoadedDialect<LLVM::LLVMDialect>()
        ->getCanonicalizationPatterns(patterns);
    for (mlir::RegisteredOperationName op :
         getContext().getRegisteredOperationsByDialect(
             LLVM::LLVMDialect::getDialectNamespace()))
      op.getCanonicalizationPatterns(patterns, &getContext());

    (void)applyPatternsGreedily(func, std::move(patterns));
  }
};
} // namespace
