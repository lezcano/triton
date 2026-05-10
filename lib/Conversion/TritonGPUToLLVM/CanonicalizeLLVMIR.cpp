#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/MathExtras.h"

#include <functional>

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

// Reuse an already-normalized masked slice instead of extracting the same bits
// from the original mask again.  For a shifted contiguous mask whose first set
// bit is k, `(x & mask) >> s == ((x & mask) >> k) << (k - s)` when s < k.
class ReuseNormalizedMaskedSlicePattern
    : public OpRewritePattern<LLVM::LShrOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::LShrOp op,
                                PatternRewriter &b) const override {
    auto masked = op.getLhs().getDefiningOp<LLVM::AndOp>();
    if (!masked)
      return failure();

    APInt mask;
    APInt shift;
    if (!matchPattern(masked.getRhs(), m_ConstantInt(&mask)) ||
        !matchPattern(op.getRhs(), m_ConstantInt(&shift)) || !mask.isIntN(32) ||
        !shift.isIntN(32))
      return failure();

    uint32_t maskValue = mask.getZExtValue();
    uint32_t shiftValue = shift.getZExtValue();
    if (!llvm::isShiftedMask_32(maskValue))
      return failure();

    uint32_t normalizedShift = llvm::countr_zero(maskValue);
    if (shiftValue >= normalizedShift)
      return failure();

    // This pass runs before CSE, so repeated layout applications can spell the
    // same input through separate, but equivalent, side-effect-free trees.
    std::function<bool(Value, Value)> equivalent = [&](Value a, Value b) {
      if (a == b)
        return true;
      if (a.getType() != b.getType() || isa<BlockArgument>(a) ||
          isa<BlockArgument>(b))
        return false;
      Operation *aDef = a.getDefiningOp();
      Operation *bDef = b.getDefiningOp();
      if (cast<OpResult>(a).getResultNumber() !=
              cast<OpResult>(b).getResultNumber() ||
          !isMemoryEffectFree(aDef) || !isMemoryEffectFree(bDef) ||
          aDef->getNumRegions() || bDef->getNumRegions())
        return false;
      return OperationEquivalence::isEquivalentTo(
          aDef, bDef,
          [&](Value a, Value b) { return success(equivalent(a, b)); },
          /*markEquivalent=*/nullptr, OperationEquivalence::IgnoreLocations);
    };

    auto hasSameMask = [&](LLVM::AndOp candidate) {
      APInt candidateMask;
      return equivalent(candidate.getLhs(), masked.getLhs()) &&
             matchPattern(candidate.getRhs(), m_ConstantInt(&candidateMask)) &&
             candidateMask == mask;
    };

    LLVM::LShrOp normalized;
    for (Operation &candidateOp : *op->getBlock()) {
      if (&candidateOp == op)
        break;
      auto candidate = dyn_cast<LLVM::LShrOp>(&candidateOp);
      if (!candidate)
        continue;
      auto candidateMasked = candidate.getLhs().getDefiningOp<LLVM::AndOp>();
      APInt candidateShift;
      if (candidateMasked && hasSameMask(candidateMasked) &&
          matchPattern(candidate.getRhs(), m_ConstantInt(&candidateShift)) &&
          candidateShift.isIntN(32) &&
          candidateShift.getZExtValue() == normalizedShift) {
        normalized = candidate;
        break;
      }
    }
    if (!normalized)
      return failure();

    Value delta = LLVM::ConstantOp::create(
        b, op.getLoc(), op.getType(),
        b.getIntegerAttr(op.getType(), normalizedShift - shiftValue));
    b.replaceOpWithNewOp<LLVM::ShlOp>(op, normalized, delta);
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
    patterns.add<SelectConstantConditionPattern,
                 ReuseNormalizedMaskedSlicePattern, ThreeInputOrPattern>(
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
