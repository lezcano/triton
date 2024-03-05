#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

using namespace mlir;
namespace tt = mlir::triton;

namespace {

// Implement 3xTF32 trick https://github.com/NVIDIA/cutlass/discussions/385
// For a, b fp32
// dot(a, b, f32Backend="3xtf32") ->
//  let aBig = f32ToTF32(a), aSmall = a - aBig;
//  let bBig = f32ToTF32(b), bSmall = b - bBig;
//  dot(aSmall, bBig, f32Backend="tf32") +
//  dot(aBig, bSmall, f32Backend="tf32") +
//  dot(aBig, bBig, f32Backend="tf32")
class TF32x3 : public OpRewritePattern<tt::DotOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult
  matchAndRewrite(tt::DotOp dotOp,
                  mlir::PatternRewriter &rewriter) const override {

    auto isF32 = [](mlir::Value operand) {
      return operand.getType()
          .cast<RankedTensorType>()
          .getElementType()
          .isF32();
    };

    if (!(dotOp.getF32Backend() == "3xtf32" && isF32(dotOp.getOperand(0)) &&
          isF32(dotOp.getOperand(1)))) {
      return mlir::failure();
    }

    // Aux functions
    auto f32ToTF32 = [&](mlir::Value value) -> mlir::Value {
      return rewriter
          .create<tt::ElementwiseInlineAsmOp>(
              dotOp.getLoc(), value.getType(), "cvt.rna.tf32.f32 $0, $1;",
              "=r,r",
              /*isPure*/ true, /*pack*/ 1, std::vector<mlir::Value>{value})
          .getResult()[0];
    };
    auto sub = [&](mlir::Value a, mlir::Value b) -> mlir::Value {
      return rewriter.create<mlir::arith::SubFOp>(dotOp.getLoc(), a, b);
    };
    auto dot = [&](mlir::Value a, mlir::Value b, mlir::Value c) -> mlir::Value {
      return rewriter.create<tt::DotOp>(dotOp->getLoc(), c.getType(), a, b, c,
                                        "tf32", dotOp.getMaxNumImpreciseAcc());
    };
    auto a = dotOp.getOperand(0);
    auto b = dotOp.getOperand(1);

    auto aBig = f32ToTF32(a);
    auto aSmall = sub(a, aBig);

    auto bBig = f32ToTF32(b);
    auto bSmall = sub(b, bBig);

    auto dot1 = dot(aSmall, bBig, dotOp.getOperand(2));
    auto dot2 = dot(aBig, bSmall, dot1);
    auto dot3 = dot(aBig, bBig, dot2);

    rewriter.replaceOp(dotOp, dot3);
    return mlir::success();
  }
};

} // anonymous namespace

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

struct FP32DotTCPass : public TritonGPUFP32DotTCBase<FP32DotTCPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    mlir::RewritePatternSet decomposePatterns(context);
    decomposePatterns.add<TF32x3>(context);
    if (mlir::applyPatternsAndFoldGreedily(m, std::move(decomposePatterns))
            .failed()) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<Pass> mlir::triton::gpu::createFP32DotTCPass() {
  return std::make_unique<FP32DotTCPass>();
}
