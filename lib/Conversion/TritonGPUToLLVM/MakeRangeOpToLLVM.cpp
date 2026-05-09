#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LayoutUtils.h"

namespace {

using namespace mlir;
using namespace mlir::triton;
namespace ttg = mlir::triton::gpu;

struct MakeRangeOpConversion
    : public ConvertOpToLLVMPattern<triton::MakeRangeOp> {
  MakeRangeOpConversion(LLVMTypeConverter &converter,
                        const TargetInfoBase &targetInfo,
                        PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::MakeRangeOp>(converter, benefit),
        targetInfo(targetInfo) {}
  LogicalResult
  matchAndRewrite(triton::MakeRangeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    RankedTensorType ty = op.getType();
    auto elemTy = ty.getElementType();
    assert(elemTy.isInteger(32));
    Value start = createIndexAttrConstant(rewriter, loc, elemTy, op.getStart());
    MLIRContext *ctx = rewriter.getContext();
    LinearLayout ll = ttg::toLinearLayout(ty);
    auto idxs = [&]() -> SmallVector<SmallVector<Value>> {
      StringAttr kRegister = str_attr("register");
      StringAttr kLane = str_attr("lane");
      StringAttr kWarp = str_attr("warp");
      StringAttr kBlock = str_attr("block");

      auto isPow2Sequence = [](auto bases, uint32_t first) {
        for (auto [i, basis] : llvm::enumerate(bases)) {
          if (basis.size() != 1 || basis[0] != (first << i))
            return false;
        }
        return true;
      };
      auto isZeroSequence = [](auto bases) {
        return llvm::all_of(bases, [](auto basis) {
          return basis.size() == 1 && basis[0] == 0;
        });
      };

      // For contiguous rank-1 layouts, the register-0 make_range value is
      // exactly the CTA-local thread ID. Keep the generic path for layouts
      // where lane/warp/block bases do anything more interesting.
      if (ty.getRank() != 1 ||
          getWarpGroupStartThreadId(rewriter.getInsertionBlock()))
        return emitIndices(loc, rewriter, targetInfo, ll, ty, true);
      auto laneBases = ll.getBases().lookup(kLane);
      auto warpBases = ll.getBases().lookup(kWarp);
      auto blockBases = ll.getBases().lookup(kBlock);
      auto regBases = ll.getBases().lookup(kRegister);
      if (!isPow2Sequence(laneBases, 1) || !isZeroSequence(blockBases))
        return emitIndices(loc, rewriter, targetInfo, ll, ty, true);

      uint32_t threadsPerWarp = 1u << laneBases.size();
      if (!isPow2Sequence(warpBases, threadsPerWarp))
        return emitIndices(loc, rewriter, targetInfo, ll, ty, true);
      uint32_t threadsPerCTA = threadsPerWarp << warpBases.size();
      if (!isPow2Sequence(regBases, threadsPerCTA))
        return emitIndices(loc, rewriter, targetInfo, ll, ty, true);

      Operation *lookupPt = &rewriter.getInsertionBlock()->front();
      if (threadsPerWarp != ttg::lookupThreadsPerWarp(rewriter) ||
          threadsPerCTA != threadsPerWarp * ttg::lookupNumWarps(lookupPt))
        return emitIndices(loc, rewriter, targetInfo, ll, ty, true);

      Value tid = ::mlir::gpu::ThreadIdOp::create(rewriter, loc,
                                                  ::mlir::gpu::Dimension::x);
      tid = arith::IndexCastOp::create(rewriter, loc, i32_ty, tid);
      SmallVector<SmallVector<Value>> contiguousIdxs;
      for (uint32_t reg = 0; reg < ll.getInDimSize(kRegister); ++reg) {
        Value idx = reg == 0 ? tid : b.add(tid, b.i32_val(reg * threadsPerCTA));
        contiguousIdxs.push_back({idx});
      }
      return contiguousIdxs;
    }();
    unsigned elems = idxs.size();
    SmallVector<Value> retVals(elems);
    // TODO: slice layout has more elements than expected.
    // Unexpected behavior for make range, but generally OK when followed by
    // expand dims + broadcast. very weird behavior otherwise potentially.
    for (const auto &multiDim : llvm::enumerate(idxs)) {
      assert(multiDim.value().size() == 1);
      retVals[multiDim.index()] = b.add(multiDim.value()[0], start);
    }
    auto typeConverter = getTypeConverter();
    Value result = packLLElements(loc, typeConverter, retVals, rewriter, ty);
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

// Convert arith::ConstantOp with an array DenseElementsAttr to a
// LLVM::StructType value.
class ArithConstantArrayOpConversion
    : public ConvertOpToLLVMPattern<arith::ConstantOp> {
public:
  ArithConstantArrayOpConversion(LLVMTypeConverter &typeConverter,
                                 const TargetInfoBase &targetInfo,
                                 PatternBenefit benefit)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  LogicalResult matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {
    auto value = op.getValue();
    auto values = dyn_cast<DenseElementsAttr>(value);
    if (!values || isa<SplatElementsAttr>(value))
      return failure();
    auto tensorTy = cast<RankedTensorType>(op.getType());
    if (!tensorTy.getElementType().isIntOrFloat())
      return failure();

    MLIRContext *ctx = op->getContext();
    Location loc = op->getLoc();
    TritonLLVMOpBuilder tb(loc, b);
    auto kBlock = str_attr("block");
    auto kWarp = str_attr("warp");
    auto kLane = str_attr("lane");
    auto kRegister = str_attr("register");

    SmallVector<Attribute> attrs = to_vector(values.getValues<Attribute>());
    auto type =
        LLVM::LLVMArrayType::get(tensorTy.getElementType(), attrs.size());
    auto module = op->getParentOfType<ModuleOp>();
    LLVM::GlobalOp global = LLVM::getOrInsertGlobalConstant(
        b, module, type, b.getArrayAttr(attrs), "tensor_constant_");

    LinearLayout ll = ttg::toLinearLayout(tensorTy);
    auto [laneId, warpId] = getLaneAndWarpId(b, loc);
    Value blockId = targetInfo.getClusterCTAId(b, loc);
    SmallVector<Value> llValues;
    for (unsigned reg : llvm::seq(ll.getInDimSize(kRegister))) {
      auto indices = applyLinearLayout(loc, b, ll,
                                       {{kRegister, tb.i32_val(reg)},
                                        {kLane, laneId},
                                        {kWarp, warpId},
                                        {kBlock, blockId}});
      Value index =
          LLVM::linearize(b, loc, to_vector(make_second_range(indices)),
                          convertType<unsigned>(tensorTy.getShape()));
      Value addr = tb.address_of(global);
      addr = tb.gep(ptr_ty(ctx), tensorTy.getElementType(), addr, index);
      llValues.push_back(tb.load(tensorTy.getElementType(), addr));
    }

    Value result =
        packLLElements(loc, getTypeConverter(), llValues, b, tensorTy);
    b.replaceOp(op, result);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

} // namespace

void mlir::triton::populateMakeRangeOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, const TargetInfoBase &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<ArithConstantArrayOpConversion>(typeConverter, targetInfo,
                                               benefit);
  patterns.add<MakeRangeOpConversion>(typeConverter, targetInfo, benefit);
}
