#include "PatternTritonGPUOpToLLVM.h"
#include "TargetInfo.h"
#include "Utility.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/GenericSwizzling.h"
#include "triton/Tools/LayoutUtils.h"

namespace {

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
using mlir::LLVM::NVIDIA::lowerLdStMatrix;

struct ConvertLayoutOpSwizzlingConversion
    : public ConvertOpToLLVMPattern<triton::gpu::ConvertLayoutOp> {
  const NVIDIA::TargetInfo &targetInfo;
  mutable DenseMap<std::pair<Value, Attribute>, Value> convertedValues;

  explicit ConvertLayoutOpSwizzlingConversion(
      LLVMTypeConverter &typeConverter, const NVIDIA::TargetInfo &targetInfo,
      PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  LogicalResult
  matchAndRewrite(ConvertLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getType();

    LinearLayout conversion = minimalCvtLayout(srcTy, dstTy);
    LinearLayout srcLayout = toLinearLayout(srcTy);
    LinearLayout dstLayout = toLinearLayout(dstTy);

    assert(to_vector(conversion.getInDimNames()) ==
           to_vector(conversion.getOutDimNames()));
    if (!cvtAlwaysUseWarpShuffle(op) && cvtNeedsSharedMemory(srcTy, dstTy)) {
      if (succeeded(tryLowerPowerOfTwoIndexOpAfterTransfer(op, rewriter)))
        return success();

      auto loc = op.getLoc();

      auto llvmElemTy = getTypeConverter()->convertType(srcTy.getElementType());
      auto smemBase = LLVM::getSharedMemoryBase(loc, rewriter, targetInfo,
                                                op.getOperation());
      auto inVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
      auto outVals = transferWithinBlockSwizzling(
          loc, rewriter, srcLayout, dstLayout, inVals, llvmElemTy, smemBase);

      Value result =
          packLLElements(loc, getTypeConverter(), outVals, rewriter, dstTy);
      rewriter.replaceOp(op, result);
      return success();
    }
    return failure();
  }

  std::optional<APInt> getSplatInt(Value value) const {
    Attribute attr;
    if (!matchPattern(value, m_Constant(&attr)))
      return std::nullopt;
    if (auto intAttr = dyn_cast<IntegerAttr>(attr))
      return intAttr.getValue();
    auto denseAttr = dyn_cast<DenseIntElementsAttr>(attr);
    if (!denseAttr || !denseAttr.isSplat())
      return std::nullopt;
    return denseAttr.getSplatValue<APInt>();
  }

  Value getConvertedValue(ConvertLayoutOp op, Value src,
                          ConversionPatternRewriter &rewriter) const {
    auto srcTy = dyn_cast<RankedTensorType>(src.getType());
    auto dstTy = dyn_cast<RankedTensorType>(op.getType());
    if (!srcTy || !dstTy)
      return {};

    auto dstEncoding = dstTy.getEncoding();
    auto key = std::pair<Value, Attribute>(src, dstEncoding);
    if (auto it = convertedValues.find(key); it != convertedValues.end()) {
      Value converted = it->second;
      Operation *def = converted.getDefiningOp();
      if (def && def->getBlock() == op->getBlock() && def->isBeforeInBlock(op))
        return converted;
    }

    Value remapped = rewriter.getRemappedValue(src);
    if (!remapped)
      return {};

    auto loc = op.getLoc();
    auto convertedTy =
        cast<RankedTensorType>(srcTy.cloneWithEncoding(dstEncoding));
    auto llvmElemTy = getTypeConverter()->convertType(srcTy.getElementType());
    auto smemBase =
        LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op.getOperation());
    auto inVals = unpackLLElements(loc, remapped, rewriter);
    auto outVals = transferWithinBlockSwizzling(
        loc, rewriter, toLinearLayout(srcTy), toLinearLayout(convertedTy),
        inVals, llvmElemTy, smemBase);
    Value converted =
        packLLElements(loc, getTypeConverter(), outVals, rewriter, convertedTy);
    convertedValues[key] = converted;
    return converted;
  }

  LogicalResult tryLowerPowerOfTwoIndexOpAfterTransfer(
      ConvertLayoutOp op, ConversionPatternRewriter &rewriter) const {
    auto resultTy = dyn_cast<RankedTensorType>(op.getType());
    if (!resultTy || !resultTy.getElementType().isInteger(32))
      return failure();

    auto buildResult = [&](Value base, auto &&buildElem) {
      Value converted = getConvertedValue(op, base, rewriter);
      if (!converted)
        return failure();

      auto b = TritonLLVMOpBuilder(op.getLoc(), rewriter);
      SmallVector<Value> outVals;
      for (Value elem : unpackLLElements(op.getLoc(), converted, rewriter))
        outVals.push_back(buildElem(b, elem));
      Value result = packLLElements(op.getLoc(), getTypeConverter(), outVals,
                                    rewriter, op.getType());
      rewriter.replaceOp(op, result);
      return success();
    };

    if (auto div = op.getSrc().getDefiningOp<arith::DivUIOp>()) {
      auto divisor = getSplatInt(div.getRhs());
      if (!divisor || !divisor->isPowerOf2())
        return failure();
      unsigned shift = divisor->exactLogBase2();
      return buildResult(div.getLhs(),
                         [shift](TritonLLVMOpBuilder &b, Value elem) {
                           return b.lshr(elem, b.i32_val(shift));
                         });
    }

    auto shl = op.getSrc().getDefiningOp<arith::ShLIOp>();
    auto rem =
        shl ? shl.getRhs().getDefiningOp<arith::RemUIOp>() : arith::RemUIOp();
    auto one = shl ? getSplatInt(shl.getLhs()) : std::nullopt;
    auto divisor = rem ? getSplatInt(rem.getRhs()) : std::nullopt;
    if (!rem || !one || !one->isOne() || !divisor || !divisor->isPowerOf2())
      return failure();

    uint64_t mask = divisor->getZExtValue() - 1;
    return buildResult(rem.getLhs(),
                       [mask](TritonLLVMOpBuilder &b, Value elem) {
                         Value shift = b.and_(elem, b.i32_val(mask));
                         return b.shl(b.i32_val(1), shift);
                       });
  }

  SmallVector<Value> transferWithinBlockSwizzling(
      Location loc, ConversionPatternRewriter &rewriter,
      const LinearLayout &srcLayout, const LinearLayout &dstLayout,
      ArrayRef<Value> inVals, Type llvmElemTy, Value smemBase) const {
    auto *ctx = rewriter.getContext();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    // We handle transformations recursively as they all need a preprocessing
    // and a postprocessing step.

    // Handle pointer types as 64-bit integers
    if (isa<LLVM::LLVMPointerType>(llvmElemTy)) {
      auto llvmElemTyPtr = i64_ty;
      auto newInVals = llvm::to_vector(llvm::map_range(inVals, [&](Value v) {
        return b.ptrtoint(llvmElemTyPtr, v).getResult();
      }));
      auto outVals =
          transferWithinBlockSwizzling(loc, rewriter, srcLayout, dstLayout,
                                       newInVals, llvmElemTyPtr, smemBase);
      for (auto &v : outVals) {
        v = b.inttoptr(llvmElemTy, v);
      }
      return outVals;
    }

    // Handle sub-byte elements like i1
    if (llvmElemTy.getIntOrFloatBitWidth() < 8) {
      // Upcast to i8
      auto i8ElemTy = i8_ty;
      auto newInVals = llvm::to_vector(llvm::map_range(
          inVals, [&](Value v) { return b.zext(i8ElemTy, v).getResult(); }));
      auto outVals = transferWithinBlockSwizzling(
          loc, rewriter, srcLayout, dstLayout, newInVals, i8ElemTy, smemBase);
      for (auto &v : outVals) {
        v = b.trunc(llvmElemTy, v);
      }
      return outVals;
    }

    // Remove broadcasting in src
    auto removeBroadcastSrc = actionRemoveBroadcastedRegs(srcLayout);
    if (!removeBroadcastSrc.isIdentity()) {
      auto prmtSrc = removeBroadcastSrc.apply(srcLayout);
      auto newInVals = removeBroadcastSrc.apply(inVals);
      return transferWithinBlockSwizzling(loc, rewriter, prmtSrc, dstLayout,
                                          newInVals, llvmElemTy, smemBase);
    }

    // Remove broadcasting in dst
    auto removeBroadcastDst = actionRemoveBroadcastedRegs(dstLayout);
    if (!removeBroadcastDst.isIdentity()) {
      auto prmtDst = removeBroadcastDst.apply(dstLayout);
      auto outVals = transferWithinBlockSwizzling(
          loc, rewriter, srcLayout, prmtDst, inVals, llvmElemTy, smemBase);
      return broadcastAs(outVals, dstLayout);
    }

    // At this point we have a type that's at least 8-bit
    // and we don't have broadcasting in the registers
    auto bitwidth = llvmElemTy.getIntOrFloatBitWidth();
    auto kBlock = str_attr("block");
    bool crossCTA =
        !dstLayout.invertAndCompose(srcLayout).isTrivialOver({kBlock});
    auto [srcTiles, dstTiles] = getSrcDstTiles(targetInfo, bitwidth, crossCTA);
    auto [smem, instr] =
        optimalSwizzling(srcLayout, dstLayout, srcTiles, dstTiles, bitwidth);
    auto [idxSrc, idxDst] = instr;

    // Extract reps from smem
    auto kReg = str_attr("register");
    auto kLane = str_attr("lane");
    auto kWarp = str_attr("warp");
    auto kReps = str_attr("reps");
    auto nReps = smem.getInDimSize(kReps);
    auto reps = LinearLayout::identity1D(nReps, kReg, kReps);

    auto totalStoreCvt = srcLayout.invertAndCompose(smem);
    auto totalLoadCvt = dstLayout.invertAndCompose(smem);

    // The permutation exists by construction of the reps dimension in
    // optimalSwizzling
    auto permStore =
        regPermForDivide(totalStoreCvt, reps, /*left=*/false).value();
    totalStoreCvt = permStore.apply(totalStoreCvt);
    auto permutedInVals = permStore.apply(inVals);
    auto permLoad =
        regPermForDivide(totalLoadCvt, reps, /*left=*/false).value();
    totalLoadCvt = permLoad.apply(totalLoadCvt);

    // Remove the reps and flatten into offset
    auto storeCvt = *divideRight(totalStoreCvt, reps);
    auto loadCvt = *divideRight(totalLoadCvt, reps);
    auto kOffset = str_attr("offset");
    auto nBlock = storeCvt.getInDimSize(kBlock);
    storeCvt = storeCvt.reshapeOuts(
        {{kOffset, storeCvt.getTotalOutDimSize() / nBlock}, {kBlock, nBlock}});
    loadCvt = loadCvt.reshapeOuts(
        {{kOffset, loadCvt.getTotalOutDimSize() / nBlock}, {kBlock, nBlock}});

    // We never do cross-CTA writes by construction. We may do cross-CTA reads,
    // but in that case we lower to ld.shared/st.shared
    assert(storeCvt.isTrivialOver({kBlock}));
    assert(loadCvt.isTrivialOver({kBlock}) || idxDst == 0);

    auto tileSize = storeCvt.getInDimSize(kReg);

    assert(permutedInVals.size() == tileSize * nReps);
    SmallVector<Value> outVals;
    auto affineOffset = b.i32_val(0);
    auto maskSpanAffineOffset = 0;
    bool isWarpSync = mlir::isCvtDimSync(srcLayout, dstLayout, kWarp);
    bool isBlockSync = mlir::isCvtDimSync(srcLayout, dstLayout, kBlock);
    auto emitBarrier = [&]() {
      if (isWarpSync) {
        targetInfo.warpSync(loc, rewriter);
      } else if (isBlockSync) {
        targetInfo.barrier(loc, rewriter, triton::gpu::AddrSpace::Local);
      } else {
        targetInfo.clusterBarrier(loc, rewriter);
      }
    };
    auto dropBlock = [&](const LinearLayout &cvt) {
      SmallVector<StringAttr> inDims = {kReg, kLane, kWarp};
      SmallVector<StringAttr> outDims = {kOffset};
      return cvt.sublayout(inDims, outDims);
    };
    for (int i = 0; i < nReps; ++i) {
      if (i > 0)
        emitBarrier();
      auto tileInVals =
          to_vector(ArrayRef(permutedInVals).slice(i * tileSize, tileSize));
      // Store
      // idxSrc 0: st.shared, idxSrc 1: stmatrix, idxSrc 2: stmatrix.trans
      if (idxSrc == 0) {
        lowerLdStShared(loc, ctx, storeCvt, tileInVals, llvmElemTy, smemBase,
                        /*paddingShifts=*/{}, affineOffset,
                        maskSpanAffineOffset, rewriter, targetInfo);
      } else {
        assert(idxSrc == 1 || idxSrc == 2);
        bool transpose = idxSrc == 2;
        auto storeCvtNoBlock = dropBlock(storeCvt);
        auto result = lowerLdStMatrix(
            loc, storeCvtNoBlock, transpose, tileInVals, smemBase, affineOffset,
            maskSpanAffineOffset, llvmElemTy, rewriter, targetInfo);
        assert(succeeded(result));
      }
      emitBarrier();
      // Load
      SmallVector<Value> tileOutVals;
      // idxDst 0: ld.shared, idxDst 1: ldmatrix, idxDst 2: ldmatrix.trans
      if (idxDst == 0) {
        tileOutVals = lowerLdStShared(
            loc, ctx, loadCvt, {}, llvmElemTy, smemBase, /*paddingShifts=*/{},
            affineOffset, maskSpanAffineOffset, rewriter, targetInfo);
      } else {
        assert(idxDst == 1 || idxDst == 2);
        bool transpose = idxDst == 2;
        auto loadCvtNoBlock = dropBlock(loadCvt);
        auto result = lowerLdStMatrix(
            loc, loadCvtNoBlock, transpose, tileOutVals, smemBase, affineOffset,
            maskSpanAffineOffset, llvmElemTy, rewriter, targetInfo);
        assert(succeeded(result));
      }
      llvm::append_range(outVals, tileOutVals);
    }

    // Undo the permLoad used to divideRight
    outVals = permLoad.inverse().apply(outVals);
    return outVals;
  }
};

} // namespace

void mlir::triton::NVIDIA::populateConvertLayoutOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  // Give this convertLayoutOpSwizzlingConversion a higher benefit as it
  // matches optimized ldmatrix/stmatrix cases
  patterns.add<ConvertLayoutOpSwizzlingConversion>(typeConverter, targetInfo,
                                                   benefit.getBenefit() + 1);
  mlir::triton::populateConvertLayoutOpToLLVMPatterns(typeConverter, targetInfo,
                                                      patterns, benefit);
}
