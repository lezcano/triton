#include "Dialect/NVGPU/IR/Dialect.h"
#include "TritonNVIDIAGPUToLLVM/Passes.h"
#include "TritonNVIDIAGPUToLLVM/Utility.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Membar.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Gluon/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/Transforms/ConSanTargetHooks.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/ClusterBarrierInsertion.h"

#include "Allocation.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"

namespace ttng = mlir::triton::nvidia_gpu;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_CONVERTTRITONGPUTOLLVM
#include "TritonNVIDIAGPUToLLVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton::NVIDIA;

namespace {

class TritonLLVMFunctionConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMFunctionConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalDialect<NVVM::NVVMDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

class TritonLLVMConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalDialect<NVVM::NVVMDialect>();
    addLegalDialect<cf::ControlFlowDialect>();
    addLegalDialect<mlir::triton::nvgpu::NVGPUDialect>();
    addIllegalDialect<triton::TritonDialect>();
    addDynamicallyLegalDialect<triton::gpu::TritonGPUDialect>(
        [](mlir::Operation *op) {
          // We handle the warp ID op during NVGPUToLLVM.
          return isa<triton::gpu::WarpIdOp>(op);
        });
    addIllegalDialect<triton::nvidia_gpu::TritonNvidiaGPUDialect>();
    addIllegalDialect<mlir::gpu::GPUDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();

    // Warp specialization is lowered later.
    addLegalOp<triton::gpu::WarpSpecializeOp>();
    addLegalOp<triton::gpu::WarpYieldOp>();
    addLegalOp<triton::gpu::WarpSpecializePartitionsOp>();
    addLegalOp<triton::gpu::WarpReturnOp>();
  }
};

static bool isConstantInt(Value value, int64_t expected) {
  APInt constant;
  return matchPattern(value, m_ConstantInt(&constant)) && constant == expected;
}

// Match:
//   incr = index + 1
//   rollover = incr == 2
// and return `index`.
static Value getModuloTwoCounterInput(Value condition) {
  auto cmp = condition.getDefiningOp<arith::CmpIOp>();
  if (!cmp || cmp.getPredicate() != arith::CmpIPredicate::eq)
    return {};

  Value incr;
  if (isConstantInt(cmp.getLhs(), 2))
    incr = cmp.getRhs();
  else if (isConstantInt(cmp.getRhs(), 2))
    incr = cmp.getLhs();
  else
    return {};

  auto add = incr.getDefiningOp<arith::AddIOp>();
  if (!add)
    return {};
  if (isConstantInt(add.getLhs(), 1))
    return add.getRhs();
  if (isConstantInt(add.getRhs(), 1))
    return add.getLhs();
  return {};
}

static bool isModuloTwoCounterUpdate(arith::SelectOp op, Value &input) {
  input = getModuloTwoCounterInput(op.getCondition());
  if (!input || !isConstantInt(op.getTrueValue(), 0))
    return false;

  auto add = op.getFalseValue().getDefiningOp<arith::AddIOp>();
  if (!add)
    return false;
  return (add.getLhs() == input && isConstantInt(add.getRhs(), 1)) ||
         (add.getRhs() == input && isConstantInt(add.getLhs(), 1));
}

static SmallVector<Value> getIncomingValues(BlockArgument arg) {
  SmallVector<Value> incoming;
  Block *block = arg.getOwner();
  unsigned argNo = arg.getArgNumber();
  for (Block *pred : block->getPredecessors()) {
    Operation *terminator = pred->getTerminator();
    if (auto br = dyn_cast<cf::BranchOp>(terminator)) {
      if (br.getDest() == block && argNo < br.getDestOperands().size())
        incoming.push_back(br.getDestOperands()[argNo]);
      continue;
    }
    if (auto condBr = dyn_cast<cf::CondBranchOp>(terminator)) {
      if (condBr.getTrueDest() == block &&
          argNo < condBr.getTrueDestOperands().size())
        incoming.push_back(condBr.getTrueDestOperands()[argNo]);
      if (condBr.getFalseDest() == block &&
          argNo < condBr.getFalseDestOperands().size())
        incoming.push_back(condBr.getFalseDestOperands()[argNo]);
    }
  }
  return incoming;
}

static bool collectZeroOrOneDependencies(Value value,
                                         SmallVectorImpl<Value> &deps) {
  if (isConstantInt(value, 0) || isConstantInt(value, 1))
    return true;

  if (auto xorOp = value.getDefiningOp<arith::XOrIOp>()) {
    if (isConstantInt(xorOp.getLhs(), 1))
      return collectZeroOrOneDependencies(xorOp.getRhs(), deps);
    if (isConstantInt(xorOp.getRhs(), 1))
      return collectZeroOrOneDependencies(xorOp.getLhs(), deps);
    return false;
  }

  if (auto select = value.getDefiningOp<arith::SelectOp>()) {
    Value input;
    if (isModuloTwoCounterUpdate(select, input))
      return collectZeroOrOneDependencies(input, deps);
    return collectZeroOrOneDependencies(select.getTrueValue(), deps) &&
           collectZeroOrOneDependencies(select.getFalseValue(), deps);
  }

  auto arg = dyn_cast<BlockArgument>(value);
  if (!arg)
    return false;
  if (!llvm::is_contained(deps, value))
    deps.push_back(value);
  return true;
}

static bool isKnownZeroOrOne(Value value) {
  SmallVector<Value> roots;
  if (!collectZeroOrOneDependencies(value, roots))
    return false;
  if (roots.empty())
    return true;

  DenseSet<Value> inGroup;
  SmallVector<Value> group;
  auto addToGroup = [&](Value dependency) {
    if (inGroup.insert(dependency).second)
      group.push_back(dependency);
  };
  for (Value root : roots)
    addToGroup(root);

  DenseMap<Value, SmallVector<SmallVector<Value>>> incomingDeps;
  for (size_t i = 0; i < group.size(); ++i) {
    auto arg = cast<BlockArgument>(group[i]);
    SmallVector<Value> incoming = getIncomingValues(arg);
    if (incoming.empty())
      return false;

    auto &depsByEdge = incomingDeps[arg];
    for (Value incomingValue : incoming) {
      SmallVector<Value> deps;
      if (!collectZeroOrOneDependencies(incomingValue, deps))
        return false;
      for (Value dep : deps)
        addToGroup(dep);
      depsByEdge.push_back(std::move(deps));
    }
  }

  // Every one-bit recurrence group must be reachable from a non-recursive
  // base.  This proves mutually recursive block arguments without accepting an
  // uninitialized cycle.
  DenseSet<Value> reached;
  bool changed = true;
  while (changed) {
    changed = false;
    for (Value arg : group) {
      if (reached.contains(arg))
        continue;
      bool hasReachedIncoming =
          llvm::any_of(incomingDeps[arg], [&](ArrayRef<Value> deps) {
            return llvm::all_of(
                deps, [&](Value dep) { return reached.contains(dep); });
          });
      if (hasReachedIncoming) {
        reached.insert(arg);
        changed = true;
      }
    }
  }
  return llvm::all_of(group, [&](Value arg) { return reached.contains(arg); });
}

class FoldModuloTwoCounterSelect : public OpRewritePattern<arith::SelectOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::SelectOp op,
                                PatternRewriter &rewriter) const override {
    Value input;
    if (!isModuloTwoCounterUpdate(op, input) || !isKnownZeroOrOne(input))
      return failure();
    Value one = arith::ConstantIntOp::create(rewriter, op.getLoc(), 1, 32);
    rewriter.replaceOpWithNewOp<arith::XOrIOp>(op, input, one);
    return success();
  }
};

class FoldModuloTwoPhaseSelect : public OpRewritePattern<arith::SelectOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::SelectOp op,
                                PatternRewriter &rewriter) const override {
    Value input = getModuloTwoCounterInput(op.getCondition());
    if (!input || !isKnownZeroOrOne(input))
      return failure();

    auto xorOp = op.getTrueValue().getDefiningOp<arith::XOrIOp>();
    if (!xorOp)
      return failure();
    Value phase = op.getFalseValue();
    if (!((xorOp.getLhs() == phase && isConstantInt(xorOp.getRhs(), 1)) ||
          (xorOp.getRhs() == phase && isConstantInt(xorOp.getLhs(), 1))))
      return failure();

    Operation *insertBefore = nullptr;
    for (Operation *user : op.getCondition().getUsers()) {
      auto counterUpdate = dyn_cast<arith::SelectOp>(user);
      Value counterInput;
      if (!counterUpdate || counterUpdate == op ||
          !isModuloTwoCounterUpdate(counterUpdate, counterInput) ||
          counterInput != input)
        continue;
      if (counterUpdate->getBlock() == op->getBlock() &&
          counterUpdate->isBeforeInBlock(op)) {
        insertBefore = counterUpdate;
        break;
      }
    }

    OpBuilder::InsertionGuard guard(rewriter);
    if (insertBefore)
      rewriter.setInsertionPoint(insertBefore);
    Value replacement =
        arith::XOrIOp::create(rewriter, op.getLoc(), phase, input);
    rewriter.replaceOp(op, replacement);
    return success();
  }
};

struct ConvertTritonGPUToLLVM
    : public triton::impl::ConvertTritonGPUToLLVMBase<ConvertTritonGPUToLLVM> {
  using ConvertTritonGPUToLLVMBase::ConvertTritonGPUToLLVMBase;

  ConvertTritonGPUToLLVM(int32_t computeCapability)
      : ConvertTritonGPUToLLVMBase({computeCapability}) {}
  ConvertTritonGPUToLLVM(int32_t computeCapability, int32_t ptxVersion)
      : ConvertTritonGPUToLLVMBase({computeCapability, ptxVersion}) {}
  ConvertTritonGPUToLLVM(int32_t computeCapability, int32_t ptxVersion,
                         bool enableConcurrencySanitizer)
      : ConvertTritonGPUToLLVMBase(
            {computeCapability, ptxVersion, enableConcurrencySanitizer}) {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();
    TargetInfo targetInfo(computeCapability, ptxVersion);

    // Allocate shared memory and set barrier
    ModuleAllocation allocation(
        mod, mlir::triton::nvidia_gpu::getNvidiaAllocationAnalysisScratchSizeFn(
                 targetInfo));
    mlir::triton::nvidia_gpu::runClusterBarrierInsertion(allocation,
                                                         computeCapability);
    if (failed(mlir::triton::nvidia_gpu::runCrossCTAMBarrierInitSyncInsertion(
            allocation, computeCapability)))
      return signalPassFailure();
    ModuleMembarAnalysis membarPass(&allocation, canSkipBarSync);
    membarPass.run();
    if (enableConcurrencySanitizer) {
      auto hooks = mlir::triton::instrument::createConSanHooks("nvidia");
      assert(hooks && "no ConSan hooks registered for nvidia");
      if (failed(mlir::triton::instrument::runConcurrencySanitizer(
              mod, hooks.get())))
        return signalPassFailure();
      mlir::PassManager cleanupPm(context);
      cleanupPm.addPass(mlir::triton::gluon::createGluonCanonicalize());
      cleanupPm.addPass(mlir::createCSEPass());
      if (failed(cleanupPm.run(mod)))
        return signalPassFailure();
    }
    bool hasGlobalScratchAlloc = false;
    mod.walk([&](triton::gpu::GlobalScratchAllocOp) {
      hasGlobalScratchAlloc = true;
    });
    if (hasGlobalScratchAlloc)
      mlir::triton::gpu::runGlobalScratchMemoryAllocation(mod);

    mlir::LowerToLLVMOptions option(context);
    option.overrideIndexBitwidth(32);
    TritonGPUToLLVMTypeConverter typeConverter(context, option, targetInfo);

    // Lower functions
    TritonLLVMFunctionConversionTarget funcTarget(*context);
    RewritePatternSet funcPatterns(context);
    mlir::triton::populateFuncOpConversionPattern(
        typeConverter, funcPatterns, targetInfo, patternBenefitDefault);
    if (failed(
            applyPartialConversion(mod, funcTarget, std::move(funcPatterns))))
      return signalPassFailure();

    // initSharedMemory is run before the conversion of call and ret ops,
    // because the call op has to know the shared memory base address of each
    // function
    initSharedMemory(typeConverter);

    RewritePatternSet moduloTwoPhasePatterns(context);
    moduloTwoPhasePatterns.add<FoldModuloTwoPhaseSelect>(context);
    (void)applyPatternsGreedily(mod, std::move(moduloTwoPhasePatterns));

    RewritePatternSet moduloTwoCounterPatterns(context);
    moduloTwoCounterPatterns.add<FoldModuloTwoCounterSelect>(context);
    (void)applyPatternsGreedily(mod, std::move(moduloTwoCounterPatterns));

    ModuleAxisInfoAnalysis axisInfoAnalysis(mod);

    RewritePatternSet patterns(context);
    int benefit = patternBenefitPrioritizeOverLLVMConversions;
    mlir::triton::NVIDIA::populateConvertLayoutOpToLLVMPatterns(
        typeConverter, targetInfo, patterns, benefit);
    mlir::triton::NVIDIA::populateTensorMemorySubviewOpToLLVMPattern(
        typeConverter, patterns, patternBenefitNvidiaTensorCoreSubviewPattern);
    mlir::triton::NVIDIA::populateTMAToLLVMPatterns(typeConverter, targetInfo,
                                                    patterns, benefit);
    populateDotOpToLLVMPatterns(typeConverter, patterns, computeCapability,
                                benefit);
    populateElementwiseOpToLLVMPatterns(typeConverter, patterns,
                                        axisInfoAnalysis, computeCapability,
                                        targetInfo, benefit);
    populateClampFOpToLLVMPattern(typeConverter, patterns, axisInfoAnalysis,
                                  computeCapability,
                                  patternBenefitClampOptimizedPattern);
    populateLoadStoreOpToLLVMPatterns(typeConverter, targetInfo,
                                      computeCapability, patterns,
                                      axisInfoAnalysis, benefit);
    mlir::triton::populateReduceOpToLLVMPatterns(typeConverter, patterns,
                                                 targetInfo, benefit);
    mlir::triton::populateScanOpToLLVMPatterns(typeConverter, patterns,
                                               targetInfo, benefit);
    mlir::triton::populateGatherOpToLLVMPatterns(typeConverter, patterns,
                                                 targetInfo, benefit);
    populateBarrierOpToLLVMPatterns(typeConverter, patterns, benefit,
                                    targetInfo);
    populateClusterOpsToLLVMPatterns(typeConverter, patterns, benefit);
    mlir::triton::populateHistogramOpToLLVMPatterns(typeConverter, patterns,
                                                    targetInfo, benefit);
    mlir::triton::populatePrintOpToLLVMPattern(typeConverter, patterns,
                                               targetInfo, benefit);
    mlir::triton::populateControlFlowOpToLLVMPattern(typeConverter, patterns,
                                                     targetInfo, benefit);
    mlir::triton::NVIDIA::populateSPMDOpToLLVMPattern(typeConverter, patterns,
                                                      benefit);
    mlir::triton::populateSPMDOpToLLVMPattern(typeConverter, patterns,
                                              targetInfo, benefit);
    // TODO(thomas): this should probably be done in a separate step to not
    // interfere with our own lowering of arith ops. Add arith/math's patterns
    // to help convert scalar expression to LLVM.
    mlir::arith::populateCeilFloorDivExpandOpsPatterns(patterns);
    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    mlir::populateMathToLLVMConversionPatterns(typeConverter, patterns);
    mlir::populateGpuToNVVMConversionPatterns(typeConverter, patterns);
    mlir::ub::populateUBToLLVMConversionPatterns(typeConverter, patterns);
    mlir::triton::populateViewOpToLLVMPatterns(typeConverter, patterns,
                                               benefit);
    mlir::triton::populateAssertOpToLLVMPattern(typeConverter, patterns,
                                                targetInfo, benefit);
    mlir::triton::NVIDIA::populateMemoryOpToLLVMPatterns(
        typeConverter, targetInfo, patterns, benefit);
    mlir::triton::NVIDIA::populateTensorMemoryOpToLLVMPattern(
        typeConverter, patterns, benefit);
    mlir::triton::populateMakeRangeOpToLLVMPattern(typeConverter, targetInfo,
                                                   patterns, benefit);
    mlir::triton::NVIDIA::populateTCGen5MMAOpToLLVMPattern(typeConverter,
                                                           patterns, benefit);
    mlir::triton::NVIDIA::populateFp4ToFpToLLVMPatterns(typeConverter, patterns,
                                                        benefit);
    mlir::triton::populateInstrumentationToLLVMPatterns(typeConverter, patterns,
                                                        targetInfo);
    mlir::triton::populateFpSanToLLVMPatterns(typeConverter, patterns);
    mlir::triton::populateGSanToLLVMPatterns(typeConverter, patterns,
                                             axisInfoAnalysis, targetInfo);

    TritonLLVMConversionTarget convTarget(*context);
    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns))))
      return signalPassFailure();

    // Lower CF ops separately to avoid breaking analysis.
    TritonLLVMFunctionConversionTarget cfTarget(*context);
    cfTarget.markUnknownOpDynamicallyLegal([&](Operation *op) {
      return op->getDialect() !=
             context->getLoadedDialect<cf::ControlFlowDialect>();
    });
    RewritePatternSet cfPatterns(context);
    mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                          cfPatterns);
    if (failed(applyPartialConversion(mod, cfTarget, std::move(cfPatterns))))
      return signalPassFailure();

    // Fold CTAId when there is only 1 CTA.
    int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(mod);
    if (numCTAs == 1) {
      mod.walk([](triton::nvgpu::ClusterCTAIdOp id) {
        OpBuilder b(id);
        Value zero = LLVM::createConstantI32(id->getLoc(), b, 0);
        id.replaceAllUsesWith(zero);
      });
    }
    fixUpLoopAnnotation(mod);

    // Ensure warp group code is isolated from above.
    makeAllWarpGroupsIsolatedFromAbove(mod);
  }

private:
  void initSharedMemory(LLVMTypeConverter &typeConverter) {
    ModuleOp mod = getOperation();
    OpBuilder b(mod.getBodyRegion());
    auto loc = mod.getLoc();
    auto elemTy = typeConverter.convertType(b.getIntegerType(8));
    // Set array size 0 and external linkage indicates that we use dynamic
    // shared allocation to allow a larger shared memory size for each kernel.
    //
    // Ask for 16B alignment on global_smem because that's the largest we should
    // ever need (4xi32).
    auto arrayTy = LLVM::LLVMArrayType::get(elemTy, 0);
    LLVM::GlobalOp::create(
        b, loc, arrayTy, /*isConstant=*/false, LLVM::Linkage::External,
        "global_smem", /*value=*/Attribute(), /*alignment=*/16,
        // Add ROCm support.
        static_cast<unsigned>(NVVM::NVVMMemorySpace::Shared));
  }
};

} // anonymous namespace

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createConvertTritonGPUToLLVMPass() {
  return std::make_unique<ConvertTritonGPUToLLVM>();
}
std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonGPUToLLVMPass(int32_t computeCapability) {
  return std::make_unique<ConvertTritonGPUToLLVM>(computeCapability);
}
std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonGPUToLLVMPass(int32_t computeCapability,
                                 int32_t ptxVersion) {
  return std::make_unique<ConvertTritonGPUToLLVM>(computeCapability,
                                                  ptxVersion);
}

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonGPUToLLVMPass(int32_t computeCapability, int32_t ptxVersion,
                                 bool enableConcurrencySanitizer) {
  return std::make_unique<ConvertTritonGPUToLLVM>(computeCapability, ptxVersion,
                                                  enableConcurrencySanitizer);
}

bool NVIDIA::canSkipBarSync(Operation *before, Operation *after,
                            bool /*beforeIsRead*/, bool /*afterIsRead*/,
                            Allocation *allocation) {
  // These mbarrier ops are single threaded, so are always synchronized wrt.
  // each other.
  if (isa<ttng::InitBarrierOp, ttng::InvalBarrierOp, ttng::BarrierExpectOp>(
          before) &&
      isa<ttng::InitBarrierOp, ttng::InvalBarrierOp, ttng::BarrierExpectOp>(
          after))
    return true;

  // wait_barrier will never run ahead of the load it's waiting on
  if (isa<ttng::TMALoadLikeOpInterface>(before) &&
      isa<ttng::WaitBarrierOp>(after))
    return true;

  return false;
}

} // namespace triton
} // namespace mlir
