#ifndef TRITON_CONVERSION_TRITONGPU_TO_LLVM_COMPILE_UTILS_H
#define TRITON_CONVERSION_TRITONGPU_TO_LLVM_COMPILE_UTILS_H

#include <mlir/Transforms/DialectConversion.h>

namespace mlir::triton {
void compileReduction(mlir::ConversionPatternRewriter &rewriter);
void loadReduction(mlir::ConversionPatternRewriter &rewriter);
} // namespace mlir::triton

#endif
