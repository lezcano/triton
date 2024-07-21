#include "triton/Conversion/TritonGPUToLLVM/CompileUtils.h"

#include <cstdlib>
#include <memory>
#include <string>

#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Support/MemoryBuffer.h"

#include "mlir/IR/OwningOpRef.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Target/LLVMIR/ModuleImport.h"

#include "clang/CodeGen/CodeGenAction.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/Tooling.h"

using namespace mlir;
using namespace llvm;
using namespace clang;

namespace mlir::triton {

void loadReduction(mlir::ConversionPatternRewriter &rewriter) {
  const std::string path = "/home/lezcano/git/triton/lib/Conversion/"
                           "TritonGPUToLLVM/templates/reduce.bc";
  auto buffer = MemoryBuffer::getFile(path);
  if (!buffer) {
    llvm::errs() << "Failed to open bitcode file: " << path << "\n";
    std::exit(EXIT_FAILURE);
  }

  LLVMContext llvmContext;
  auto mod = parseBitcodeFile(buffer->get()->getMemBufferRef(), llvmContext);
  if (!mod) {
    llvm::errs() << "LoadLLVMBCIntoMLIR"
                 << "\n";
    std::exit(EXIT_FAILURE);
  }

  auto mlirModule =
      translateLLVMIRToModule(std::move(*mod), rewriter.getContext());
  if (!mlirModule) {
    llvm::errs() << "Error translatingLLVMIR Module"
                 << "\n";
    std::exit(EXIT_FAILURE);
  }
}
void compileReduction(mlir::ConversionPatternRewriter &rewriter) {
  const std::string path = "/home/lezcano/git/triton/lib/Conversion/"
                           "TritonGPUToLLVM/templates/reduce.bc";

  CompilerInstance compiler;
  compiler.createDiagnostics();

  auto invocation = std::make_shared<CompilerInvocation>();
  // TODO use correct sm

  CompilerInvocation::CreateFromArgs(
      *invocation,
      {"clang++", path.c_str(), "--cuda-gpu-arch=sm_75",
       "--cuda-path=/usr/local/cuda", "-I$CONDA_PREFIX/include", "-emit-llvm",
       "-c"},
      compiler.getDiagnostics());

  compiler.setInvocation(std::move(invocation));

  LLVMContext llvmContext;
  auto action = std::make_unique<EmitLLVMOnlyAction>(&llvmContext);
  if (!compiler.ExecuteAction(*action)) {
    llvm::errs() << "Error compiling Module"
                 << "\n";
    std::exit(EXIT_FAILURE);
  }
  translateLLVMIRToModule(std::move(action->takeModule()),
                          rewriter.getContext());
}
} // namespace mlir::triton
