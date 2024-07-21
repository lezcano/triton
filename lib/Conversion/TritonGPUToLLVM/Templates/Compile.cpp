#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"

using namespace llvm;
using namespace clang;

void compileFile(llvm::StringRef inputFile) {
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions;

  TextDiagnosticPrinter *DiagClient =
      new TextDiagnosticPrinter(llvm::errs(), &*DiagOpts);

  IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());

  DiagnosticsEngine Diags(DiagID, &*DiagOpts, DiagClient);

  std::string clangPath = "/home/lezcano/git/llvm-project/build/bin/clang++";
  clang::driver::Driver TheDriver(clangPath,
                                  llvm::sys::getDefaultTargetTriple(), Diags);

  assert(inputFile.endswith(".cu"));

  // Path to the C file
  std::string inputPath = "ReduceTemplate.cu";

  // Path to the object file
  std::string outputPath = "reduce.o";

  std::vector<const char *> args = {
      clangPath.c_str(),
      inputPath.c_str(),
      "--cuda-gpu-arch=sm_75",
      "--cuda-path=/usr/local/cuda",
      "-I/home/lezcano/.conda/envs/pytorch-dev/include",
      "-c"
      "-emit-llvm",
      "--cuda-device-only",
  };

  // Compile C++
  TheDriver.CCCIsCXX = true;

  // Create the set of actions to perform
  std::unique_ptr<clang::driver::Compilation> Actions(
      TheDriver.BuildCompilation(args));

  // Print the set of actions
  TheDriver.PrintActions(*Actions);
  if (Actions && !Actions->containsError()) {
    SmallVector<std::pair<int, const clang::driver::Command *>, 4>
        FailingCommands;
    return TheDriver.ExecuteCompilation(*Actions, FailingCommands);
  } else {
    return -1;
  }
}
