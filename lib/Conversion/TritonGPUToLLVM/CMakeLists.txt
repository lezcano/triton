get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

add_subdirectory(Templates)

add_triton_library(TritonGPUToLLVM
    ConvertLayoutOpToLLVM/SharedToDotOperandFMA.cpp
    DotOpToLLVM/FMA.cpp
    TypeConverter.cpp
    Utility.cpp
    CompileUtils.cpp
    ElementwiseOpToLLVM.cpp
    MemoryOpToLLVM.cpp
    AssertOpToLLVM.cpp
    ViewOpToLLVM.cpp
    MakeRangeOpToLLVM.cpp
    HistogramOpToLLVM.cpp
    AllocateSharedMemory.cpp
    ReduceOpToLLVM.cpp
    ScanOpToLLVM.cpp
    ConvertLayoutOpToLLVM.cpp
    ControlFlowOpToLLVM.cpp
    FuncOpToLLVM.cpp
    SPMDOpToLLVM.cpp
    DecomposeUnsupportedConversions.cpp

    DEPENDS
    TritonGPUConversionPassIncGen

    LINK_LIBS PUBLIC
    ${translation_libs}
    LLVMSupport
    MLIRIR
    MLIRTargetLLVMIRImport
    MLIRPass
    MLIRGPUDialect
    MLIRGPUToNVVMTransforms
    MLIRGPUToROCDLTransforms
    MLIRGPUTransforms
    TritonAnalysis
    TritonIR
    TritonGPUIR
    TritonGPUTransforms
    TritonNvidiaGPUTransforms
    NVGPUIR
    Templates
)
