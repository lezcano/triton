add_triton_library(Templates
    Compile.cpp

    LINK_LIBS PUBLIC
    clangFrontend
    clangTooling
    clangCodeGen
)
