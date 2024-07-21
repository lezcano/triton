To generate the LLVM IR files, compile LLVM with
```bash
cmake -S llvm -B build -G Ninja -DLLVM_ENABLE_PROJECTS='mlir;clang;lld' -DLLVM_TARGETS_TO_BUILD='AMDGPU;NVPTX;X86' -DCMAKE_BUILD_TYPE=Debug -DLLVM_DEFAULT_TARGET_TRIPLE=x86_64-unknown-linux-gnu
```
and compile triton using this LLVM as per
```bash
LLVM_SYSPATH=/path-to-llvm/build pip install -e python
```

Then, compile the relevant file as per
```bash
${LLVM_SYSPATH}/bin/clang++ reduce.cu --cuda-gpu-arch=sm_75 --cuda-path=/usr/local/cuda -I$CONDA_PREFIX/include -emit-llvm -c
```
