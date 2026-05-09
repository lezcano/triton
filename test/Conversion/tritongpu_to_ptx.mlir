// RUN: triton-opt %s --allocate-shared-memory-nv='compute-capability=90 ptx-version=83' --convert-triton-gpu-to-llvm='compute-capability=90 ptx-version=83' --convert-nv-gpu-to-llvm | mlir-translate --mlir-to-llvmir | opt -O3 -S | llc -mtriple nvptx64-nvidia-cuda -mcpu=sm_90 -mattr=+ptx83 | FileCheck --check-prefixes CHECK,SM90 --dump-input-context=20 %s
// RUN: triton-opt %s --allocate-shared-memory-nv='compute-capability=80 ptx-version=83' --convert-triton-gpu-to-llvm='compute-capability=80 ptx-version=83' --convert-nv-gpu-to-llvm | mlir-translate --mlir-to-llvmir | opt -O3 -S | llc -mtriple nvptx64-nvidia-cuda -mcpu=sm_80 -mattr=+ptx83 | FileCheck --check-prefixes CHECK,SM80 --dump-input-context=20 %s
// RUN: triton-opt %s --allocate-shared-memory-nv='compute-capability=100 ptx-version=87' --convert-triton-gpu-to-llvm='compute-capability=100 ptx-version=87' --convert-nv-gpu-to-llvm | mlir-translate --mlir-to-llvmir | opt -O3 -S | llc -mtriple nvptx64-nvidia-cuda -mcpu=sm_100 -mattr=+ptx87 | FileCheck --check-prefixes CHECK,SM100 --dump-input-context=20 %s
// RUN: triton-opt %s --convert-triton-gpu-to-llvm='compute-capability=80 ptx-version=83' -cse | FileCheck --check-prefix=VEC80 --dump-input-context=20 %s
// RUN: triton-opt %s --convert-triton-gpu-to-llvm='compute-capability=90 ptx-version=83' -cse | FileCheck --check-prefix=VEC90 --dump-input-context=20 %s
// RUN: triton-opt %s --convert-triton-gpu-to-llvm='compute-capability=100 ptx-version=87' -cse | FileCheck --check-prefix=VEC100 --dump-input-context=20 %s


#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
#blocked_reduce = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [1, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @add_bf16(%ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg0: tensor<256xbf16, #blocked>, %arg1: tensor<256xbf16, #blocked>) {
    // CHECK-LABEL: add_bf16
    // SM80-COUNT-4: fma.rn.bf16x2
    // SM90-COUNT-4: add.rn.bf16x2
    %0 = arith.addf %arg0, %arg1 : tensor<256xbf16, #blocked>
    %1 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %2 = tt.splat %ptr : !tt.ptr<bf16> -> tensor<256x!tt.ptr<bf16>, #blocked>
    %3 = tt.addptr %2, %1 : tensor<256x!tt.ptr<bf16>, #blocked>, tensor<256xi32, #blocked>
    tt.store %3, %0 : tensor<256x!tt.ptr<bf16>, #blocked>
    tt.return
  }

  tt.func public @sub_bf16(%ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg0: tensor<256xbf16, #blocked>, %arg1: tensor<256xbf16, #blocked>) {
    // CHECK-LABEL: sub_bf16
    // SM80-COUNT-4: fma.rn.bf16x2
    // SM90-COUNT-4: sub.rn.bf16x2
    %0 = arith.subf %arg0, %arg1 : tensor<256xbf16, #blocked>
    %1 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %2 = tt.splat %ptr : !tt.ptr<bf16> -> tensor<256x!tt.ptr<bf16>, #blocked>
    %3 = tt.addptr %2, %1 : tensor<256x!tt.ptr<bf16>, #blocked>, tensor<256xi32, #blocked>
    tt.store %3, %0 : tensor<256x!tt.ptr<bf16>, #blocked>
    tt.return
  }

  tt.func public @mul_bf16(%ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg0: tensor<256xbf16, #blocked>, %arg1: tensor<256xbf16, #blocked>) {
    // CHECK-LABEL: mul_bf16
    // SM80-COUNT-4: fma.rn.bf16x2
    // SM90-COUNT-4: mul.rn.bf16x2
    %0 = arith.mulf %arg0, %arg1 : tensor<256xbf16, #blocked>
    %1 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %2 = tt.splat %ptr : !tt.ptr<bf16> -> tensor<256x!tt.ptr<bf16>, #blocked>
    %3 = tt.addptr %2, %1 : tensor<256x!tt.ptr<bf16>, #blocked>, tensor<256xi32, #blocked>
    tt.store %3, %0 : tensor<256x!tt.ptr<bf16>, #blocked>
    tt.return
  }

  tt.func public @extf_bf16(%ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg0: tensor<256xbf16, #blocked>) {
    // CHECK-LABEL: extf_bf16
    // CHECK-COUNT-8: cvt.f32.bf16
    %0 = arith.extf %arg0 : tensor<256xbf16, #blocked> to tensor<256xf32, #blocked>
    %1 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %2 = tt.splat %ptr : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked>
    %3 = tt.addptr %2, %1 : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>
    tt.store %3, %0 : tensor<256x!tt.ptr<f32>, #blocked>
    tt.return
  }

  tt.func public @truncf_bf16(%ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg0: tensor<256xf32, #blocked>) {
    // CHECK-LABEL: truncf_bf16
    // CHECK-COUNT-4: cvt.rn.bf16x2.f32
    %0 = arith.truncf %arg0 : tensor<256xf32, #blocked> to tensor<256xbf16, #blocked>
    %1 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %2 = tt.splat %ptr : !tt.ptr<bf16> -> tensor<256x!tt.ptr<bf16>, #blocked>
    %3 = tt.addptr %2, %1 : tensor<256x!tt.ptr<bf16>, #blocked>, tensor<256xi32, #blocked>
    tt.store %3, %0 : tensor<256x!tt.ptr<bf16>, #blocked>
    tt.return
  }

  tt.func public @extf_f16(%ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg0: tensor<256xf16, #blocked>) {
    // CHECK-LABEL: extf_f16
    // CHECK-COUNT-8: cvt.f32.f16
    %0 = arith.extf %arg0 : tensor<256xf16, #blocked> to tensor<256xf32, #blocked>
    %1 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %2 = tt.splat %ptr : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked>
    %3 = tt.addptr %2, %1 : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>
    tt.store %3, %0 : tensor<256x!tt.ptr<f32>, #blocked>
    tt.return
  }

  tt.func public @truncf_f16(%ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg0: tensor<256xf32, #blocked>) {
    // CHECK-LABEL: truncf_f16
    // CHECK-COUNT-4: cvt.rn.f16x2.f32
    // VEC80-LABEL: llvm.func @truncf_f16
    // VEC80-COUNT-4: llvm.inline_asm {{.*}}cvt.rn.f16x2.f32
    // VEC80-NOT: llvm.fptrunc
    // VEC90-LABEL: llvm.func @truncf_f16
    // VEC90-COUNT-4: llvm.inline_asm {{.*}}cvt.rn.f16x2.f32
    // VEC90-NOT: llvm.fptrunc
    // VEC100-LABEL: llvm.func @truncf_f16
    // VEC100-COUNT-4: llvm.inline_asm {{.*}}cvt.rn.f16x2.f32
    // VEC100-NOT: llvm.fptrunc
    %0 = arith.truncf %arg0 : tensor<256xf32, #blocked> to tensor<256xf16, #blocked>
    %1 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %2 = tt.splat %ptr : !tt.ptr<f16> -> tensor<256x!tt.ptr<f16>, #blocked>
    %3 = tt.addptr %2, %1 : tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xi32, #blocked>
    tt.store %3, %0 : tensor<256x!tt.ptr<f16>, #blocked>
    tt.return
  }

  tt.func public @key_to_fpval_i16(%ptr: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %arg0: tensor<256xi16, #blocked>) {
    // CHECK-LABEL: key_to_fpval_i16
    // CHECK-COUNT-4: prmt.b32 {{.*}}0xbb99
    %top = arith.constant dense<-32768> : tensor<256xi16, #blocked>
    %full = arith.constant dense<-1> : tensor<256xi16, #blocked>
    %zero = arith.constant dense<0> : tensor<256xi32, #blocked>
    %sign = arith.andi %arg0, %top : tensor<256xi16, #blocked>
    %wide = arith.extui %sign : tensor<256xi16, #blocked> to tensor<256xi32, #blocked>
    %is_positive = arith.cmpi eq, %wide, %zero : tensor<256xi32, #blocked>
    %mask = arith.select %is_positive, %full, %top : tensor<256xi1, #blocked>, tensor<256xi16, #blocked>
    %decoded = arith.xori %arg0, %mask : tensor<256xi16, #blocked>
    %1 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %2 = tt.splat %ptr : !tt.ptr<i16> -> tensor<256x!tt.ptr<i16>, #blocked>
    %3 = tt.addptr %2, %1 : tensor<256x!tt.ptr<i16>, #blocked>, tensor<256xi32, #blocked>
    tt.store %3, %decoded : tensor<256x!tt.ptr<i16>, #blocked>
    tt.return
  }

  tt.func public @fpval_to_key_i16(%ptr: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %arg0: tensor<256xi16, #blocked>) {
    // CHECK-LABEL: fpval_to_key_i16
    // CHECK-COUNT-4: prmt.b32 {{.*}}0xbb99
    %top = arith.constant dense<-32768> : tensor<256xi16, #blocked>
    %full = arith.constant dense<-1> : tensor<256xi16, #blocked>
    %zero = arith.constant dense<0> : tensor<256xi32, #blocked>
    %sign = arith.andi %arg0, %top : tensor<256xi16, #blocked>
    %wide = arith.extui %sign : tensor<256xi16, #blocked> to tensor<256xi32, #blocked>
    %is_negative = arith.cmpi ne, %wide, %zero : tensor<256xi32, #blocked>
    %mask = arith.select %is_negative, %full, %top : tensor<256xi1, #blocked>, tensor<256xi16, #blocked>
    %key = arith.xori %arg0, %mask : tensor<256xi16, #blocked>
    %1 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %2 = tt.splat %ptr : !tt.ptr<i16> -> tensor<256x!tt.ptr<i16>, #blocked>
    %3 = tt.addptr %2, %1 : tensor<256x!tt.ptr<i16>, #blocked>, tensor<256xi32, #blocked>
    tt.store %3, %key : tensor<256x!tt.ptr<i16>, #blocked>
    tt.return
  }

  tt.func public @fpval_to_key_extui_i16(%ptr: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg0: tensor<256xi16, #blocked>) {
    // CHECK-LABEL: fpval_to_key_extui_i16
    // CHECK-COUNT-4: prmt.b32 {{.*}}0xbb99
    // CHECK-COUNT-4: shr.u32
    // CHECK-NOT: cvt.u32.u16
    %top = arith.constant dense<-32768> : tensor<256xi16, #blocked>
    %full = arith.constant dense<-1> : tensor<256xi16, #blocked>
    %zero = arith.constant dense<0> : tensor<256xi32, #blocked>
    %sign = arith.andi %arg0, %top : tensor<256xi16, #blocked>
    %wide = arith.extui %sign : tensor<256xi16, #blocked> to tensor<256xi32, #blocked>
    %is_negative = arith.cmpi ne, %wide, %zero : tensor<256xi32, #blocked>
    %mask = arith.select %is_negative, %full, %top : tensor<256xi1, #blocked>, tensor<256xi16, #blocked>
    %key = arith.xori %arg0, %mask : tensor<256xi16, #blocked>
    %key_wide = arith.extui %key : tensor<256xi16, #blocked> to tensor<256xi32, #blocked>
    %1 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %2 = tt.splat %ptr : !tt.ptr<i32> -> tensor<256x!tt.ptr<i32>, #blocked>
    %3 = tt.addptr %2, %1 : tensor<256x!tt.ptr<i32>, #blocked>, tensor<256xi32, #blocked>
    tt.store %3, %key_wide : tensor<256x!tt.ptr<i32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: reduce_f16_store
  // SM80-NOT: add.rn.f16x2
  // SM90: add.rn.f16x2
  // SM100: add.rn.f16x2
  // VEC80-LABEL: llvm.func @reduce_f16_store
  // VEC80-NOT: llvm.fadd {{.*}} : vector<2xf16>
  // VEC90-LABEL: llvm.func @reduce_f16_store
  // VEC90: llvm.fadd {{.*}} : vector<2xf16>
  // VEC100-LABEL: llvm.func @reduce_f16_store
  // VEC100: llvm.fadd {{.*}} : vector<2xf16>
  tt.func public @reduce_f16_store(%out: !tt.ptr<f16>, %arg0: tensor<1x256xf16, #blocked_reduce>) {
    %r = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
    ^bb0(%a: f16, %b: f16):
      %sum = arith.addf %a, %b : f16
      tt.reduce.return %sum : f16
    }) {allocation.offset = 0 : i32} : (tensor<1x256xf16, #blocked_reduce>) -> tensor<1xf16, #ttg.slice<{dim = 1, parent = #blocked_reduce}>>
    %ptr = tt.splat %out : !tt.ptr<f16> -> tensor<1x!tt.ptr<f16>, #ttg.slice<{dim = 1, parent = #blocked_reduce}>>
    tt.store %ptr, %r : tensor<1x!tt.ptr<f16>, #ttg.slice<{dim = 1, parent = #blocked_reduce}>>
    tt.return
  }

  // CHECK-LABEL: reduce_f32_store
  // VEC80-LABEL: llvm.func @reduce_f32_store
  // VEC80-NOT: llvm.fadd {{.*}} : vector<2xf32>
  // VEC90-LABEL: llvm.func @reduce_f32_store
  // VEC90-NOT: llvm.fadd {{.*}} : vector<2xf32>
  // VEC100-LABEL: llvm.func @reduce_f32_store
  // VEC100: llvm.fadd {{.*}} : vector<2xf32>
  tt.func public @reduce_f32_store(%out: !tt.ptr<f32>, %arg0: tensor<1x256xf32, #blocked_reduce>) {
    %r = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %sum = arith.addf %a, %b : f32
      tt.reduce.return %sum : f32
    }) {allocation.offset = 0 : i32} : (tensor<1x256xf32, #blocked_reduce>) -> tensor<1xf32, #ttg.slice<{dim = 1, parent = #blocked_reduce}>>
    %ptr = tt.splat %out : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked_reduce}>>
    tt.store %ptr, %r : tensor<1x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked_reduce}>>
    tt.return
  }
}
