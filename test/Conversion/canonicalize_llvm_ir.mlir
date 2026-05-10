// RUN: triton-opt %s -canonicalize-llvm-ir | FileCheck %s

llvm.func @reuse_normalized_masked_slice(%arg0: i32) -> i32 {
  %c96_i32 = llvm.mlir.constant(96 : i32) : i32
  %c5_i32 = llvm.mlir.constant(5 : i32) : i32
  %c3_i32 = llvm.mlir.constant(3 : i32) : i32
  %masked = llvm.and %arg0, %c96_i32 : i32
  %normalized = llvm.lshr exact %masked, %c5_i32 : i32
  %wider = llvm.lshr exact %masked, %c3_i32 : i32
  %ret = llvm.xor %normalized, %wider : i32
  llvm.return %ret : i32
}

// CHECK-LABEL: llvm.func @reuse_normalized_masked_slice
// CHECK: %[[C2:.+]] = llvm.mlir.constant(2 : i32)
// CHECK: %[[MASKED:.+]] = llvm.and
// CHECK: %[[NORM:.+]] = llvm.lshr exact %[[MASKED]]
// CHECK: %[[WIDER:.+]] = llvm.shl %[[NORM]], %[[C2]]
// CHECK: llvm.xor %[[NORM]], %[[WIDER]]

llvm.func @three_input_or(%arg0: i32, %arg1: i32, %arg2: i32) -> i32 {
  %ab = llvm.or disjoint %arg0, %arg1 : i32
  %abc = llvm.or disjoint %ab, %arg2 : i32
  llvm.return %abc : i32
}

// CHECK-LABEL: llvm.func @three_input_or
// CHECK-NOT: llvm.or
// CHECK: llvm.inline_asm asm_dialect = att "lop3.b32 $0, $1, $2, $3, 0xfe;", "=r,r,r,r"

llvm.func @reuse_normalized_masked_slice_before_cse(%arg0: i32) -> i32 {
  %c96_i32 = llvm.mlir.constant(96 : i32) : i32
  %c5_i32 = llvm.mlir.constant(5 : i32) : i32
  %c3_i32 = llvm.mlir.constant(3 : i32) : i32
  %masked0 = llvm.and %arg0, %c96_i32 : i32
  %normalized = llvm.lshr exact %masked0, %c5_i32 : i32
  %masked1 = llvm.and %arg0, %c96_i32 : i32
  %wider = llvm.lshr exact %masked1, %c3_i32 : i32
  %ret = llvm.xor %normalized, %wider : i32
  llvm.return %ret : i32
}

// CHECK-LABEL: llvm.func @reuse_normalized_masked_slice_before_cse
// CHECK: %[[C2:.+]] = llvm.mlir.constant(2 : i32)
// CHECK: %[[MASKED:.+]] = llvm.and
// CHECK: %[[NORM:.+]] = llvm.lshr exact %[[MASKED]]
// CHECK: %[[WIDER:.+]] = llvm.shl %[[NORM]], %[[C2]]
// CHECK: llvm.xor %[[NORM]], %[[WIDER]]

llvm.func @reuse_normalized_masked_slice_equivalent_input(%arg0: i32, %arg1: i32) -> i32 {
  %c5_i32 = llvm.mlir.constant(5 : i32) : i32
  %c96_i32 = llvm.mlir.constant(96 : i32) : i32
  %c3_i32 = llvm.mlir.constant(3 : i32) : i32
  %shifted0 = llvm.shl %arg1, %c5_i32 : i32
  %input0 = llvm.or %arg0, %shifted0 : i32
  %masked0 = llvm.and %input0, %c96_i32 : i32
  %normalized = llvm.lshr exact %masked0, %c5_i32 : i32
  %shifted1 = llvm.shl %arg1, %c5_i32 : i32
  %input1 = llvm.or %arg0, %shifted1 : i32
  %masked1 = llvm.and %input1, %c96_i32 : i32
  %wider = llvm.lshr exact %masked1, %c3_i32 : i32
  %ret = llvm.xor %normalized, %wider : i32
  llvm.return %ret : i32
}

// CHECK-LABEL: llvm.func @reuse_normalized_masked_slice_equivalent_input
// CHECK: %[[C2:.+]] = llvm.mlir.constant(2 : i32)
// CHECK: %[[SHIFTED:.+]] = llvm.shl
// CHECK: %[[INPUT:.+]] = llvm.or
// CHECK: %[[MASKED:.+]] = llvm.and %[[INPUT]]
// CHECK: %[[NORM:.+]] = llvm.lshr exact %[[MASKED]]
// CHECK: %[[WIDER:.+]] = llvm.shl %[[NORM]], %[[C2]]
// CHECK: llvm.xor %[[NORM]], %[[WIDER]]
