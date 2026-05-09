// RUN: triton-opt %s -canonicalize-llvm-ir | FileCheck %s

llvm.func @three_input_or(%arg0: i32, %arg1: i32, %arg2: i32) -> i32 {
  %ab = llvm.or disjoint %arg0, %arg1 : i32
  %abc = llvm.or disjoint %ab, %arg2 : i32
  llvm.return %abc : i32
}

// CHECK-LABEL: llvm.func @three_input_or
// CHECK-NOT: llvm.or
// CHECK: llvm.inline_asm asm_dialect = att "lop3.b32 $0, $1, $2, $3, 0xfe;", "=r,r,r,r"
