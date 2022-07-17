// RUN: lcl-opt --convert-gpu-to-cpp %s | FileCheck %s
module {
  // CHECK: emitc.include <"metal_stdlib">
  // CHECK: emitc.include <"simd/simd.h">
  // CHECK-NOT: gpu.module
  gpu.module @ocl_program {
    func.func private @_Z13get_global_idj(%arg0: i32, %arg1: vector<3xi32>, %arg2: vector<3xi32>, %arg3: vector<3xi32>, %arg4: vector<3xi32>) -> i64 {
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %0 = arith.cmpi eq, %arg0, %c0_i32 : i32
      cf.cond_br %0, ^bb3(%c0 : index), ^bb1
    ^bb1:  // pred: ^bb0
      %1 = arith.cmpi eq, %arg0, %c1_i32 : i32
      cf.cond_br %1, ^bb3(%c1 : index), ^bb2
    ^bb2:  // pred: ^bb1
      %2 = arith.cmpi eq, %arg0, %c2_i32 : i32
      cf.cond_br %2, ^bb3(%c2 : index), ^bb4(%c0 : index)
    ^bb3(%3: index):  // 3 preds: ^bb0, ^bb1, ^bb2
      %4 = vector.extractelement %arg1[%3 : index] : vector<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      cf.br ^bb4(%5 : index)
    ^bb4(%6: index):  // 2 preds: ^bb2, ^bb3
      %7 = arith.index_cast %6 : index to i64
      return %7 : i64
    }
    // CHECK-NOT: gpu.func
    // CHECK: func @vectorAdd
    gpu.func @vectorAdd(%arg0: !rawmem.ptr<f32, 1>, %arg1: !rawmem.ptr<f32, 1>, %arg2: !rawmem.ptr<f32, 1>, %arg3: !rawmem.ptr<i32, 1>, %arg4: vector<3xi32>, %arg5: vector<3xi32>, %arg6: vector<3xi32>, %arg7: vector<3xi32>) kernel attributes {spv.entry_point_abi = {local_size = dense<1> : vector<3xi32>}} {
      %c0_i32 = arith.constant 0 : i32
      %0 = rawmem.load %arg3 : !rawmem.ptr<i32, 1>, i32
      %1 = func.call @_Z13get_global_idj(%c0_i32, %arg4, %arg5, %arg6, %arg7) : (i32, vector<3xi32>, vector<3xi32>, vector<3xi32>, vector<3xi32>) -> i64
      %2 = arith.trunci %1 : i64 to i32
      %3 = arith.cmpi ult, %2, %0 : i32
      cf.cond_br %3, ^bb1, ^bb2
    ^bb1:  // pred: ^bb0
      %4 = arith.index_cast %2 : i32 to index
      %5 = rawmem.load %arg0[%4] : !rawmem.ptr<f32, 1>, f32
      %6 = arith.index_cast %2 : i32 to index
      %7 = rawmem.load %arg1[%6] : !rawmem.ptr<f32, 1>, f32
      %8 = arith.addf %5, %7 : f32
      %9 = arith.index_cast %2 : i32 to index
      rawmem.store %8, %arg2[%9] : f32, !rawmem.ptr<f32, 1>
      cf.br ^bb2
    ^bb2:  // 2 preds: ^bb0, ^bb1
      // CHECK-NOT: gpu.return
      // CHECK: return
      gpu.return
    }
  }
}
