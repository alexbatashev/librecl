// RUN: lcl-opt --structure-cfg %s | FileCheck %s
module {
  gpu.module @ocl_program {
    func.func private @_Z13get_global_idj(%arg0: i32) -> i64
    gpu.func @vectorAdd(%arg0: !rawmem.ptr<f32, 1>, %arg1: !rawmem.ptr<f32, 1>, %arg2: !rawmem.ptr<f32, 1>, %arg3: i32) kernel attributes {spv.entry_point_abi = {local_size = dense<1> : vector<3xi32>}} {
      %c0_i32 = arith.constant 0 : i32
      %0 = func.call @_Z13get_global_idj(%c0_i32) : (i32) -> i64
      %1 = arith.trunci %0 : i64 to i32
      %2 = arith.cmpi ult, %1, %arg3 : i32
    // CHECK-NOT: cf.cond_br
    // CHECK: scf.if
      cf.cond_br %2, ^bb1, ^bb2
    ^bb1:  // pred: ^bb0
      %3 = arith.index_cast %1 : i32 to index
      %4 = rawmem.load %arg0[%3] : !rawmem.ptr<f32, 1>, f32
      %5 = arith.index_cast %1 : i32 to index
      %6 = rawmem.load %arg1[%5] : !rawmem.ptr<f32, 1>, f32
      %7 = arith.addf %4, %6 : f32
      %8 = arith.index_cast %1 : i32 to index
      rawmem.store %7, %arg2[%8] : f32, !rawmem.ptr<f32, 1>
      cf.br ^bb2
    ^bb2:  // 2 preds: ^bb0, ^bb1
      gpu.return
    }
  }
}

