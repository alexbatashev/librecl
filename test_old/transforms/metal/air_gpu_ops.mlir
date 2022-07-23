// RUN: lcl-opt --expand-gpu-builtins %s | FileCheck %s

module {
  gpu.module @ocl_program {
    func.func private @_Z13get_global_idj(%arg0: i32, %arg1: vector<3xi32>, %arg2: vector<3xi32>, %arg3: vector<3xi32>, %arg4: vector<3xi32>) -> i64 {
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %0 = arith.cmpi eq, %arg0, %c0_i32 : i32
      cf.cond_br %0, ^bb3, ^bb1
    ^bb1:  // pred: ^bb0
      %1 = arith.cmpi eq, %arg0, %c1_i32 : i32
      cf.cond_br %1, ^bb4, ^bb2
    ^bb2:  // pred: ^bb1
      %2 = arith.cmpi eq, %arg0, %c2_i32 : i32
      cf.cond_br %2, ^bb5, ^bb6(%c0 : index)
    ^bb3:  // pred: ^bb0
      // CHECK: [[VAL1:%[0-9]+]] = vector.extractelement %arg1[%c0_0 : index] : vector<3xi32>
      %3 = gpu.global_id  x
      cf.br ^bb6(%3 : index)
    ^bb4:  // pred: ^bb1
      // CHECK: [[VAL2:%[0-9]+]] = vector.extractelement %arg1[%c1 : index] : vector<3xi32>
      %4 = gpu.global_id  y
      cf.br ^bb6(%4 : index)
    ^bb5:  // pred: ^bb2
      // CHECK: [[VAL3:%[0-9]+]] = vector.extractelement %arg1[%c2 : index] : vector<3xi32>
      %5 = gpu.global_id  z
      cf.br ^bb6(%5 : index)
    ^bb6(%6: index):  // 4 preds: ^bb2, ^bb3, ^bb4, ^bb5
      %7 = arith.index_cast %6 : index to i64
      return %7 : i64
    }
    gpu.func @vectorAdd(%arg0: !rawmem.ptr<f32, 1>, %arg1: !rawmem.ptr<f32, 1>, %arg2: !rawmem.ptr<f32, 1>, %arg3: !rawmem.ptr<i32, 1>, %arg4: vector<3xi32>, %arg5: vector<3xi32>, %arg6: vector<3xi32>, %arg7: vector<3xi32>) kernel attributes {spv.entry_point_abi = {local_size = dense<1> : vector<3xi32>}} {
      %0 = rawmem.load %arg3 : !rawmem.ptr<i32, 1>, i32
      %c0_i32 = arith.constant 0 : i32
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
      gpu.return
    }
  }
}
