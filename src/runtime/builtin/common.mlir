module {
  gpu.module @ocl_program {
    func.func private @_Z13get_global_idj(%arg0: i32) -> i64 attributes {ocl_builtin} {
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %0 = arith.cmpi eq, %arg0, %c0_i32 : i32
      %1 = scf.if %0 -> (index) {
        %4 = gpu.global_id  x
        scf.yield %4 : index
      } else {
        %4 = arith.cmpi eq, %arg0, %c1_i32 : i32
        %5 = scf.if %4 -> (index) {
          %6 = gpu.global_id  y
          scf.yield %6 : index
        } else {
          %6 = arith.cmpi eq, %arg0, %c2_i32 : i32
          %7 = scf.if %6 -> (index) {
            %8 = gpu.global_id  z
            scf.yield %8 : index
          } else {
            scf.yield %c0 : index
          }
          scf.yield %7 : index
        }
        scf.yield %5 : index
      }
      %2 = arith.index_cast %1 : index to i32
      %3 = arith.extsi %2 : i32 to i64
      return %3 : i64
    }
  }
}
