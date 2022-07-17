// RUN: lcloc --stage=msl --target=metal-ios %s | FileCheck %s

__kernel void vectorAdd(__global float *a, __global float *b, __global float *c, const unsigned int n) {
  int id = get_global_id(0);

  if (id < n) {
    c[id] = a[id] + b[id];
  }
}

// CHECK: #include <metal_stdlib>
// CHECK-NEXT: #include <simd/simd.h>
// CHECK-NEXT:  int64_t _Z13get_global_idj(int32_t v1, vec<int32_t, 3> v2, vec<int32_t, 3> v3, vec<int32_t, 3> v4, vec<int32_t, 3> v5) {
// CHECK-NEXT:    size_t v6;
// CHECK-NEXT:    size_t v7;
// CHECK-NEXT:    size_t v8;
// CHECK-NEXT:    int32_t v9;
// CHECK-NEXT:    int32_t v10;
// CHECK-NEXT:    int32_t v11;
// CHECK-NEXT:    bool v12;
// CHECK-NEXT:    bool v13;
// CHECK-NEXT:    bool v14;
// CHECK-NEXT:    int32_t v15;
// CHECK-NEXT:    size_t v16;
// CHECK-NEXT:    int64_t v17;
// CHECK-NEXT:    size_t v18;
// CHECK-NEXT:    size_t v19;
// CHECK-NEXT:    v6 = 2;
// CHECK-NEXT:    v7 = 1;
// CHECK-NEXT:    v8 = 0;
// CHECK-NEXT:    v9 = 2;
// CHECK-NEXT:    v10 = 1;
// CHECK-NEXT:    v11 = 0;
// CHECK-NEXT:    v12 = v1 == v11;
// CHECK-NEXT:    if (v12) {
// CHECK-NEXT:      v18 = v8;
// CHECK-NEXT:      goto label4;
// CHECK-NEXT:    } else {
// CHECK-NEXT:      goto label2;
// CHECK-NEXT:    }
// CHECK-NEXT:  label2:
// CHECK-NEXT:    v13 = v1 == v10;
// CHECK-NEXT:    if (v13) {
// CHECK-NEXT:      v18 = v7;
// CHECK-NEXT:      goto label4;
// CHECK-NEXT:    } else {
// CHECK-NEXT:      goto label3;
// CHECK-NEXT:    }
// CHECK-NEXT:  label3:
// CHECK-NEXT:    v14 = v1 == v9;
// CHECK-NEXT:    if (v14) {
// CHECK-NEXT:      v18 = v6;
// CHECK-NEXT:      goto label4;
// CHECK-NEXT:    } else {
// CHECK-NEXT:      v19 = v8;
// CHECK-NEXT:      goto label5;
// CHECK-NEXT:    }
// CHECK-NEXT:  label4:
// CHECK-NEXT:    v15 = v2[v18];
// CHECK-NEXT:    v16 = (size_t) v15;
// CHECK-NEXT:    v19 = v16;
// CHECK-NEXT:    goto label5;
// CHECK-NEXT:  label5:
// CHECK-NEXT:    v17 = (int64_t) v19;
// CHECK-NEXT:    return v17;
// CHECK-NEXT:  }

// CHECK-NEXT:  kernel void vectorAdd(device float* v1, device float* v2, device float* v3, device int32_t* v4, vec<int32_t, 3> v5 [[thread_position_in_grid]], vec<int32_t, 3> v6 [[thread_position_in_threadgroup]], vec<int32_t, 3> v7 [[threads_per_threadgroup]], vec<int32_t, 3> v8 [[threadgroups_per_grid]]) {
// CHECK-NEXT:    int32_t v9;
// CHECK-NEXT:    int32_t v10;
// CHECK-NEXT:    int64_t v11;
// CHECK-NEXT:    int32_t v12;
// CHECK-NEXT:    bool v13;
// CHECK-NEXT:    size_t v14;
// CHECK-NEXT:    float v15;
// CHECK-NEXT:    size_t v16;
// CHECK-NEXT:    float v17;
// CHECK-NEXT:    float v18;
// CHECK-NEXT:    size_t v19;
// CHECK-NEXT:    v9 = 0;
// CHECK-NEXT:    v10 = v4[0;
// CHECK-NEXT:    v11 = _Z13get_global_idj(v9, v5, v6, v7, v8);
// CHECK-NEXT:    v12 = (int32_t) v11;
// CHECK-NEXT:    v13 = v12 <= v10;
// CHECK-NEXT:    if (v13) {
// CHECK-NEXT:      goto label2;
// CHECK-NEXT:    } else {
// CHECK-NEXT:      goto label3;
// CHECK-NEXT:    }
// CHECK-NEXT:  label2:
// CHECK-NEXT:    v14 = (size_t) v12;
// CHECK-NEXT:    v15 = v1[v14];
// CHECK-NEXT:    v16 = (size_t) v12;
// CHECK-NEXT:    v17 = v2[v16];
// CHECK-NEXT:    v18 = v15 + v17;
// CHECK-NEXT:    v19 = (size_t) v12;
// CHECK-NEXT:    v3[v19] = v18;
// CHECK-NEXT:    goto label3;
// CHECK-NEXT:  label3:
// CHECK-NEXT:    return;
// CHECK-NEXT:  }
