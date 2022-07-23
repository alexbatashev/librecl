// RUN: lcloc --stage=msl --target=metal-ios %s | FileCheck %s

__kernel void vectorAdd(__global float *a, __global float *b, __global float *c, const unsigned int n) {
  int id = get_global_id(0);

  if (id < n) {
    c[id] = a[id] + b[id];
  }
}

// CHECK:  #include <metal_stdlib>
// CHECK-NEXT: #include <simd/simd.h>
// CHECK-NEXT: int64_t _Z13get_global_idj(int32_t v1, vec<int32_t, 3> v2, vec<int32_t, 3> v3, vec<int32_t, 3> v4, vec<int32_t, 3> v5) {
// CHECK-NEXT:   size_t v6 = 2;
// CHECK-NEXT:   size_t v7 = 1;
// CHECK-NEXT:   size_t v8 = 0;
// CHECK-NEXT:   int32_t v9 = 2;
// CHECK-NEXT:   int32_t v10 = 1;
// CHECK-NEXT:   int32_t v11 = 0;
// CHECK-NEXT:   bool v12 = v1 == v11;
// CHECK-NEXT:   size_t v13;
// CHECK-NEXT:   if (v12) {
// CHECK-NEXT:     int32_t v14 = v2[v8];
// CHECK-NEXT:     size_t v15 = (size_t) v14;
// CHECK-NEXT:     v13 = v15;
// CHECK-NEXT:   } else {
// CHECK-NEXT:     bool v16 = v1 == v10;
// CHECK-NEXT:     size_t v17;
// CHECK-NEXT:     if (v16) {
// CHECK-NEXT:       int32_t v18 = v2[v7];
// CHECK-NEXT:       size_t v19 = (size_t) v18;
// CHECK-NEXT:       v17 = v19;
// CHECK-NEXT:     } else {
// CHECK-NEXT:       bool v20 = v1 == v9;
// CHECK-NEXT:       size_t v21;
// CHECK-NEXT:       if (v20) {
// CHECK-NEXT:         int32_t v22 = v2[v6];
// CHECK-NEXT:         size_t v23 = (size_t) v22;
// CHECK-NEXT:         v21 = v23;
// CHECK-NEXT:       } else {
// CHECK-NEXT:         v21 = v8;
// CHECK-NEXT:       };
// CHECK-NEXT:       v17 = v21;
// CHECK-NEXT:     };
// CHECK-NEXT:     v13 = v17;
// CHECK-NEXT:   }
// CHECK-NEXT:   int32_t v24 = (int32_t) v13;
// CHECK-NEXT:   int64_t v25 = (int64_t) v24;
// CHECK-NEXT:   return v25;
// CHECK-NEXT: }

// CHECK: kernel void vectorAdd(device float* v1, device float* v2, device float* v3, device int32_t* v4, vec<int32_t, 3> v5 {{\[\[}}thread_position_in_grid{{\]\]}}, vec<int32_t, 3> v6 {{\[\[}}thread_position_in_threadgroup{{\]\]}}, vec<int32_t, 3> v7 {{\[\[}}threads_per_threadgroup{{\]\]}}, vec<int32_t, 3> v8 {{\[\[}}threadgroups_per_grid{{\]\]}}) {
// CHECK-NEXT:   int32_t v9 = 0;
// CHECK-NEXT:   int32_t v10 = v4[0;
// CHECK-NEXT:   int64_t v11 = _Z13get_global_idj(v9, v5, v6, v7, v8);
// CHECK-NEXT:   int32_t v12 = (int32_t) v11;
// CHECK-NEXT:   bool v13 = v12 <= v10;
// CHECK-NEXT:   if (v13) {
// CHECK-NEXT:     size_t v14 = (size_t) v12;
// CHECK-NEXT:     float v15 = v1[v14];
// CHECK-NEXT:     size_t v16 = (size_t) v12;
// CHECK-NEXT:     float v17 = v2[v16];
// CHECK-NEXT:     float v18 = v15 + v17;
// CHECK-NEXT:     size_t v19 = (size_t) v12;
// CHECK-NEXT:     v3[v19] = v18;
// CHECK-NEXT:     ;
// CHECK-NEXT:   }
// CHECK-NEXT:   return;
// CHECK-NEXT: }
