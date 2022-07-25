// REQUIRES: spirv-tools
// RUN: lcloc --target=vulkan-spv %s -o %t.spv
// RUN: spirv-dis %t.spv | FileCheck %s

__kernel void vectorAdd(__global float *a, __global float *b, __global float *c, const unsigned int n) {
  int id = get_global_id(0);

  if (id < n) {
    c[id] = a[id] + b[id];
  }
}
// CHECK: ; SPIR-V
// CHECK-NEXT:  ; Version: 1.0
// CHECK-NEXT: ; Generator: Khronos; 22
// CHECK-NEXT: ; Bound: 80
// CHECK-NEXT: ; Schema: 0
// CHECK-NEXT:                OpCapability Int64
// CHECK-NEXT:                OpCapability Shader
// CHECK-NEXT:                OpExtension "SPV_KHR_storage_buffer_storage_class"
// CHECK-NEXT:                OpMemoryModel Logical GLSL450
// CHECK-NEXT:                OpEntryPoint GLCompute %vectorAdd "vectorAdd"
// CHECK-NEXT:                OpExecutionMode %vectorAdd LocalSize 1 1 1
// CHECK-NEXT:                OpName %__builtin_var_GlobalInvocationId__ "__builtin_var_GlobalInvocationId__"
// CHECK-NEXT:                OpName %_Z13get_global_idj "_Z13get_global_idj"
// CHECK-NEXT:                OpName %vectorAdd_arg_0 "vectorAdd_arg_0"
// CHECK-NEXT:                OpName %vectorAdd_arg_1 "vectorAdd_arg_1"
// CHECK-NEXT:                OpName %vectorAdd_arg_2 "vectorAdd_arg_2"
// CHECK-NEXT:                OpName %vectorAdd_arg_3 "vectorAdd_arg_3"
// CHECK-NEXT:                OpName %vectorAdd "vectorAdd"
// CHECK-NEXT:                OpDecorate %__builtin_var_GlobalInvocationId__ BuiltIn GlobalInvocationId
// CHECK-NEXT:                OpDecorate %_runtimearr_float ArrayStride 4
// CHECK-NEXT:                OpMemberDecorate %_struct_44 0 Offset 0
// CHECK-NEXT:                OpDecorate %_struct_44 Block
// CHECK-NEXT:                OpDecorate %vectorAdd_arg_0 Binding 0
// CHECK-NEXT:                OpDecorate %vectorAdd_arg_0 DescriptorSet 0
// CHECK-NEXT:                OpDecorate %vectorAdd_arg_1 Binding 1
// CHECK-NEXT:                OpDecorate %vectorAdd_arg_1 DescriptorSet 0
// CHECK-NEXT:                OpDecorate %vectorAdd_arg_2 Binding 2
// CHECK-NEXT:                OpDecorate %vectorAdd_arg_2 DescriptorSet 0
// CHECK-NEXT:                OpMemberDecorate %_struct_51 0 Offset 0
// CHECK-NEXT:                OpDecorate %_struct_51 Block
// CHECK-NEXT:                OpDecorate %vectorAdd_arg_3 Binding 3
// CHECK-NEXT:                OpDecorate %vectorAdd_arg_3 DescriptorSet 0
// CHECK-NEXT:                OpDecorate %_runtimearr_float_0 ArrayStride 1
// CHECK-NEXT:                OpMemberDecorate %_struct_61 0 Offset 0
// CHECK-NEXT:                OpDecorate %_struct_61 Block
// CHECK-NEXT:        %uint = OpTypeInt 32 0
// CHECK-NEXT:      %v3uint = OpTypeVector %uint 3
// CHECK-NEXT: %_ptr_Input_v3uint = OpTypePointer Input %v3uint
// CHECK-NEXT: %__builtin_var_GlobalInvocationId__ = OpVariable %_ptr_Input_v3uint Input
// CHECK-NEXT:       %ulong = OpTypeInt 64 0
// CHECK-NEXT:           %5 = OpTypeFunction %ulong %uint
// CHECK-NEXT:      %uint_0 = OpConstant %uint 0
// CHECK-NEXT:      %uint_2 = OpConstant %uint 2
// CHECK-NEXT:      %uint_1 = OpConstant %uint 1
// CHECK-NEXT:        %bool = OpTypeBool
// CHECK-NEXT: %_ptr_Function_uint = OpTypePointer Function %uint
// CHECK-NEXT:       %float = OpTypeFloat 32
// CHECK-NEXT: %_runtimearr_float = OpTypeRuntimeArray %float
// CHECK-NEXT:  %_struct_44 = OpTypeStruct %_runtimearr_float
// CHECK-NEXT: %_ptr_StorageBuffer__struct_44 = OpTypePointer StorageBuffer %_struct_44
// CHECK-NEXT: %vectorAdd_arg_0 = OpVariable %_ptr_StorageBuffer__struct_44 StorageBuffer
// CHECK-NEXT: %vectorAdd_arg_1 = OpVariable %_ptr_StorageBuffer__struct_44 StorageBuffer
// CHECK-NEXT: %vectorAdd_arg_2 = OpVariable %_ptr_StorageBuffer__struct_44 StorageBuffer
// CHECK-NEXT:  %_struct_51 = OpTypeStruct %uint
// CHECK-NEXT: %_ptr_StorageBuffer__struct_51 = OpTypePointer StorageBuffer %_struct_51
// CHECK-NEXT: %vectorAdd_arg_3 = OpVariable %_ptr_StorageBuffer__struct_51 StorageBuffer
// CHECK-NEXT:        %void = OpTypeVoid
// CHECK-NEXT:          %53 = OpTypeFunction %void
// CHECK-NEXT: %_ptr_StorageBuffer_uint = OpTypePointer StorageBuffer %uint
// CHECK-NEXT: %_runtimearr_float_0 = OpTypeRuntimeArray %float
// CHECK-NEXT:  %_struct_61 = OpTypeStruct %_runtimearr_float_0
// CHECK-NEXT: %_ptr_StorageBuffer__struct_61 = OpTypePointer StorageBuffer %_struct_61
// CHECK-NEXT:     %ulong_0 = OpConstant %ulong 0
// CHECK-NEXT: %_ptr_StorageBuffer_float = OpTypePointer StorageBuffer %float
// CHECK-NEXT: %_Z13get_global_idj = OpFunction %ulong None %5
// CHECK-NEXT:           %8 = OpFunctionParameter %uint
// CHECK-NEXT:           %9 = OpLabel
// CHECK-NEXT:          %16 = OpVariable %_ptr_Function_uint Function
// CHECK-NEXT:          %24 = OpVariable %_ptr_Function_uint Function
// CHECK-NEXT:          %32 = OpVariable %_ptr_Function_uint Function
// CHECK-NEXT:          %14 = OpIEqual %bool %8 %uint_0
// CHECK-NEXT:                OpBranch %17
// CHECK-NEXT:          %17 = OpLabel
// CHECK-NEXT:                OpSelectionMerge %20 None
// CHECK-NEXT:                OpBranchConditional %14 %18 %19
// CHECK-NEXT:          %18 = OpLabel
// CHECK-NEXT:          %21 = OpLoad %v3uint %__builtin_var_GlobalInvocationId__
// CHECK-NEXT:          %22 = OpCompositeExtract %uint %21 0
// CHECK-NEXT:                OpStore %16 %22
// CHECK-NEXT:                OpBranch %20
// CHECK-NEXT:          %19 = OpLabel
// CHECK-NEXT:          %23 = OpIEqual %bool %8 %uint_1
// CHECK-NEXT:                OpBranch %25
// CHECK-NEXT:          %25 = OpLabel
// CHECK-NEXT:                OpSelectionMerge %28 None
// CHECK-NEXT:                OpBranchConditional %23 %26 %27
// CHECK-NEXT:          %26 = OpLabel
// CHECK-NEXT:          %29 = OpLoad %v3uint %__builtin_var_GlobalInvocationId__
// CHECK-NEXT:          %30 = OpCompositeExtract %uint %29 1
// CHECK-NEXT:                OpStore %24 %30
// CHECK-NEXT:                OpBranch %28
// CHECK-NEXT:          %27 = OpLabel
// CHECK-NEXT:          %31 = OpIEqual %bool %8 %uint_2
// CHECK-NEXT:                OpBranch %33
// CHECK-NEXT:          %33 = OpLabel
// CHECK-NEXT:                OpSelectionMerge %36 None
// CHECK-NEXT:                OpBranchConditional %31 %34 %35
// CHECK-NEXT:          %34 = OpLabel
// CHECK-NEXT:          %37 = OpLoad %v3uint %__builtin_var_GlobalInvocationId__
// CHECK-NEXT:          %38 = OpCompositeExtract %uint %37 2
// CHECK-NEXT:                OpStore %32 %38
// CHECK-NEXT:                OpBranch %36
// CHECK-NEXT:          %35 = OpLabel
// CHECK-NEXT:                OpStore %32 %uint_0
// CHECK-NEXT:                OpBranch %36
// CHECK-NEXT:          %36 = OpLabel
// CHECK-NEXT:          %39 = OpLoad %uint %32
// CHECK-NEXT:                OpStore %24 %39
// CHECK-NEXT:                OpBranch %28
// CHECK-NEXT:          %28 = OpLabel
// CHECK-NEXT:          %40 = OpLoad %uint %24
// CHECK-NEXT:                OpStore %16 %40
// CHECK-NEXT:                OpBranch %20
// CHECK-NEXT:          %20 = OpLabel
// CHECK-NEXT:          %41 = OpLoad %uint %16
// CHECK-NEXT:          %42 = OpSConvert %ulong %41
// CHECK-NEXT:                OpReturnValue %42
// CHECK-NEXT:                OpFunctionEnd
// CHECK-NEXT:   %vectorAdd = OpFunction %void None %53
// CHECK-NEXT:          %56 = OpLabel
// CHECK-NEXT:          %58 = OpAccessChain %_ptr_StorageBuffer_uint %vectorAdd_arg_3 %uint_0
// CHECK-NEXT:          %59 = OpLoad %uint %58
// CHECK-NEXT:          %63 = OpBitcast %_ptr_StorageBuffer__struct_61 %vectorAdd_arg_2
// CHECK-NEXT:          %64 = OpBitcast %_ptr_StorageBuffer__struct_61 %vectorAdd_arg_1
// CHECK-NEXT:          %65 = OpBitcast %_ptr_StorageBuffer__struct_61 %vectorAdd_arg_0
// CHECK-NEXT:          %66 = OpFunctionCall %ulong %_Z13get_global_idj %uint_0
// CHECK-NEXT:          %67 = OpSConvert %uint %66
// CHECK-NEXT:          %68 = OpULessThan %bool %67 %59
// CHECK-NEXT:                OpBranch %69
// CHECK-NEXT:          %69 = OpLabel
// CHECK-NEXT:                OpSelectionMerge %71 None
// CHECK-NEXT:                OpBranchConditional %68 %70 %71
// CHECK-NEXT:          %70 = OpLabel
// CHECK-NEXT:          %74 = OpAccessChain %_ptr_StorageBuffer_float %65 %ulong_0 %67
// CHECK-NEXT:          %75 = OpLoad %float %74
// CHECK-NEXT:          %76 = OpAccessChain %_ptr_StorageBuffer_float %64 %ulong_0 %67
// CHECK-NEXT:          %77 = OpLoad %float %76
// CHECK-NEXT:          %78 = OpFAdd %float %75 %77
// CHECK-NEXT:          %79 = OpAccessChain %_ptr_StorageBuffer_float %63 %ulong_0 %67
// CHECK-NEXT:                OpStore %79 %78
// CHECK-NEXT:                OpBranch %71
// CHECK-NEXT:          %71 = OpLabel
// CHECK-NEXT:                OpReturn
// CHECK-NEXT:                OpFunctionEnd
