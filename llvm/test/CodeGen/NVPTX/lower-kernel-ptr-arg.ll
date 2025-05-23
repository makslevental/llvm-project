; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Verify that both %input and %output are converted to global pointers and then
; addrspacecast'ed back to the original type.
define ptx_kernel void @kernel(ptr %input, ptr %output) {
; CHECK-LABEL: .visible .entry kernel(
; CHECK: cvta.to.global.u64
; CHECK: cvta.to.global.u64
  %1 = load float, ptr %input, align 4
; CHECK: ld.global.b32
  store float %1, ptr %output, align 4
; CHECK: st.global.b32
  ret void
}

define ptx_kernel void @kernel2(ptr addrspace(1) %input, ptr addrspace(1) %output) {
; CHECK-LABEL: .visible .entry kernel2(
; CHECK-NOT: cvta.to.global.u64
  %1 = load float, ptr addrspace(1) %input, align 4
; CHECK: ld.global.b32
  store float %1, ptr addrspace(1) %output, align 4
; CHECK: st.global.b32
  ret void
}

%struct.S = type { ptr, ptr }

define ptx_kernel void @ptr_in_byval_kernel(ptr byval(%struct.S) %input, ptr %output) {
; CHECK-LABEL: .visible .entry ptr_in_byval_kernel(
; CHECK: ld.param.b64 	%[[optr:rd.*]], [ptr_in_byval_kernel_param_1]
; CHECK: cvta.to.global.u64 %[[optr_g:.*]], %[[optr]];
; CHECK: ld.param.b64 	%[[iptr:rd.*]], [ptr_in_byval_kernel_param_0+8]
; CHECK: cvta.to.global.u64 %[[iptr_g:.*]], %[[iptr]];
  %b_ptr = getelementptr inbounds %struct.S, ptr %input, i64 0, i32 1
  %b = load ptr, ptr %b_ptr, align 8
  %v = load i32, ptr %b, align 4
; CHECK: ld.global.b32 %[[val:.*]], [%[[iptr_g]]]
  store i32 %v, ptr %output, align 4
; CHECK: st.global.b32 [%[[optr_g]]], %[[val]]
  ret void
}

; Regular functions lower byval arguments differently. We need to make
; sure that we're loading byval argument data using [symbol+offset].
; There's also no assumption that all pointers within are in global space.
define void @ptr_in_byval_func(ptr byval(%struct.S) %input, ptr %output) {
; CHECK-LABEL: .visible .func ptr_in_byval_func(
; CHECK: ld.param.b64 	%[[optr:rd.*]], [ptr_in_byval_func_param_1]
; CHECK: ld.param.b64 	%[[iptr:rd.*]], [ptr_in_byval_func_param_0+8]
  %b_ptr = getelementptr inbounds %struct.S, ptr %input, i64 0, i32 1
  %b = load ptr, ptr %b_ptr, align 8
  %v = load i32, ptr %b, align 4
; CHECK: ld.b32 %[[val:.*]], [%[[iptr]]]
  store i32 %v, ptr %output, align 4
; CHECK: st.b32 [%[[optr]]], %[[val]]
  ret void
}

