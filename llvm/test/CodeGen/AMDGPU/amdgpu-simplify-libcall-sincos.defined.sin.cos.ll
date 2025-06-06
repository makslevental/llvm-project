; NOTE: Assertions have been autogenerated by utils/update_test_checks.py UTC_ARGS: --version 2
; RUN: opt -S -mtriple=amdgcn-- -passes=amdgpu-simplifylib < %s | FileCheck %s

; sin and cos are already defined in the module but sincos isn't.

define float @_Z3sinf(float noundef %x) {
; CHECK-LABEL: define float @_Z3sinf
; CHECK-SAME: (float noundef [[X:%.*]]) {
; CHECK-NEXT:    [[RESULT:%.*]] = call float asm "
; CHECK-NEXT:    ret float [[RESULT]]
;
  %result = call float asm "; $0 = sin($1)","=v,v"(float %x)
  ret float %result
}

define float @_Z3cosf(float noundef %x) {
; CHECK-LABEL: define float @_Z3cosf
; CHECK-SAME: (float noundef [[X:%.*]]) {
; CHECK-NEXT:    [[RESULT:%.*]] = call float asm "
; CHECK-NEXT:    ret float [[RESULT]]
;
  %result = call float asm "; $0 = cos($1)","=v,v"(float %x)
  ret float %result
}

define <2 x float> @_Z3sinDv2_f(<2 x float> noundef %x) {
; CHECK-LABEL: define <2 x float> @_Z3sinDv2_f
; CHECK-SAME: (<2 x float> noundef [[X:%.*]]) {
; CHECK-NEXT:    [[RESULT:%.*]] = call <2 x float> asm "
; CHECK-NEXT:    ret <2 x float> [[RESULT]]
;
  %result = call <2 x float> asm "; $0 = sin($1)","=v,v"(<2 x float> %x)
  ret <2 x float> %result
}

define <2 x float> @_Z3cosDv2_f(<2 x float> noundef %x) {
; CHECK-LABEL: define <2 x float> @_Z3cosDv2_f
; CHECK-SAME: (<2 x float> noundef [[X:%.*]]) {
; CHECK-NEXT:    [[RESULT:%.*]] = call <2 x float> asm "
; CHECK-NEXT:    ret <2 x float> [[RESULT]]
;
  %result = call <2 x float> asm "; $0 = cos($1)","=v,v"(<2 x float> %x)
  ret <2 x float> %result
}

define void @sincos_f32(float noundef %x, ptr addrspace(1) nocapture noundef writeonly %sin_out, ptr addrspace(1) nocapture noundef writeonly %cos_out) {
; CHECK-LABEL: define void @sincos_f32
; CHECK-SAME: (float noundef [[X:%.*]], ptr addrspace(1) noundef writeonly captures(none) [[SIN_OUT:%.*]], ptr addrspace(1) noundef writeonly captures(none) [[COS_OUT:%.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CALL:%.*]] = tail call contract float @_Z3sinf(float noundef [[X]])
; CHECK-NEXT:    store float [[CALL]], ptr addrspace(1) [[SIN_OUT]], align 4
; CHECK-NEXT:    [[CALL1:%.*]] = tail call contract float @_Z3cosf(float noundef [[X]])
; CHECK-NEXT:    store float [[CALL1]], ptr addrspace(1) [[COS_OUT]], align 4
; CHECK-NEXT:    ret void
;
entry:
  %call = tail call contract float @_Z3sinf(float noundef %x)
  store float %call, ptr addrspace(1) %sin_out, align 4
  %call1 = tail call contract float @_Z3cosf(float noundef %x)
  store float %call1, ptr addrspace(1) %cos_out, align 4
  ret void
}

define void @sincos_f32_value_is_same_constantfp(ptr addrspace(1) nocapture noundef writeonly %sin_out, ptr addrspace(1) nocapture noundef writeonly %cos_out) {
; CHECK-LABEL: define void @sincos_f32_value_is_same_constantfp
; CHECK-SAME: (ptr addrspace(1) noundef writeonly captures(none) [[SIN_OUT:%.*]], ptr addrspace(1) noundef writeonly captures(none) [[COS_OUT:%.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CALL:%.*]] = tail call contract float @_Z3sinf(float 4.200000e+01)
; CHECK-NEXT:    store float [[CALL]], ptr addrspace(1) [[SIN_OUT]], align 4
; CHECK-NEXT:    [[CALL1:%.*]] = tail call contract float @_Z3cosf(float 4.200000e+01)
; CHECK-NEXT:    store float [[CALL1]], ptr addrspace(1) [[COS_OUT]], align 4
; CHECK-NEXT:    ret void
;
entry:
  %call = tail call contract float @_Z3sinf(float 42.0)
  store float %call, ptr addrspace(1) %sin_out, align 4
  %call1 = tail call contract float @_Z3cosf(float 42.0)
  store float %call1, ptr addrspace(1) %cos_out, align 4
  ret void
}

define void @sincos_v2f32(<2 x float> noundef %x, ptr addrspace(1) nocapture noundef writeonly %sin_out, ptr addrspace(1) nocapture noundef writeonly %cos_out) {
; GCN-LABEL: define void @sincos_v2f32
; GCN-SAME: (<2 x float> noundef [[X:%.*]], ptr addrspace(1) nocapture noundef writeonly [[SIN_OUT:%.*]], ptr addrspace(1) nocapture noundef writeonly [[COS_OUT:%.*]]) local_unnamed_addr {
; GCN-NEXT:  entry:
; GCN-NEXT:    [[CALL:%.*]] = tail call contract <2 x float> @_Z3sinDv2_f(<2 x float> noundef [[X]])
; GCN-NEXT:    store <2 x float> [[CALL]], ptr addrspace(1) [[SIN_OUT]], align 8
; GCN-NEXT:    [[CALL1:%.*]] = tail call contract <2 x float> @_Z3cosDv2_f(<2 x float> noundef [[X]])
; GCN-NEXT:    store <2 x float> [[CALL1]], ptr addrspace(1) [[COS_OUT]], align 8
; GCN-NEXT:    ret void
;
; CHECK-LABEL: define void @sincos_v2f32
; CHECK-SAME: (<2 x float> noundef [[X:%.*]], ptr addrspace(1) noundef writeonly captures(none) [[SIN_OUT:%.*]], ptr addrspace(1) noundef writeonly captures(none) [[COS_OUT:%.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CALL:%.*]] = tail call contract <2 x float> @_Z3sinDv2_f(<2 x float> noundef [[X]])
; CHECK-NEXT:    store <2 x float> [[CALL]], ptr addrspace(1) [[SIN_OUT]], align 8
; CHECK-NEXT:    [[CALL1:%.*]] = tail call contract <2 x float> @_Z3cosDv2_f(<2 x float> noundef [[X]])
; CHECK-NEXT:    store <2 x float> [[CALL1]], ptr addrspace(1) [[COS_OUT]], align 8
; CHECK-NEXT:    ret void
;
entry:
  %call = tail call contract <2 x float> @_Z3sinDv2_f(<2 x float> noundef %x)
  store <2 x float> %call, ptr addrspace(1) %sin_out, align 8
  %call1 = tail call contract <2 x float> @_Z3cosDv2_f(<2 x float> noundef %x)
  store <2 x float> %call1, ptr addrspace(1) %cos_out, align 8
  ret void
}
