; RUN: llc -mtriple=amdgcn < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -mtriple=amdgcn -mcpu=tonga -mattr=-flat-for-global < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s

; FUNC-LABEL: {{^}}fpext_f32_to_f64:
; SI: v_cvt_f64_f32_e32 {{v\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
define amdgpu_kernel void @fpext_f32_to_f64(ptr addrspace(1) %out, float %in) {
  %result = fpext float %in to double
  store double %result, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}fpext_v2f32_to_v2f64:
; SI: v_cvt_f64_f32_e32
; SI: v_cvt_f64_f32_e32
define amdgpu_kernel void @fpext_v2f32_to_v2f64(ptr addrspace(1) %out, <2 x float> %in) {
  %result = fpext <2 x float> %in to <2 x double>
  store <2 x double> %result, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}fpext_v3f32_to_v3f64:
; SI: v_cvt_f64_f32_e32
; SI: v_cvt_f64_f32_e32
; SI: v_cvt_f64_f32_e32
define amdgpu_kernel void @fpext_v3f32_to_v3f64(ptr addrspace(1) %out, <3 x float> %in) {
  %result = fpext <3 x float> %in to <3 x double>
  store <3 x double> %result, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}fpext_v4f32_to_v4f64:
; SI: v_cvt_f64_f32_e32
; SI: v_cvt_f64_f32_e32
; SI: v_cvt_f64_f32_e32
; SI: v_cvt_f64_f32_e32
define amdgpu_kernel void @fpext_v4f32_to_v4f64(ptr addrspace(1) %out, <4 x float> %in) {
  %result = fpext <4 x float> %in to <4 x double>
  store <4 x double> %result, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}fpext_v8f32_to_v8f64:
; SI: v_cvt_f64_f32_e32
; SI: v_cvt_f64_f32_e32
; SI: v_cvt_f64_f32_e32
; SI: v_cvt_f64_f32_e32
; SI: v_cvt_f64_f32_e32
; SI: v_cvt_f64_f32_e32
; SI: v_cvt_f64_f32_e32
; SI: v_cvt_f64_f32_e32
define amdgpu_kernel void @fpext_v8f32_to_v8f64(ptr addrspace(1) %out, <8 x float> %in) {
  %result = fpext <8 x float> %in to <8 x double>
  store <8 x double> %result, ptr addrspace(1) %out
  ret void
}
