; RUN: opt %s -safepoint-ir-verifier-print-only -verify-safepoint-ir -S 2>&1 | FileCheck %s

; CHECK:      Illegal use of unrelocated value found!
; CHECK-NEXT: Def:   %base_phi4 = phi ptr addrspace(1) [ %addr98.relocated, %not_zero146 ], [ %base_phi2, %bci_37-aload ], !is_base_value !0
; CHECK-NEXT: Use:   %safepoint_token = tail call token (i64, i32, ptr, i32, i32, ...) @llvm.experimental.gc.statepoint.p0(i64 0, i32 0, ptr elementtype(i32 ()) undef, i32 0, i32 0, i32 0, i32 0) [ "gc-live"(ptr addrspace(1) %base_phi1, ptr addrspace(1) %base_phi4, ptr addrspace(1) %relocated4, ptr addrspace(1) %relocated7) ]


%jObject = type { [8 x i8] }

declare ptr addrspace(1) @generate_obj1() #1

declare ptr addrspace(1) @generate_obj2() #1

declare ptr addrspace(1) @generate_obj3() #1

; Function Attrs: nounwind
define  void @test(ptr addrspace(1), ptr addrspace(1), i32, i1 %new_arg) #3 gc "statepoint-example" {
bci_0:
  %result608 = call ptr addrspace(1) @generate_obj3()
  br label %bci_37-aload

bci_37-aload:                                     ; preds = %not_zero179, %bci_0
  %base_phi = phi ptr addrspace(1) [ %base_phi1.relocated, %not_zero179 ], [ %result608, %bci_0 ], !is_base_value !0
  %base_phi2 = phi ptr addrspace(1) [ %base_phi3, %not_zero179 ], [ %result608, %bci_0 ], !is_base_value !0
  %relocated8 = phi ptr addrspace(1) [ %relocated7.relocated, %not_zero179 ], [ %result608, %bci_0 ]
  %tmp3 = getelementptr inbounds %jObject, ptr addrspace(1) %relocated8, i64 0, i32 0, i64 32
  br i1 %new_arg, label %not_zero179, label %not_zero146

not_zero146:                                      ; preds = %bci_37-aload
  %addr98.relocated = call ptr addrspace(1) @generate_obj2() #1
  %obj609.relocated = call ptr addrspace(1) @generate_obj1() #1
  br label %not_zero179

not_zero179:                                      ; preds = %not_zero146, %bci_37-aload
  %base_phi1 = phi ptr addrspace(1) [ %obj609.relocated, %not_zero146 ], [ %base_phi, %bci_37-aload ], !is_base_value !0
  %base_phi3 = phi ptr addrspace(1) [ %obj609.relocated, %not_zero146 ], [ %base_phi2, %bci_37-aload ], !is_base_value !0
  %relocated7 = phi ptr addrspace(1) [ %obj609.relocated, %not_zero146 ], [ %relocated8, %bci_37-aload ]
  %base_phi4 = phi ptr addrspace(1) [ %addr98.relocated, %not_zero146 ], [ %base_phi2, %bci_37-aload ], !is_base_value !0
  %relocated4 = phi ptr addrspace(1) [ %addr98.relocated, %not_zero146 ], [ %tmp3, %bci_37-aload ]
  %safepoint_token = tail call  token (i64, i32, ptr, i32, i32, ...) @llvm.experimental.gc.statepoint.p0(i64 0, i32 0, ptr elementtype(i32 ()) undef, i32 0, i32 0, i32 0, i32 0) ["gc-live"(ptr addrspace(1) %base_phi1, ptr addrspace(1) %base_phi4, ptr addrspace(1) %relocated4, ptr addrspace(1) %relocated7)]
  %tmp4 = call i32 @llvm.experimental.gc.result.i32(token %safepoint_token)
  %base_phi1.relocated = call coldcc ptr addrspace(1) @llvm.experimental.gc.relocate.p1(token %safepoint_token, i32 0, i32 0)
  %base_phi4.relocated = call coldcc ptr addrspace(1) @llvm.experimental.gc.relocate.p1(token %safepoint_token, i32 1, i32 1)
  %relocated4.relocated = call coldcc ptr addrspace(1) @llvm.experimental.gc.relocate.p1(token %safepoint_token, i32 1, i32 2)
  %relocated7.relocated = call coldcc ptr addrspace(1) @llvm.experimental.gc.relocate.p1(token %safepoint_token, i32 0, i32 3)
  br label %bci_37-aload
}

declare token @llvm.experimental.gc.statepoint.p0(i64, i32, ptr, i32, i32, ...)

; Function Attrs: nounwind
declare i32 @llvm.experimental.gc.result.i32(token) #4

; Function Attrs: nounwind
declare ptr addrspace(1) @llvm.experimental.gc.relocate.p1(token, i32, i32) #4

; Function Attrs: nounwind

attributes #0 = { noinline nounwind "gc-leaf-function"="true" }
attributes #1 = { "gc-leaf-function"="true" }
attributes #2 = { nounwind readonly "gc-leaf-function"="true" }
attributes #3 = { nounwind }
attributes #4 = { nounwind }

!0 = !{i32 1}
