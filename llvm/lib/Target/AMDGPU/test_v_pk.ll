; test_v_pk.ll -mtriple=amdgcn -mcpu=gfx942 -o -
; test_v_pk.ll -mattr=-packed-fp32-ops -mtriple=amdgcn -mcpu=gfx942 -o -
; test_v_pk.ll -mtriple=amdgcn -mcpu=gfx942 -stop-after=finalize-isel -o -
; test_v_pk.mir -mtriple=amdgcn -mcpu=gfx942 -x mir -filetype=null -o -

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define amdgpu_kernel void @add_kernel(ptr addrspace(1) nocapture readonly %0, ptr addrspace(1) nocapture readonly %1, ptr addrspace(1) nocapture writeonly %2, i32 %3) local_unnamed_addr #0 {
  %5 = tail call i32 @llvm.amdgcn.workgroup.id.x()
  %6 = shl i32 %5, 10
  %7 = tail call i32 @llvm.amdgcn.workitem.id.x()
  %8 = shl i32 %7, 2
  %9 = and i32 %8, 1020
  %10 = or disjoint i32 %9, %6
  %11 = icmp slt i32 %10, %3
  br i1 %11, label %.critedge, label %.critedge2

.critedge:                                        ; preds = %4
  %12 = or disjoint i32 %10, 3
  %13 = or disjoint i32 %10, 2
  %14 = or disjoint i32 %10, 1
  %15 = sext i32 %10 to i64
  %16 = getelementptr float, ptr addrspace(1) %0, i64 %15
  %17 = addrspacecast ptr addrspace(1) %16 to ptr
  %18 = load float, ptr %17, align 16
  %19 = getelementptr inbounds i8, ptr %17, i64 4
  %20 = load float, ptr %19, align 4
  
  %v_100 = insertelement <2 x float> undef, float %18, i32 0
  %v_102 = insertelement <2 x float> %v_100, float %20, i32 1

  %21 = getelementptr inbounds i8, ptr %17, i64 8
  %22 = load float, ptr %21, align 8
  %23 = getelementptr inbounds i8, ptr %17, i64 12
  %24 = load float, ptr %23, align 4
  
  %v_200 = insertelement <2 x float> undef, float %22, i32 0
  %v_202 = insertelement <2 x float> %v_200, float %24, i32 1  
  
  %25 = getelementptr float, ptr addrspace(1) %1, i64 %15
  %26 = addrspacecast ptr addrspace(1) %25 to ptr
  %27 = sext i32 %12 to i64
  %28 = getelementptr float, ptr addrspace(1) %2, i64 %27
  %29 = sext i32 %13 to i64
  %30 = getelementptr float, ptr addrspace(1) %2, i64 %29
  %31 = sext i32 %14 to i64
  %32 = getelementptr float, ptr addrspace(1) %2, i64 %31
  %33 = getelementptr inbounds i8, ptr %26, i64 12
  %34 = load float, ptr %33, align 4

  %36 = getelementptr inbounds i8, ptr %26, i64 8
  %37 = load float, ptr %36, align 8
  
  %v_300 = insertelement <2 x float> undef, float %34, i32 0
  %v_302 = insertelement <2 x float> %v_300, float %37, i32 1  
  
  %39 = getelementptr inbounds i8, ptr %26, i64 4
  %40 = load float, ptr %39, align 4
  %42 = load float, ptr %26, align 16

  %v_400 = insertelement <2 x float> undef, float %40, i32 0
  %v_402 = insertelement <2 x float> %v_400, float %42, i32 1  

  %v_500 = fadd <2 x float> %v_102, %v_402
  ; %v_501 = fadd <2 x float> %v_202, %v_302
  ; tail call void @llvm.amdgcn.iglp.opt(i32 4)

  ; %v_45 = extractelement <2 x float> %v_501, i32 1
  ; %v_32 = extractelement <2 x float> %v_501, i32 0
  %v_30 = extractelement <2 x float> %v_500, i32 1
  %v_28 = extractelement <2 x float> %v_500, i32 0

  %i_44 = sext i32 %10 to i64
  %p_45 = getelementptr float, ptr addrspace(1) %2, i64 %i_44
  store float %v_28, ptr addrspace(1) %p_45, align 4

  ; %i_31 = sext i32 %14 to i64
  ; %p_32 = getelementptr float, ptr addrspace(1) %2, i64 %i_31
  ; store float %v_32, ptr addrspace(1) %p_32, align 4

  %i_29 = sext i32 %13 to i64
  %p_30 = getelementptr float, ptr addrspace(1) %2, i64 %i_29
  store float %v_30, ptr addrspace(1) %p_30, align 4

  %i_27 = sext i32 %12 to i64
  %p_28 = getelementptr float, ptr addrspace(1) %2, i64 %i_27
  store float %v_28, ptr addrspace(1) %p_28, align 4

  br label %.critedge2

.critedge2:                                       ; preds = %4, %.critedge
  ret void
}
