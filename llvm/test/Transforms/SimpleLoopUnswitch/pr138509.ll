; NOTE: Assertions have been autogenerated by utils/update_test_checks.py UTC_ARGS: --version 5
; RUN: opt -S -passes="loop-mssa(loop-simplifycfg,licm,loop-rotate,simple-loop-unswitch<nontrivial>)" < %s | FileCheck %s

@a = global i32 0, align 4
@b = global i32 0, align 4
@c = global i32 0, align 4
@d = global i32 0, align 4

define i32 @main() {
entry:
  br label %outer.loop.header

outer.loop.header:                                ; preds = %outer.loop.latch, %entry
  br i1 false, label %exit, label %outer.loop.body

outer.loop.body:                                  ; preds = %inner.loop.header, %outer.loop.header
  store i32 1, ptr @c, align 4
  %cmp = icmp sgt i32 0, -1
  br i1 %cmp, label %outer.loop.latch, label %exit

inner.loop.header:                                ; preds = %outer.loop.latch, %inner.loop.body
  %a_val = load i32, ptr @a, align 4
  %c_val = load i32, ptr @c, align 4
  %mul = mul nsw i32 %c_val, %a_val
  store i32 %mul, ptr @b, align 4
  %cmp2 = icmp sgt i32 %mul, -1
  br i1 %cmp2, label %inner.loop.body, label %outer.loop.body

inner.loop.body:                                  ; preds = %inner.loop.header
  %mul2 = mul nsw i32 %c_val, 3
  store i32 %mul2, ptr @c, align 4
  store i32 %c_val, ptr @d, align 4
  %mul3 = mul nsw i32 %c_val, %a_val
  %cmp3 = icmp sgt i32 %mul3, -1
  br i1 %cmp3, label %inner.loop.header, label %exit

outer.loop.latch:                                 ; preds = %outer.loop.body
  %d_val = load i32, ptr @d, align 4
  store i32 %d_val, ptr @b, align 4
  %cmp4 = icmp eq i32 %d_val, 0
  br i1 %cmp4, label %inner.loop.header, label %outer.loop.header

exit:                                             ; preds = %inner.loop.body, %outer.loop.body, %outer.loop.header
  ret i32 0
}

; CHECK: [[LOOP0:.*]] = distinct !{[[LOOP0]], [[META1:![0-9]+]]}
; CHECK: [[META1]] = !{!"llvm.loop.unswitch.nontrivial.disable"}
; CHECK: [[LOOP2:.*]] = distinct !{[[LOOP2]], [[META1]]}
