; NOTE: Assertions have been autogenerated by utils/update_test_checks.py UTC_ARGS: --version 5
; RUN: opt -S -passes=loop-fusion < %s | FileCheck %s

define void @hoist_preheader(i32 %N) {
; CHECK-LABEL: define void @hoist_preheader(
; CHECK-SAME: i32 [[N:%.*]]) {
; CHECK-NEXT:  [[PRE1:.*]]:
; CHECK-NEXT:    [[HOISTME:%.*]] = add i32 1, [[N]]
; CHECK-NEXT:    [[HOISTME2:%.*]] = add i32 1, [[HOISTME]]
; CHECK-NEXT:    br label %[[BODY1:.*]]
; CHECK:       [[BODY1]]:
; CHECK-NEXT:    [[I:%.*]] = phi i32 [ [[I_NEXT:%.*]], %[[BODY1]] ], [ 0, %[[PRE1]] ]
; CHECK-NEXT:    [[I2:%.*]] = phi i32 [ [[I_NEXT2:%.*]], %[[BODY1]] ], [ 0, %[[PRE1]] ]
; CHECK-NEXT:    [[I_NEXT]] = add i32 1, [[I]]
; CHECK-NEXT:    [[COND:%.*]] = icmp ne i32 [[I]], [[N]]
; CHECK-NEXT:    [[I_NEXT2]] = add i32 1, [[I2]]
; CHECK-NEXT:    [[COND2:%.*]] = icmp ne i32 [[I2]], [[N]]
; CHECK-NEXT:    br i1 [[COND2]], label %[[BODY1]], label %[[EXIT:.*]]
; CHECK:       [[EXIT]]:
; CHECK-NEXT:    ret void
;
pre1:
  br label %body1

body1:  ; preds = %pre1, %body1
  %i = phi i32 [%i_next, %body1], [0, %pre1]
  %i_next = add i32 1, %i
  %cond = icmp ne i32 %i, %N
  br i1 %cond, label %body1, label %pre2

pre2:
  %hoistme = add i32 1, %N
  %hoistme2 = add i32 1, %hoistme
  br label %body2

body2:  ; preds = %pre2, %body2
  %i2 = phi i32 [%i_next2, %body2], [0, %pre2]
  %i_next2 = add i32 1, %i2
  %cond2 = icmp ne i32 %i2, %N
  br i1 %cond2, label %body2, label %exit

exit:
  ret void
}
