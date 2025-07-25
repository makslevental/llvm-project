// RUN: not llvm-mc -triple=amdgcn -mcpu=bonaire %s 2>&1 | FileCheck %s --implicit-check-not=error: --strict-whitespace

//==============================================================================
// cache policy is not supported for SMRD instructions

s_load_dword s1, s[2:3], 0xfc glc slc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: cache policy is not supported for SMRD instructions
// CHECK-NEXT:{{^}}s_load_dword s1, s[2:3], 0xfc glc slc
// CHECK-NEXT:{{^}}                              ^

s_load_dword s1, s[2:3], 0xfc slc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: cache policy is not supported for SMRD instructions
// CHECK-NEXT:{{^}}s_load_dword s1, s[2:3], 0xfc slc
// CHECK-NEXT:{{^}}                              ^

//==============================================================================
// d16 modifier is not supported on this GPU

image_gather4 v[5:6], v1, s[8:15], s[12:15] dmask:0x1 d16
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: d16 modifier is not supported on this GPU
// CHECK-NEXT:{{^}}image_gather4 v[5:6], v1, s[8:15], s[12:15] dmask:0x1 d16
// CHECK-NEXT:{{^}}                                                      ^

//==============================================================================
// integer clamping is not supported on this GPU

v_add_co_u32 v84, s[4:5], v13, v31 clamp
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: integer clamping is not supported on this GPU
// CHECK-NEXT:{{^}}v_add_co_u32 v84, s[4:5], v13, v31 clamp
// CHECK-NEXT:{{^}}                                   ^

//==============================================================================
// literal operands are not supported

v_and_b32_e64 v0, 0.159154943091895317852646485335, v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: literal operands are not supported
// CHECK-NEXT:{{^}}v_and_b32_e64 v0, 0.159154943091895317852646485335, v1
// CHECK-NEXT:{{^}}                  ^

//==============================================================================
// cache policy is not supported for SMRD instructions

s_load_dword s5, s[2:3], glc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: cache policy is not supported for SMRD instructions
// CHECK-NEXT:{{^}}s_load_dword s5, s[2:3], glc
// CHECK-NEXT:{{^}}                         ^

//==============================================================================
// not a valid operand

v_alignbit_b32 v5, v1, v2, v3 op_sel:[1,1,1,1]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// CHECK-NEXT:{{^}}v_alignbit_b32 v5, v1, v2, v3 op_sel:[1,1,1,1]
// CHECK-NEXT:{{^}}                              ^

v_alignbyte_b32 v5, v1, v2, v3 op_sel:[1,1,1,1]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// CHECK-NEXT:{{^}}v_alignbyte_b32 v5, v1, v2, v3 op_sel:[1,1,1,1]
// CHECK-NEXT:{{^}}                               ^
