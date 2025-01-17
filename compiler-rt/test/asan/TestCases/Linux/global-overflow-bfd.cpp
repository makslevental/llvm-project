// Test that gc-sections-friendly instrumentation of globals does not introduce
// false negatives with the BFD linker.
// RUN: %clangxx_asan -fuse-ld=bfd -Wl,-gc-sections -ffunction-sections -fdata-sections -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s

// Android does not use bfd.
// UNSUPPORTED: android

#include <string.h>
int main(int argc, char **argv) {
  static char XXX[10];
  static char YYY[10];
  static char ZZZ[10];
  memset(XXX, 0, 10);
  memset(YYY, 0, 10);
  memset(ZZZ, 0, 10);
  int res = YYY[argc * 10];  // BOOOM
  // CHECK: {{READ of size 1 at}}
  // CHECK: {{located 0 bytes after global variable}}
  res += XXX[argc] + ZZZ[argc];
  return res;
}
