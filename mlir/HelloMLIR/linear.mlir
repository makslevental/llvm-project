module attributes {torch.debug_module_name = "Linear"} {
  func.func @forward(%arg0: memref<1x10xf32>) -> index {
    %c0 = arith.constant 0 : index
    return %c0 : index
  }
}
