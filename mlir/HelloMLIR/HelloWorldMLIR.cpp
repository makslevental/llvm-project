#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Analysis/CallGraph.h"
#include <cassert>
#include <iostream>
#include <mlir/Pass/PassPlugin.h>

using namespace mlir;

namespace {

struct BbqPass
    : public PassWrapper<BbqPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BbqPass)

  StringRef getArgument() const final { return "bbq"; }
  StringRef getDescription() const final {
    return "my bbq pass the description";
  }
  void runOnOperation() override {
    auto f = getOperation();
    std::cerr << "-- dump first op in module\n";
    f.getBodyRegion().getOps().begin()->dump();
  }
};

std::unique_ptr<mlir::Pass> createMyCallGraphPass() {
  return std::make_unique<BbqPass>();
}

} // namespace

namespace mlir {
namespace test {

void registerBbqPass() {
  std::cerr << "-- registering pass\n";
  registerPass(createMyCallGraphPass);
  const auto *previousPass = BbqPass::lookupPassInfo(
      "test-nvgpu-mmasync-f32-to-tf32-patterns");
  std::cerr << "-- check previously registered pass: " << previousPass->getPassArgument().str() << "\n";
  const auto *myPass = BbqPass::lookupPassInfo("bbq");
  std::cerr << "-- check this pass: " << myPass->getPassArgument().str() << "\n";
  std::cerr << "-- registered pass\n";
}

} // namespace test
} // namespace mlir

mlir::PassPluginLibraryInfo getPluginInfo() {
  auto callback = [](mlir::PassManager &) {
    test::registerBbqPass();
  };
  PassPluginLibraryInfo info{MLIR_PLUGIN_API_VERSION, "Bbq",
                             LLVM_VERSION_STRING, callback

  };
  return info;
}

extern "C" LLVM_ATTRIBUTE_WEAK ::mlir::PassPluginLibraryInfo
mlirGetPassPluginInfo() {
  return getPluginInfo();
}