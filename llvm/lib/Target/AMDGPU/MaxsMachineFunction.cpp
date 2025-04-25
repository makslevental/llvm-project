//===---------------------- MaxsMachineFunctionPass.cpp -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SIMachineFunctionInfo.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/PassRegistry.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "maxsmachinefunction"

namespace {

struct MaxsMachineFunction : MachineFunctionPass {
  static char ID;

  MaxsMachineFunction() : MachineFunctionPass(ID) {
    // initializeMaxsMachineFunctionPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &au) const override;

  bool runOnMachineFunction(MachineFunction &MF) override;

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::IsSSA);
  }
};
} // end anonymous namespace

char MaxsMachineFunction::ID = 0;
char &llvm::MaxsMachineFunctionID = MaxsMachineFunction::ID;

static void initializeMaxsMachineFunctionPassOnce(PassRegistry &Registry) {
  PassInfo *PI = new PassInfo(
      "MaxsMachineFunction", "maxsmachinefunction", &MaxsMachineFunction::ID,
      PassInfo::NormalCtor_t(callDefaultCtor<MaxsMachineFunction>), false,
      false);
  Registry.registerPass(*PI, true);
}

static llvm::once_flag InitializeMaxsMachineFunctionPassFlag;

void llvm::initializeMaxsMachineFunctionPass(PassRegistry &Registry) {
  llvm::call_once(InitializeMaxsMachineFunctionPassFlag,
                  initializeMaxsMachineFunctionPassOnce, std::ref(Registry));
}

FunctionPass *createMaxsMachineFunctionPass() {
  return new MaxsMachineFunction();
}

void MaxsMachineFunction::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  AU.addPreserved<AAResultsWrapperPass>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

std::optional<MachineInstr *> findNthUser(MachineInstr &MI,
                                          MachineRegisterInfo *MRI,
                                          const Register &CRReg,
                                          unsigned N = 1) {
  MachineBasicBlock::iterator I = MI;
  unsigned Idx = 0;
  for (MachineBasicBlock::iterator EL = MI.getParent()->end(); I != EL; ++I) {
    for (MachineRegisterInfo::use_instr_iterator
             J = MRI->use_instr_begin(CRReg),
             JE = MRI->use_instr_end();
         J != JE; ++J)
      if (&*J == &*I) {
        Idx++;
      }
    if (Idx == N) {
      return &*I;
    }
  }
  return {};
}

bool MaxsMachineFunction::runOnMachineFunction(MachineFunction &MF) {

  LLVM_DEBUG(dbgs() << "********** MaxsMachineFunction **********\n"
                    << "********** Function: " << MF.getName() << '\n');

  bool Changed = false;

  MachineRegisterInfo *MRI = &MF.getRegInfo();
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  const SIInstrInfo *TII = ST.getInstrInfo();

  SmallVector<MachineInstr *> toRemove;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (MI.getOpcode() == AMDGPU::V_PK_ADD_F32) {
        MachineOperand &OldDest = MI.getOperand(0);
        MachineOperand &Lhs = MI.getOperand(2);
        MachineOperand &Rhs = MI.getOperand(4);

        auto lhsLow = MachineOperand::CreateReg(
            Lhs.getReg(), Lhs.isDef(), Lhs.isImplicit(), Lhs.isKill(),
            Lhs.isDead(), Lhs.isUndef(), Lhs.isEarlyClobber(), AMDGPU::sub0,
            Lhs.isDebug(), Lhs.isInternalRead());

        auto rhsLow = MachineOperand::CreateReg(
            Rhs.getReg(), Rhs.isDef(), Rhs.isImplicit(), Rhs.isKill(),
            Rhs.isDead(), Rhs.isUndef(), Rhs.isEarlyClobber(), AMDGPU::sub0,
            Rhs.isDebug(), Rhs.isInternalRead());

        Register DstReg1 = MRI->createVirtualRegister(&AMDGPU::VGPR_32RegClass);

        MachineInstrBuilder MIB =
            BuildMI(MBB, MI, {}, TII->get(AMDGPU::V_ADD_F32_e32), DstReg1)
                .add({lhsLow, rhsLow});
        if (MI.getFlag(MachineInstr::MIFlag::NoFPExcept))
          (void)MIB.setMIFlag(MachineInstr::MIFlag::NoFPExcept);

        auto lhsHigh = MachineOperand::CreateReg(
            Lhs.getReg(), Lhs.isDef(), Lhs.isImplicit(), Lhs.isKill(),
            Lhs.isDead(), Lhs.isUndef(), Lhs.isEarlyClobber(), AMDGPU::sub1,
            Lhs.isDebug(), Lhs.isInternalRead());

        auto rhsHigh = MachineOperand::CreateReg(
            Rhs.getReg(), Rhs.isDef(), Rhs.isImplicit(), Rhs.isKill(),
            Rhs.isDead(), Rhs.isUndef(), Rhs.isEarlyClobber(), AMDGPU::sub1,
            Rhs.isDebug(), Rhs.isInternalRead());

        Register DstReg2 = MRI->createVirtualRegister(&AMDGPU::VGPR_32RegClass);

        MIB = BuildMI(MBB, MI, {}, TII->get(AMDGPU::V_ADD_F32_e32), DstReg2)
                  .add({lhsHigh, rhsHigh});
        if (MI.getFlag(MachineInstr::MIFlag::NoFPExcept))
          (void)MIB.setMIFlag(MachineInstr::MIFlag::NoFPExcept);

        Register nextOperand;
        std::optional<MachineInstr *> I;
        if (I = findNthUser(MI, MRI, OldDest.getReg()); *I) {
          nextOperand = (*I)->getOperand(0).getReg();
          (*I)->getOperand(1).ChangeToRegister(DstReg2, /*isDef*/ false);
        }

        Register DstReg3 =
            MRI->createVirtualRegister(&AMDGPU::VReg_64_Align2RegClass);

        auto reqSeq = BuildMI(MBB, *I, {}, TII->get(AMDGPU::REG_SEQUENCE))
                          .addDef(DstReg3)
                          .addUse(DstReg1)
                          .addImm(AMDGPU::sub0)
                          .addUse(DstReg2)
                          .addImm(AMDGPU::sub1)
                          .getInstr();

        if (auto I = findNthUser(MI, MRI, nextOperand, 2)) {
          reqSeq->getOperand(0).dump();
          (*I)->getOperand(1).ChangeToRegister(reqSeq->getOperand(0).getReg(),
                                               /*isDef*/ false);
        }

        toRemove.push_back(&MI);
        Changed = true;
      }
    }
  }
  for (auto remove : toRemove)
    remove->eraseFromParent();
  return Changed;
}
