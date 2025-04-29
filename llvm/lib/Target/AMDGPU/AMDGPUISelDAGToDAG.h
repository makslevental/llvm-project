//===-- AMDGPUISelDAGToDAG.h - A dag to dag inst selector for AMDGPU ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//==-----------------------------------------------------------------------===//
//
/// \file
/// Defines an instruction selector for the AMDGPU target.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUISELDAGTODAG_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUISELDAGTODAG_H

#include "GCNSubtarget.h"
#include "SIMachineFunctionInfo.h"
#include "SIModeRegisterDefaults.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

static inline bool getConstantValue(SDValue N, uint32_t &Out) {
  // This is only used for packed vectors, where using 0 for undef should
  // always be good.
  if (N.isUndef()) {
    Out = 0;
    return true;
  }

  if (const ConstantSDNode *C = dyn_cast<ConstantSDNode>(N)) {
    Out = C->getAPIntValue().getSExtValue();
    return true;
  }

  if (const ConstantFPSDNode *C = dyn_cast<ConstantFPSDNode>(N)) {
    Out = C->getValueAPF().bitcastToAPInt().getSExtValue();
    return true;
  }

  return false;
}

// TODO: Handle undef as zero
static inline SDNode *packConstantV2I16(const SDNode *N, SelectionDAG &DAG) {
  assert(N->getOpcode() == ISD::BUILD_VECTOR && N->getNumOperands() == 2);
  uint32_t LHSVal, RHSVal;
  if (getConstantValue(N->getOperand(0), LHSVal) &&
      getConstantValue(N->getOperand(1), RHSVal)) {
    SDLoc SL(N);
    uint32_t K = (LHSVal & 0xffff) | (RHSVal << 16);
    return DAG.getMachineNode(AMDGPU::S_MOV_B32, SL, N->getValueType(0),
                              DAG.getTargetConstant(K, SL, MVT::i32));
  }

  return nullptr;
}

/// AMDGPU specific code to select AMDGPU machine instructions for
/// SelectionDAG operations.
class AMDGPUDAGToDAGISel : public SelectionDAGISel {
  // Subtarget - Keep a pointer to the AMDGPU Subtarget around so that we can
  // make the right decision when generating code for different targets.
  const GCNSubtarget *Subtarget;

  // Default FP mode for the current function.
  SIModeRegisterDefaults Mode;

  // Instructions that will be lowered with a final instruction that zeros the
  // high result bits.
  bool fp16SrcZerosHighBits(unsigned Opc) const;

public:
  AMDGPUDAGToDAGISel() = delete;

  explicit AMDGPUDAGToDAGISel(TargetMachine &TM, CodeGenOptLevel OptLevel);

  bool runOnMachineFunction(MachineFunction &MF) override;
  bool matchLoadD16FromBuildVector(SDNode *N) const;
  void PreprocessISelDAG() override;
  void Select(SDNode *N) override;
  void PostprocessISelDAG() override;

protected:
  void SelectBuildVector(SDNode *N, unsigned RegClassID);
  void SelectVectorShuffle(SDNode *N);

private:
  std::pair<SDValue, SDValue> foldFrameIndex(SDValue N) const;

  bool isInlineImmediate(const SDNode *N) const;

  bool isInlineImmediate(const APInt &Imm) const {
    return Subtarget->getInstrInfo()->isInlineConstant(Imm);
  }

  bool isInlineImmediate(const APFloat &Imm) const {
    return Subtarget->getInstrInfo()->isInlineConstant(Imm);
  }

  bool isVGPRImm(const SDNode *N) const;
  bool isUniformLoad(const SDNode *N) const;
  bool isUniformBr(const SDNode *N) const;

  // Returns true if ISD::AND SDNode `N`'s masking of the shift amount operand's
  // `ShAmtBits` bits is unneeded.
  bool isUnneededShiftMask(const SDNode *N, unsigned ShAmtBits) const;

  bool isBaseWithConstantOffset64(SDValue Addr, SDValue &LHS,
                                  SDValue &RHS) const;

  MachineSDNode *buildSMovImm64(SDLoc &DL, uint64_t Val, EVT VT) const;

  SDNode *glueCopyToOp(SDNode *N, SDValue NewChain, SDValue Glue) const;
  SDNode *glueCopyToM0(SDNode *N, SDValue Val) const;
  SDNode *glueCopyToM0LDSInit(SDNode *N) const;

  const TargetRegisterClass *getOperandRegClass(SDNode *N, unsigned OpNo) const;
  virtual bool SelectADDRVTX_READ(SDValue Addr, SDValue &Base, SDValue &Offset);
  virtual bool SelectADDRIndirect(SDValue Addr, SDValue &Base, SDValue &Offset);
  bool isDSOffsetLegal(SDValue Base, unsigned Offset) const;
  bool isDSOffset2Legal(SDValue Base, unsigned Offset0, unsigned Offset1,
                        unsigned Size) const;

  bool isFlatScratchBaseLegal(SDValue Addr) const;
  bool isFlatScratchBaseLegalSV(SDValue Addr) const;
  bool isFlatScratchBaseLegalSVImm(SDValue Addr) const;
  bool isSOffsetLegalWithImmOffset(SDValue *SOffset, bool Imm32Only,
                                   bool IsBuffer, int64_t ImmOffset = 0) const;

  bool SelectDS1Addr1Offset(SDValue Ptr, SDValue &Base, SDValue &Offset) const;
  bool SelectDS64Bit4ByteAligned(SDValue Ptr, SDValue &Base, SDValue &Offset0,
                                 SDValue &Offset1) const;
  bool SelectDS128Bit8ByteAligned(SDValue Ptr, SDValue &Base, SDValue &Offset0,
                                  SDValue &Offset1) const;
  bool SelectDSReadWrite2(SDValue Ptr, SDValue &Base, SDValue &Offset0,
                          SDValue &Offset1, unsigned Size) const;
  bool SelectMUBUF(SDValue Addr, SDValue &SRsrc, SDValue &VAddr,
                   SDValue &SOffset, SDValue &Offset, SDValue &Offen,
                   SDValue &Idxen, SDValue &Addr64) const;
  bool SelectMUBUFAddr64(SDValue Addr, SDValue &SRsrc, SDValue &VAddr,
                         SDValue &SOffset, SDValue &Offset) const;
  bool SelectMUBUFScratchOffen(SDNode *Parent, SDValue Addr, SDValue &RSrc,
                               SDValue &VAddr, SDValue &SOffset,
                               SDValue &ImmOffset) const;
  bool SelectMUBUFScratchOffset(SDNode *Parent, SDValue Addr, SDValue &SRsrc,
                                SDValue &Soffset, SDValue &Offset) const;

  bool SelectMUBUFOffset(SDValue Addr, SDValue &SRsrc, SDValue &Soffset,
                         SDValue &Offset) const;
  bool SelectBUFSOffset(SDValue Addr, SDValue &SOffset) const;
  bool MaxsComplexPatternPackedFP(SDNode *Parent, SDValue Addr, SDValue &SOffset) const;

  bool SelectFlatOffsetImpl(SDNode *N, SDValue Addr, SDValue &VAddr,
                            SDValue &Offset, uint64_t FlatVariant) const;
  bool SelectFlatOffset(SDNode *N, SDValue Addr, SDValue &VAddr,
                        SDValue &Offset) const;
  bool SelectGlobalOffset(SDNode *N, SDValue Addr, SDValue &VAddr,
                          SDValue &Offset) const;
  bool SelectScratchOffset(SDNode *N, SDValue Addr, SDValue &VAddr,
                           SDValue &Offset) const;
  bool SelectGlobalSAddr(SDNode *N, SDValue Addr, SDValue &SAddr,
                         SDValue &VOffset, SDValue &Offset) const;
  bool SelectScratchSAddr(SDNode *N, SDValue Addr, SDValue &SAddr,
                          SDValue &Offset) const;
  bool checkFlatScratchSVSSwizzleBug(SDValue VAddr, SDValue SAddr,
                                     uint64_t ImmOffset) const;
  bool SelectScratchSVAddr(SDNode *N, SDValue Addr, SDValue &VAddr,
                           SDValue &SAddr, SDValue &Offset) const;

  bool SelectSMRDOffset(SDValue ByteOffsetNode, SDValue *SOffset,
                        SDValue *Offset, bool Imm32Only = false,
                        bool IsBuffer = false, bool HasSOffset = false,
                        int64_t ImmOffset = 0) const;
  SDValue Expand32BitAddress(SDValue Addr) const;
  bool SelectSMRDBaseOffset(SDValue Addr, SDValue &SBase, SDValue *SOffset,
                            SDValue *Offset, bool Imm32Only = false,
                            bool IsBuffer = false, bool HasSOffset = false,
                            int64_t ImmOffset = 0) const;
  bool SelectSMRD(SDValue Addr, SDValue &SBase, SDValue *SOffset,
                  SDValue *Offset, bool Imm32Only = false) const;
  bool SelectSMRDImm(SDValue Addr, SDValue &SBase, SDValue &Offset) const;
  bool SelectSMRDImm32(SDValue Addr, SDValue &SBase, SDValue &Offset) const;
  bool SelectSMRDSgpr(SDValue Addr, SDValue &SBase, SDValue &SOffset) const;
  bool SelectSMRDSgprImm(SDValue Addr, SDValue &SBase, SDValue &SOffset,
                         SDValue &Offset) const;
  bool SelectSMRDBufferImm(SDValue N, SDValue &Offset) const;
  bool SelectSMRDBufferImm32(SDValue N, SDValue &Offset) const;
  bool SelectSMRDBufferSgprImm(SDValue N, SDValue &SOffset,
                               SDValue &Offset) const;
  bool SelectSMRDPrefetchImm(SDValue Addr, SDValue &SBase,
                             SDValue &Offset) const;
  bool SelectMOVRELOffset(SDValue Index, SDValue &Base, SDValue &Offset) const;

  bool SelectVOP3ModsImpl(SDValue In, SDValue &Src, unsigned &SrcMods,
                          bool IsCanonicalizing = true,
                          bool AllowAbs = true) const;
  bool SelectVOP3Mods(SDValue In, SDValue &Src, SDValue &SrcMods) const;
  bool SelectVOP3ModsNonCanonicalizing(SDValue In, SDValue &Src,
                                       SDValue &SrcMods) const;
  bool SelectVOP3BMods(SDValue In, SDValue &Src, SDValue &SrcMods) const;
  bool SelectVOP3NoMods(SDValue In, SDValue &Src) const;
  bool SelectVOP3Mods0(SDValue In, SDValue &Src, SDValue &SrcMods,
                       SDValue &Clamp, SDValue &Omod) const;
  bool SelectVOP3BMods0(SDValue In, SDValue &Src, SDValue &SrcMods,
                        SDValue &Clamp, SDValue &Omod) const;
  bool SelectVOP3NoMods0(SDValue In, SDValue &Src, SDValue &SrcMods,
                         SDValue &Clamp, SDValue &Omod) const;

  bool SelectVINTERPModsImpl(SDValue In, SDValue &Src, SDValue &SrcMods,
                             bool OpSel) const;
  bool SelectVINTERPMods(SDValue In, SDValue &Src, SDValue &SrcMods) const;
  bool SelectVINTERPModsHi(SDValue In, SDValue &Src, SDValue &SrcMods) const;

  bool SelectVOP3OMods(SDValue In, SDValue &Src, SDValue &Clamp,
                       SDValue &Omod) const;

  bool SelectVOP3PMods(SDValue In, SDValue &Src, SDValue &SrcMods,
                       bool IsDOT = false) const;
  bool SelectVOP3PModsDOT(SDValue In, SDValue &Src, SDValue &SrcMods) const;

  bool SelectVOP3PModsNeg(SDValue In, SDValue &Src) const;
  bool SelectWMMAOpSelVOP3PMods(SDValue In, SDValue &Src) const;

  bool SelectWMMAModsF32NegAbs(SDValue In, SDValue &Src,
                               SDValue &SrcMods) const;
  bool SelectWMMAModsF16Neg(SDValue In, SDValue &Src, SDValue &SrcMods) const;
  bool SelectWMMAModsF16NegAbs(SDValue In, SDValue &Src,
                               SDValue &SrcMods) const;
  bool SelectWMMAVISrc(SDValue In, SDValue &Src) const;

  bool SelectSWMMACIndex8(SDValue In, SDValue &Src, SDValue &IndexKey) const;
  bool SelectSWMMACIndex16(SDValue In, SDValue &Src, SDValue &IndexKey) const;

  bool SelectVOP3OpSel(SDValue In, SDValue &Src, SDValue &SrcMods) const;

  bool SelectVOP3OpSelMods(SDValue In, SDValue &Src, SDValue &SrcMods) const;
  bool SelectVOP3PMadMixModsImpl(SDValue In, SDValue &Src,
                                 unsigned &Mods) const;
  bool SelectVOP3PMadMixModsExt(SDValue In, SDValue &Src,
                                SDValue &SrcMods) const;
  bool SelectVOP3PMadMixMods(SDValue In, SDValue &Src, SDValue &SrcMods) const;

  bool SelectBITOP3(SDValue In, SDValue &Src0, SDValue &Src1, SDValue &Src2,
                   SDValue &Tbl) const;

  SDValue getHi16Elt(SDValue In) const;

  SDValue getMaterializedScalarImm32(int64_t Val, const SDLoc &DL) const;

  void SelectADD_SUB_I64(SDNode *N);
  void SelectAddcSubb(SDNode *N);
  void SelectUADDO_USUBO(SDNode *N);
  void SelectDIV_SCALE(SDNode *N);
  void SelectMAD_64_32(SDNode *N);
  void SelectMUL_LOHI(SDNode *N);
  void SelectFMA_W_CHAIN(SDNode *N);
  void SelectFMUL_W_CHAIN(SDNode *N);
  SDNode *getBFE32(bool IsSigned, const SDLoc &DL, SDValue Val, uint32_t Offset,
                   uint32_t Width);
  void SelectS_BFEFromShifts(SDNode *N);
  void SelectS_BFE(SDNode *N);
  bool isCBranchSCC(const SDNode *N) const;
  void SelectBRCOND(SDNode *N);
  void SelectFMAD_FMA(SDNode *N);
  void SelectFP_EXTEND(SDNode *N);
  void SelectDSAppendConsume(SDNode *N, unsigned IntrID);
  void SelectDSBvhStackIntrinsic(SDNode *N, unsigned IntrID);
  void SelectDS_GWS(SDNode *N, unsigned IntrID);
  void SelectInterpP1F16(SDNode *N);
  void SelectINTRINSIC_W_CHAIN(SDNode *N);
  void SelectINTRINSIC_WO_CHAIN(SDNode *N);
  void SelectINTRINSIC_VOID(SDNode *N);
  void SelectWAVE_ADDRESS(SDNode *N);
  void SelectSTACKRESTORE(SDNode *N);

  bool MaxsComplexPatternPackedFP(SDNode *N, SelectionDAG *CurDAG) const;

protected:
  // Include the pieces autogenerated from the target description.
#include "AMDGPUGenDAGISel.inc"

void MySelectCode(SDNode *N) {
  // Some target values are emitted as 2 bytes, TARGET_VAL handles
  // this. Coverage indexes are emitted as 4 bytes,
  // COVERAGE_IDX_VAL handles this.
  #define TARGET_VAL(X) X & 255, unsigned(X) >> 8
  #define COVERAGE_IDX_VAL(X) X & 255, (unsigned(X) >> 8) & 255, (unsigned(X) >> 16) & 255, (unsigned(X) >> 24) & 255
  static const unsigned char MatcherTable[] = {
// /*528829*/ /*SwitchOpcode*/ 120|128,4/*632*/, TARGET_VAL(ISD::FADD),// ->529465
/*528833*/  OPC_Scope, 76, /*->528911*/ // 4 children in Scope
/*528835*/   OPC_MoveChild0,
/*528836*/   OPC_CheckOpcode, TARGET_VAL(ISD::INTRINSIC_WO_CHAIN),
/*528839*/   OPC_CheckChild0Integer, 104|128,35/*4584*/,
/*528842*/   OPC_RecordChild1, // #0 = $VOP3NoMods:src0
/*528843*/   OPC_RecordChild2, // #1 = $VOP3NoMods:src1
/*528844*/   OPC_MoveParent,
/*528845*/   OPC_RecordChild1, // #2 = $VOP3NoMods:src2
/*528846*/   OPC_Scope, 35, /*->528883*/ // 2 children in Scope
/*528848*/    OPC_CheckPatternPredicate, 153, // (Subtarget->hasMadMacF32Insts()) && (MF->getInfo<SIMachineFunctionInfo>()->getMode().FP32Denormals == DenormalMode::getPreserveSign()) && (Subtarget->getGeneration() == AMDGPUSubtarget::SOUTHERN_ISLANDS ||Subtarget->getGeneration() == AMDGPUSubtarget::SEA_ISLANDS ||Subtarget->getGeneration() == AMDGPUSubtarget::GFX10)
/*528850*/    OPC_CheckComplexPat, /*CP*/22, /*#*/0, // SelectVOP3NoMods:$ #3
/*528853*/    OPC_CheckComplexPat, /*CP*/22, /*#*/1, // SelectVOP3NoMods:$ #4
/*528856*/    OPC_CheckComplexPat, /*CP*/22, /*#*/2, // SelectVOP3NoMods:$ #5
/*528859*/    OPC_EmitInteger32, 0,  // 0 #6
/*528861*/    OPC_EmitInteger32, 0,  // 0 #7
/*528863*/    OPC_EmitInteger32, 0,  // 0 #8
/*528865*/    OPC_EmitInteger, /*MVT::i1*/2, 0,  // 0 #9
/*528868*/    OPC_EmitInteger32, 0,  // 0 #10
/*528870*/    OPC_MorphNodeTo1None, TARGET_VAL(AMDGPU::V_MAC_LEGACY_F32_e64),
                  /*MVT::f32*/12, 8/*#Ops*/, 6, 3, 7, 4, 8, 5, 9, 10,
              // Src: (fadd:{ *:[f32] } (intrinsic_wo_chain:{ *:[f32] } 2292:{ *:[iPTR] }, (VOP3NoMods:{ *:[f32] } f32:{ *:[f32] }:$src0), (VOP3NoMods:{ *:[f32] } f32:{ *:[f32] }:$src1)), (VOP3NoMods:{ *:[f32] } f32:{ *:[f32] }:$src2)) - Complexity = 38
              // Dst: (V_MAC_LEGACY_F32_e64:{ *:[f32] } 0:{ *:[i32] }, ?:{ *:[f32] }:$src0, 0:{ *:[i32] }, ?:{ *:[f32] }:$src1, 0:{ *:[i32] }, ?:{ *:[f32] }:$src2, 0:{ *:[i1] }, 0:{ *:[i32] })
/*528883*/   /*Scope*/ 26, /*->528910*/
/*528884*/    OPC_CheckPatternPredicate, 154, // (Subtarget->hasMadMacF32Insts()) && (MF->getInfo<SIMachineFunctionInfo>()->getMode().FP32Denormals == DenormalMode::getPreserveSign())
/*528886*/    OPC_CheckComplexPat0, /*#*/0, // SelectVOP3Mods:$ #3 #4
/*528888*/    OPC_CheckComplexPat0, /*#*/1, // SelectVOP3Mods:$ #5 #6
/*528890*/    OPC_CheckComplexPat0, /*#*/2, // SelectVOP3Mods:$ #7 #8
/*528892*/    OPC_EmitInteger, /*MVT::i1*/2, 0,  // 0 #9
/*528895*/    OPC_EmitInteger32, 0,  // 0 #10
/*528897*/    OPC_MorphNodeTo1None, TARGET_VAL(AMDGPU::V_MAD_LEGACY_F32_e64),
                  /*MVT::f32*/12, 8/*#Ops*/, 4, 3, 6, 5, 8, 7, 9, 10,
              // Src: (fadd:{ *:[f32] } (intrinsic_wo_chain:{ *:[f32] } 2292:{ *:[iPTR] }, (VOP3Mods:{ *:[f32] } f32:{ *:[f32] }:$src0, i32:{ *:[i32] }:$src0_mod), (VOP3Mods:{ *:[f32] } f32:{ *:[f32] }:$src1, i32:{ *:[i32] }:$src1_mod)), (VOP3Mods:{ *:[f32] } f32:{ *:[f32] }:$src2, i32:{ *:[i32] }:$src2_mod)) - Complexity = 38
              // Dst: (V_MAD_LEGACY_F32_e64:{ *:[f32] } ?:{ *:[i32] }:$src0_mod, ?:{ *:[f32] }:$src0, ?:{ *:[i32] }:$src1_mod, ?:{ *:[f32] }:$src1, ?:{ *:[i32] }:$src2_mod, ?:{ *:[f32] }:$src2, 0:{ *:[i1] }, 0:{ *:[i32] })
/*528910*/   0, /*End of Scope*/
/*528911*/  /*Scope*/ 76, /*->528988*/
/*528912*/   OPC_RecordChild0, // #0 = $VOP3NoMods:src2
/*528913*/   OPC_MoveChild1,
/*528914*/   OPC_CheckOpcode, TARGET_VAL(ISD::INTRINSIC_WO_CHAIN),
/*528917*/   OPC_CheckChild0Integer, 104|128,35/*4584*/,
/*528920*/   OPC_RecordChild1, // #1 = $VOP3NoMods:src0
/*528921*/   OPC_RecordChild2, // #2 = $VOP3NoMods:src1
/*528922*/   OPC_MoveParent,
/*528923*/   OPC_Scope, 35, /*->528960*/ // 2 children in Scope
/*528925*/    OPC_CheckPatternPredicate, 153, // (Subtarget->hasMadMacF32Insts()) && (MF->getInfo<SIMachineFunctionInfo>()->getMode().FP32Denormals == DenormalMode::getPreserveSign()) && (Subtarget->getGeneration() == AMDGPUSubtarget::SOUTHERN_ISLANDS ||Subtarget->getGeneration() == AMDGPUSubtarget::SEA_ISLANDS ||Subtarget->getGeneration() == AMDGPUSubtarget::GFX10)
/*528927*/    OPC_CheckComplexPat, /*CP*/22, /*#*/0, // SelectVOP3NoMods:$ #3
/*528930*/    OPC_CheckComplexPat, /*CP*/22, /*#*/1, // SelectVOP3NoMods:$ #4
/*528933*/    OPC_CheckComplexPat, /*CP*/22, /*#*/2, // SelectVOP3NoMods:$ #5
/*528936*/    OPC_EmitInteger32, 0,  // 0 #6
/*528938*/    OPC_EmitInteger32, 0,  // 0 #7
/*528940*/    OPC_EmitInteger32, 0,  // 0 #8
/*528942*/    OPC_EmitInteger, /*MVT::i1*/2, 0,  // 0 #9
/*528945*/    OPC_EmitInteger32, 0,  // 0 #10
/*528947*/    OPC_MorphNodeTo1None, TARGET_VAL(AMDGPU::V_MAC_LEGACY_F32_e64),
                  /*MVT::f32*/12, 8/*#Ops*/, 6, 4, 7, 5, 8, 3, 9, 10,
              // Src: (fadd:{ *:[f32] } (VOP3NoMods:{ *:[f32] } f32:{ *:[f32] }:$src2), (intrinsic_wo_chain:{ *:[f32] } 2292:{ *:[iPTR] }, (VOP3NoMods:{ *:[f32] } f32:{ *:[f32] }:$src0), (VOP3NoMods:{ *:[f32] } f32:{ *:[f32] }:$src1))) - Complexity = 38
              // Dst: (V_MAC_LEGACY_F32_e64:{ *:[f32] } 0:{ *:[i32] }, ?:{ *:[f32] }:$src0, 0:{ *:[i32] }, ?:{ *:[f32] }:$src1, 0:{ *:[i32] }, ?:{ *:[f32] }:$src2, 0:{ *:[i1] }, 0:{ *:[i32] })
/*528960*/   /*Scope*/ 26, /*->528987*/
/*528961*/    OPC_CheckPatternPredicate, 154, // (Subtarget->hasMadMacF32Insts()) && (MF->getInfo<SIMachineFunctionInfo>()->getMode().FP32Denormals == DenormalMode::getPreserveSign())
/*528963*/    OPC_CheckComplexPat0, /*#*/0, // SelectVOP3Mods:$ #3 #4
/*528965*/    OPC_CheckComplexPat0, /*#*/1, // SelectVOP3Mods:$ #5 #6
/*528967*/    OPC_CheckComplexPat0, /*#*/2, // SelectVOP3Mods:$ #7 #8
/*528969*/    OPC_EmitInteger, /*MVT::i1*/2, 0,  // 0 #9
/*528972*/    OPC_EmitInteger32, 0,  // 0 #10
/*528974*/    OPC_MorphNodeTo1None, TARGET_VAL(AMDGPU::V_MAD_LEGACY_F32_e64),
                  /*MVT::f32*/12, 8/*#Ops*/, 6, 5, 8, 7, 4, 3, 9, 10,
              // Src: (fadd:{ *:[f32] } (VOP3Mods:{ *:[f32] } f32:{ *:[f32] }:$src2, i32:{ *:[i32] }:$src2_mod), (intrinsic_wo_chain:{ *:[f32] } 2292:{ *:[iPTR] }, (VOP3Mods:{ *:[f32] } f32:{ *:[f32] }:$src0, i32:{ *:[i32] }:$src0_mod), (VOP3Mods:{ *:[f32] } f32:{ *:[f32] }:$src1, i32:{ *:[i32] }:$src1_mod))) - Complexity = 38
              // Dst: (V_MAD_LEGACY_F32_e64:{ *:[f32] } ?:{ *:[i32] }:$src0_mod, ?:{ *:[f32] }:$src0, ?:{ *:[i32] }:$src1_mod, ?:{ *:[f32] }:$src1, ?:{ *:[i32] }:$src2_mod, ?:{ *:[f32] }:$src2, 0:{ *:[i1] }, 0:{ *:[i32] })
/*528987*/   0, /*End of Scope*/
/*528988*/  /*Scope*/ 75, /*->529064*/
/*528989*/   OPC_MoveChild0,
/*528990*/   OPC_CheckOpcode, TARGET_VAL(AMDGPUISD::FMUL_LEGACY),
/*528993*/   OPC_RecordChild0, // #0 = $VOP3NoMods:src0
/*528994*/   OPC_RecordChild1, // #1 = $VOP3NoMods:src1
/*528995*/   OPC_MoveParent,
/*528996*/   OPC_RecordChild1, // #2 = $VOP3NoMods:src2
/*528997*/   OPC_CheckType, /*MVT::f32*/12,
/*528999*/   OPC_Scope, 35, /*->529036*/ // 2 children in Scope
/*529001*/    OPC_CheckPatternPredicate, 153, // (Subtarget->hasMadMacF32Insts()) && (MF->getInfo<SIMachineFunctionInfo>()->getMode().FP32Denormals == DenormalMode::getPreserveSign()) && (Subtarget->getGeneration() == AMDGPUSubtarget::SOUTHERN_ISLANDS ||Subtarget->getGeneration() == AMDGPUSubtarget::SEA_ISLANDS ||Subtarget->getGeneration() == AMDGPUSubtarget::GFX10)
/*529003*/    OPC_CheckComplexPat, /*CP*/22, /*#*/0, // SelectVOP3NoMods:$ #3
/*529006*/    OPC_CheckComplexPat, /*CP*/22, /*#*/1, // SelectVOP3NoMods:$ #4
/*529009*/    OPC_CheckComplexPat, /*CP*/22, /*#*/2, // SelectVOP3NoMods:$ #5
/*529012*/    OPC_EmitInteger32, 0,  // 0 #6
/*529014*/    OPC_EmitInteger32, 0,  // 0 #7
/*529016*/    OPC_EmitInteger32, 0,  // 0 #8
/*529018*/    OPC_EmitInteger, /*MVT::i1*/2, 0,  // 0 #9
/*529021*/    OPC_EmitInteger32, 0,  // 0 #10
/*529023*/    OPC_MorphNodeTo1None, TARGET_VAL(AMDGPU::V_MAC_LEGACY_F32_e64),
                  /*MVT::f32*/12, 8/*#Ops*/, 6, 3, 7, 4, 8, 5, 9, 10,
              // Src: (fadd:{ *:[f32] } (AMDGPUfmul_legacy_impl:{ *:[f32] } (VOP3NoMods:{ *:[f32] } f32:{ *:[f32] }:$src0), (VOP3NoMods:{ *:[f32] } f32:{ *:[f32] }:$src1)), (VOP3NoMods:{ *:[f32] } f32:{ *:[f32] }:$src2)) - Complexity = 33
              // Dst: (V_MAC_LEGACY_F32_e64:{ *:[f32] } 0:{ *:[i32] }, ?:{ *:[f32] }:$src0, 0:{ *:[i32] }, ?:{ *:[f32] }:$src1, 0:{ *:[i32] }, ?:{ *:[f32] }:$src2, 0:{ *:[i1] }, 0:{ *:[i32] })
/*529036*/   /*Scope*/ 26, /*->529063*/
/*529037*/    OPC_CheckPatternPredicate, 154, // (Subtarget->hasMadMacF32Insts()) && (MF->getInfo<SIMachineFunctionInfo>()->getMode().FP32Denormals == DenormalMode::getPreserveSign())
/*529039*/    OPC_CheckComplexPat0, /*#*/0, // SelectVOP3Mods:$ #3 #4
/*529041*/    OPC_CheckComplexPat0, /*#*/1, // SelectVOP3Mods:$ #5 #6
/*529043*/    OPC_CheckComplexPat0, /*#*/2, // SelectVOP3Mods:$ #7 #8
/*529045*/    OPC_EmitInteger, /*MVT::i1*/2, 0,  // 0 #9
/*529048*/    OPC_EmitInteger32, 0,  // 0 #10
/*529050*/    OPC_MorphNodeTo1None, TARGET_VAL(AMDGPU::V_MAD_LEGACY_F32_e64),
                  /*MVT::f32*/12, 8/*#Ops*/, 4, 3, 6, 5, 8, 7, 9, 10,
              // Src: (fadd:{ *:[f32] } (AMDGPUfmul_legacy_impl:{ *:[f32] } (VOP3Mods:{ *:[f32] } f32:{ *:[f32] }:$src0, i32:{ *:[i32] }:$src0_mod), (VOP3Mods:{ *:[f32] } f32:{ *:[f32] }:$src1, i32:{ *:[i32] }:$src1_mod)), (VOP3Mods:{ *:[f32] } f32:{ *:[f32] }:$src2, i32:{ *:[i32] }:$src2_mod)) - Complexity = 33
              // Dst: (V_MAD_LEGACY_F32_e64:{ *:[f32] } ?:{ *:[i32] }:$src0_mod, ?:{ *:[f32] }:$src0, ?:{ *:[i32] }:$src1_mod, ?:{ *:[f32] }:$src1, ?:{ *:[i32] }:$src2_mod, ?:{ *:[f32] }:$src2, 0:{ *:[i1] }, 0:{ *:[i32] })
/*529063*/   0, /*End of Scope*/
/*529064*/  /*Scope*/ 14|128,3/*398*/, /*->529464*/
/*529066*/   OPC_RecordChild0, // #0 = $VOP3NoMods:src2
/*529067*/   OPC_Scope, 74, /*->529143*/ // 2 children in Scope
/*529069*/    OPC_MoveChild1,
/*529070*/    OPC_CheckOpcode, TARGET_VAL(AMDGPUISD::FMUL_LEGACY),
/*529073*/    OPC_RecordChild0, // #1 = $VOP3NoMods:src0
/*529074*/    OPC_RecordChild1, // #2 = $VOP3NoMods:src1
/*529075*/    OPC_MoveParent,
/*529076*/    OPC_CheckType, /*MVT::f32*/12,
/*529078*/    OPC_Scope, 35, /*->529115*/ // 2 children in Scope
/*529080*/     OPC_CheckPatternPredicate, 153, // (Subtarget->hasMadMacF32Insts()) && (MF->getInfo<SIMachineFunctionInfo>()->getMode().FP32Denormals == DenormalMode::getPreserveSign()) && (Subtarget->getGeneration() == AMDGPUSubtarget::SOUTHERN_ISLANDS ||Subtarget->getGeneration() == AMDGPUSubtarget::SEA_ISLANDS ||Subtarget->getGeneration() == AMDGPUSubtarget::GFX10)
/*529082*/     OPC_CheckComplexPat, /*CP*/22, /*#*/0, // SelectVOP3NoMods:$ #3
/*529085*/     OPC_CheckComplexPat, /*CP*/22, /*#*/1, // SelectVOP3NoMods:$ #4
/*529088*/     OPC_CheckComplexPat, /*CP*/22, /*#*/2, // SelectVOP3NoMods:$ #5
/*529091*/     OPC_EmitInteger32, 0,  // 0 #6
/*529093*/     OPC_EmitInteger32, 0,  // 0 #7
/*529095*/     OPC_EmitInteger32, 0,  // 0 #8
/*529097*/     OPC_EmitInteger, /*MVT::i1*/2, 0,  // 0 #9
/*529100*/     OPC_EmitInteger32, 0,  // 0 #10
/*529102*/     OPC_MorphNodeTo1None, TARGET_VAL(AMDGPU::V_MAC_LEGACY_F32_e64),
                   /*MVT::f32*/12, 8/*#Ops*/, 6, 4, 7, 5, 8, 3, 9, 10,
               // Src: (fadd:{ *:[f32] } (VOP3NoMods:{ *:[f32] } f32:{ *:[f32] }:$src2), (AMDGPUfmul_legacy_impl:{ *:[f32] } (VOP3NoMods:{ *:[f32] } f32:{ *:[f32] }:$src0), (VOP3NoMods:{ *:[f32] } f32:{ *:[f32] }:$src1))) - Complexity = 33
               // Dst: (V_MAC_LEGACY_F32_e64:{ *:[f32] } 0:{ *:[i32] }, ?:{ *:[f32] }:$src0, 0:{ *:[i32] }, ?:{ *:[f32] }:$src1, 0:{ *:[i32] }, ?:{ *:[f32] }:$src2, 0:{ *:[i1] }, 0:{ *:[i32] })
/*529115*/    /*Scope*/ 26, /*->529142*/
/*529116*/     OPC_CheckPatternPredicate, 154, // (Subtarget->hasMadMacF32Insts()) && (MF->getInfo<SIMachineFunctionInfo>()->getMode().FP32Denormals == DenormalMode::getPreserveSign())
/*529118*/     OPC_CheckComplexPat0, /*#*/0, // SelectVOP3Mods:$ #3 #4
/*529120*/     OPC_CheckComplexPat0, /*#*/1, // SelectVOP3Mods:$ #5 #6
/*529122*/     OPC_CheckComplexPat0, /*#*/2, // SelectVOP3Mods:$ #7 #8
/*529124*/     OPC_EmitInteger, /*MVT::i1*/2, 0,  // 0 #9
/*529127*/     OPC_EmitInteger32, 0,  // 0 #10
/*529129*/     OPC_MorphNodeTo1None, TARGET_VAL(AMDGPU::V_MAD_LEGACY_F32_e64),
                   /*MVT::f32*/12, 8/*#Ops*/, 6, 5, 8, 7, 4, 3, 9, 10,
               // Src: (fadd:{ *:[f32] } (VOP3Mods:{ *:[f32] } f32:{ *:[f32] }:$src2, i32:{ *:[i32] }:$src2_mod), (AMDGPUfmul_legacy_impl:{ *:[f32] } (VOP3Mods:{ *:[f32] } f32:{ *:[f32] }:$src0, i32:{ *:[i32] }:$src0_mod), (VOP3Mods:{ *:[f32] } f32:{ *:[f32] }:$src1, i32:{ *:[i32] }:$src1_mod))) - Complexity = 33
               // Dst: (V_MAD_LEGACY_F32_e64:{ *:[f32] } ?:{ *:[i32] }:$src0_mod, ?:{ *:[f32] }:$src0, ?:{ *:[i32] }:$src1_mod, ?:{ *:[f32] }:$src1, ?:{ *:[i32] }:$src2_mod, ?:{ *:[f32] }:$src2, 0:{ *:[i1] }, 0:{ *:[i32] })
/*529142*/    0, /*End of Scope*/
/*529143*/   /*Scope*/ 62|128,2/*318*/, /*->529463*/
/*529145*/    OPC_RecordChild1, // #1 = $src1
/*529146*/    OPC_Scope, 25, /*->529173*/ // 6 children in Scope
/*529148*/     OPC_CheckPredicate3,  // Predicate_anonymous_13768
/*529149*/     OPC_SwitchType /*2 cases */, 9, /*MVT::f32*/12, // ->529161
/*529152*/      OPC_CheckPatternPredicate, 22, // (Subtarget->hasSALUFloatInsts())
/*529154*/      OPC_MorphNodeTo1None, TARGET_VAL(AMDGPU::S_ADD_F32),
                    /*MVT::f32*/12, 2/*#Ops*/, 0, 1,
                // Src: (fadd:{ *:[f32] } SSrc_f32:{ *:[f32] }:$src0, SSrc_f32:{ *:[f32] }:$src1)<<P:Predicate_anonymous_13768>> - Complexity = 4
                // Dst: (S_ADD_F32:{ *:[f32] } SSrc_f32:{ *:[f32] }:$src0, SSrc_f32:{ *:[f32] }:$src1)
/*529161*/     /*SwitchType*/ 9, /*MVT::f16*/11, // ->529172
/*529163*/      OPC_CheckPatternPredicate, 22, // (Subtarget->hasSALUFloatInsts())
/*529165*/      OPC_MorphNodeTo1None, TARGET_VAL(AMDGPU::S_ADD_F16),
                    /*MVT::f16*/11, 2/*#Ops*/, 0, 1,
                // Src: (fadd:{ *:[f16] } SSrc_f16:{ *:[f16] }:$src0, SSrc_f16:{ *:[f16] }:$src1)<<P:Predicate_anonymous_13768>> - Complexity = 4
                // Dst: (S_ADD_F16:{ *:[f16] } SSrc_f16:{ *:[f16] }:$src0, SSrc_f16:{ *:[f16] }:$src1)
/*529172*/     0, // EndSwitchType
/*529173*/    /*Scope*/ 36, /*->529210*/
/*529174*/     OPC_CheckType, /*MVT::f32*/12,
/*529176*/     OPC_Scope, 15, /*->529193*/ // 2 children in Scope
/*529178*/      OPC_CheckComplexPat2, /*#*/0, // SelectVOP3Mods0:$ #2 #3 #4 #5
/*529180*/      OPC_CheckComplexPat0, /*#*/1, // SelectVOP3Mods:$ #6 #7
/*529182*/      OPC_MorphNodeTo1None, TARGET_VAL(AMDGPU::V_ADD_F32_e64),
                    /*MVT::f32*/12, 6/*#Ops*/, 3, 2, 7, 6, 4, 5,
                // Src: (fadd:{ *:[f32] } (VOP3Mods0:{ *:[f32] } f32:{ *:[f32] }:$src0, i32:{ *:[i32] }:$src0_modifiers, i1:{ *:[i1] }:$clamp, i32:{ *:[i32] }:$omod), (VOP3Mods:{ *:[f32] } f32:{ *:[f32] }:$src1, i32:{ *:[i32] }:$src1_modifiers)) - Complexity = -973
                // Dst: (V_ADD_F32_e64:{ *:[f32] } i32:{ *:[i32] }:$src0_modifiers, f32:{ *:[f32] }:$src0, i32:{ *:[i32] }:$src1_modifiers, f32:{ *:[f32] }:$src1, i1:{ *:[i1] }:$clamp, i32:{ *:[i32] }:$omod)
/*529193*/     /*Scope*/ 15, /*->529209*/
/*529194*/      OPC_CheckComplexPat0, /*#*/0, // SelectVOP3Mods:$ #2 #3
/*529196*/      OPC_CheckComplexPat2, /*#*/1, // SelectVOP3Mods0:$ #4 #5 #6 #7
/*529198*/      OPC_MorphNodeTo1None, TARGET_VAL(AMDGPU::V_ADD_F32_e64),
                    /*MVT::f32*/12, 6/*#Ops*/, 5, 4, 3, 2, 6, 7,
                // Src: (fadd:{ *:[f32] } (VOP3Mods:{ *:[f32] } f32:{ *:[f32] }:$src1, i32:{ *:[i32] }:$src1_modifiers), (VOP3Mods0:{ *:[f32] } f32:{ *:[f32] }:$src0, i32:{ *:[i32] }:$src0_modifiers, i1:{ *:[i1] }:$clamp, i32:{ *:[i32] }:$omod)) - Complexity = -973
                // Dst: (V_ADD_F32_e64:{ *:[f32] } i32:{ *:[i32] }:$src0_modifiers, f32:{ *:[f32] }:$src0, i32:{ *:[i32] }:$src1_modifiers, f32:{ *:[f32] }:$src1, i1:{ *:[i1] }:$clamp, i32:{ *:[i32] }:$omod)
/*529209*/     0, /*End of Scope*/
/*529210*/    /*Scope*/ 102, /*->529313*/
/*529211*/     OPC_CheckType, /*MVT::f16*/11,
/*529213*/     OPC_Scope, 17, /*->529232*/ // 5 children in Scope
/*529215*/      OPC_CheckPatternPredicate, 15, // (Subtarget->has16BitInsts()) && (!Subtarget->hasTrue16BitInsts())
/*529217*/      OPC_CheckComplexPat2, /*#*/0, // SelectVOP3Mods0:$ #2 #3 #4 #5
/*529219*/      OPC_CheckComplexPat0, /*#*/1, // SelectVOP3Mods:$ #6 #7
/*529221*/      OPC_MorphNodeTo1None, TARGET_VAL(AMDGPU::V_ADD_F16_e64),
                    /*MVT::f16*/11, 6/*#Ops*/, 3, 2, 7, 6, 4, 5,
                // Src: (fadd:{ *:[f16] } (VOP3Mods0:{ *:[f16] } f16:{ *:[f16] }:$src0, i32:{ *:[i32] }:$src0_modifiers, i1:{ *:[i1] }:$clamp, i32:{ *:[i32] }:$omod), (VOP3Mods:{ *:[f16] } f16:{ *:[f16] }:$src1, i32:{ *:[i32] }:$src1_modifiers)) - Complexity = -973
                // Dst: (V_ADD_F16_e64:{ *:[f16] } i32:{ *:[i32] }:$src0_modifiers, f16:{ *:[f16] }:$src0, i32:{ *:[i32] }:$src1_modifiers, f16:{ *:[f16] }:$src1, i1:{ *:[i1] }:$clamp, i32:{ *:[i32] }:$omod)
/*529232*/     /*Scope*/ 17, /*->529250*/
/*529233*/      OPC_CheckPatternPredicate, 8, // (Subtarget->hasTrue16BitInsts() && !Subtarget->useRealTrue16Insts())
/*529235*/      OPC_CheckComplexPat2, /*#*/0, // SelectVOP3Mods0:$ #2 #3 #4 #5
/*529237*/      OPC_CheckComplexPat0, /*#*/1, // SelectVOP3Mods:$ #6 #7
/*529239*/      OPC_MorphNodeTo1None, TARGET_VAL(AMDGPU::V_ADD_F16_fake16_e64),
                    /*MVT::f16*/11, 6/*#Ops*/, 3, 2, 7, 6, 4, 5,
                // Src: (fadd:{ *:[f16] } (VOP3Mods0:{ *:[f16] } f16:{ *:[f16] }:$src0, i32:{ *:[i32] }:$src0_modifiers, i1:{ *:[i1] }:$clamp, i32:{ *:[i32] }:$omod), (VOP3Mods:{ *:[f16] } f16:{ *:[f16] }:$src1, i32:{ *:[i32] }:$src1_modifiers)) - Complexity = -973
                // Dst: (V_ADD_F16_fake16_e64:{ *:[f16] } i32:{ *:[i32] }:$src0_modifiers, f16:{ *:[f16] }:$src0, i32:{ *:[i32] }:$src1_modifiers, f16:{ *:[f16] }:$src1, i1:{ *:[i1] }:$clamp, i32:{ *:[i32] }:$omod)
/*529250*/     /*Scope*/ 17, /*->529268*/
/*529251*/      OPC_CheckPatternPredicate, 15, // (Subtarget->has16BitInsts()) && (!Subtarget->hasTrue16BitInsts())
/*529253*/      OPC_CheckComplexPat0, /*#*/0, // SelectVOP3Mods:$ #2 #3
/*529255*/      OPC_CheckComplexPat2, /*#*/1, // SelectVOP3Mods0:$ #4 #5 #6 #7
/*529257*/      OPC_MorphNodeTo1None, TARGET_VAL(AMDGPU::V_ADD_F16_e64),
                    /*MVT::f16*/11, 6/*#Ops*/, 5, 4, 3, 2, 6, 7,
                // Src: (fadd:{ *:[f16] } (VOP3Mods:{ *:[f16] } f16:{ *:[f16] }:$src1, i32:{ *:[i32] }:$src1_modifiers), (VOP3Mods0:{ *:[f16] } f16:{ *:[f16] }:$src0, i32:{ *:[i32] }:$src0_modifiers, i1:{ *:[i1] }:$clamp, i32:{ *:[i32] }:$omod)) - Complexity = -973
                // Dst: (V_ADD_F16_e64:{ *:[f16] } i32:{ *:[i32] }:$src0_modifiers, f16:{ *:[f16] }:$src0, i32:{ *:[i32] }:$src1_modifiers, f16:{ *:[f16] }:$src1, i1:{ *:[i1] }:$clamp, i32:{ *:[i32] }:$omod)
/*529268*/     /*Scope*/ 17, /*->529286*/
/*529269*/      OPC_CheckPatternPredicate, 8, // (Subtarget->hasTrue16BitInsts() && !Subtarget->useRealTrue16Insts())
/*529271*/      OPC_CheckComplexPat0, /*#*/0, // SelectVOP3Mods:$ #2 #3
/*529273*/      OPC_CheckComplexPat2, /*#*/1, // SelectVOP3Mods0:$ #4 #5 #6 #7
/*529275*/      OPC_MorphNodeTo1None, TARGET_VAL(AMDGPU::V_ADD_F16_fake16_e64),
                    /*MVT::f16*/11, 6/*#Ops*/, 5, 4, 3, 2, 6, 7,
                // Src: (fadd:{ *:[f16] } (VOP3Mods:{ *:[f16] } f16:{ *:[f16] }:$src1, i32:{ *:[i32] }:$src1_modifiers), (VOP3Mods0:{ *:[f16] } f16:{ *:[f16] }:$src0, i32:{ *:[i32] }:$src0_modifiers, i1:{ *:[i1] }:$clamp, i32:{ *:[i32] }:$omod)) - Complexity = -973
                // Dst: (V_ADD_F16_fake16_e64:{ *:[f16] } i32:{ *:[i32] }:$src0_modifiers, f16:{ *:[f16] }:$src0, i32:{ *:[i32] }:$src1_modifiers, f16:{ *:[f16] }:$src1, i1:{ *:[i1] }:$clamp, i32:{ *:[i32] }:$omod)
/*529286*/     /*Scope*/ 25, /*->529312*/
/*529287*/      OPC_CheckPatternPredicate, 11, // (Subtarget->useRealTrue16Insts())
/*529289*/      OPC_CheckComplexPat4, /*#*/0, // SelectVOP3OpSelMods:$ #2 #3
/*529291*/      OPC_CheckComplexPat4, /*#*/1, // SelectVOP3OpSelMods:$ #4 #5
/*529293*/      OPC_EmitInteger, /*MVT::i1*/2, 0,  // 0 #6
/*529296*/      OPC_EmitInteger32, 0,  // 0 #7
/*529298*/      OPC_EmitInteger32, 0,  // 0 #8
/*529300*/      OPC_MorphNodeTo1None, TARGET_VAL(AMDGPU::V_ADD_F16_t16_e64),
                    /*MVT::f16*/11, 7/*#Ops*/, 3, 2, 5, 4, 6, 7, 8,
                // Src: (fadd:{ *:[f16] } (VOP3OpSelMods:{ *:[f16] } f16:{ *:[f16] }:$src0, i32:{ *:[i32] }:$src0_modifiers), (VOP3OpSelMods:{ *:[f16] } f16:{ *:[f16] }:$src1, i32:{ *:[i32] }:$src1_modifiers)) - Complexity = -979
                // Dst: (V_ADD_F16_t16_e64:{ *:[f16] } i32:{ *:[i32] }:$src0_modifiers, f16:{ *:[f16] }:$src0, i32:{ *:[i32] }:$src1_modifiers, f16:{ *:[f16] }:$src1)
/*529312*/     0, /*End of Scope*/
/*529313*/    /*Scope*/ 76, /*->529390*/
/*529314*/     OPC_CheckType, /*MVT::f64*/13,
/*529316*/     OPC_Scope, 17, /*->529335*/ // 4 children in Scope
/*529318*/      OPC_CheckPatternPredicate, 21, // (Subtarget->getGeneration() >= AMDGPUSubtarget::GFX12)
/*529320*/      OPC_CheckComplexPat2, /*#*/0, // SelectVOP3Mods0:$ #2 #3 #4 #5
/*529322*/      OPC_CheckComplexPat0, /*#*/1, // SelectVOP3Mods:$ #6 #7
/*529324*/      OPC_MorphNodeTo1None, TARGET_VAL(AMDGPU::V_ADD_F64_pseudo_e64),
                    /*MVT::f64*/13, 6/*#Ops*/, 3, 2, 7, 6, 4, 5,
                // Src: (fadd:{ *:[f64] } (VOP3Mods0:{ *:[f64] } f64:{ *:[f64] }:$src0, i32:{ *:[i32] }:$src0_modifiers, i1:{ *:[i1] }:$clamp, i32:{ *:[i32] }:$omod), (VOP3Mods:{ *:[f64] } f64:{ *:[f64] }:$src1, i32:{ *:[i32] }:$src1_modifiers)) - Complexity = -973
                // Dst: (V_ADD_F64_pseudo_e64:{ *:[f64] } i32:{ *:[i32] }:$src0_modifiers, f64:{ *:[f64] }:$src0, i32:{ *:[i32] }:$src1_modifiers, f64:{ *:[f64] }:$src1, i1:{ *:[i1] }:$clamp, i32:{ *:[i32] }:$omod)
/*529335*/     /*Scope*/ 17, /*->529353*/
/*529336*/      OPC_CheckPatternPredicate, 64, // (Subtarget->getGeneration() <= AMDGPUSubtarget::GFX11)
/*529338*/      OPC_CheckComplexPat2, /*#*/0, // SelectVOP3Mods0:$ #2 #3 #4 #5
/*529340*/      OPC_CheckComplexPat0, /*#*/1, // SelectVOP3Mods:$ #6 #7
/*529342*/      OPC_MorphNodeTo1None, TARGET_VAL(AMDGPU::V_ADD_F64_e64),
                    /*MVT::f64*/13, 6/*#Ops*/, 3, 2, 7, 6, 4, 5,
                // Src: (fadd:{ *:[f64] } (VOP3Mods0:{ *:[f64] } f64:{ *:[f64] }:$src0, i32:{ *:[i32] }:$src0_modifiers, i1:{ *:[i1] }:$clamp, i32:{ *:[i32] }:$omod), (VOP3Mods:{ *:[f64] } f64:{ *:[f64] }:$src1, i32:{ *:[i32] }:$src1_modifiers)) - Complexity = -973
                // Dst: (V_ADD_F64_e64:{ *:[f64] } i32:{ *:[i32] }:$src0_modifiers, f64:{ *:[f64] }:$src0, i32:{ *:[i32] }:$src1_modifiers, f64:{ *:[f64] }:$src1, i1:{ *:[i1] }:$clamp, i32:{ *:[i32] }:$omod)
/*529353*/     /*Scope*/ 17, /*->529371*/
/*529354*/      OPC_CheckPatternPredicate, 21, // (Subtarget->getGeneration() >= AMDGPUSubtarget::GFX12)
/*529356*/      OPC_CheckComplexPat0, /*#*/0, // SelectVOP3Mods:$ #2 #3
/*529358*/      OPC_CheckComplexPat2, /*#*/1, // SelectVOP3Mods0:$ #4 #5 #6 #7
/*529360*/      OPC_MorphNodeTo1None, TARGET_VAL(AMDGPU::V_ADD_F64_pseudo_e64),
                    /*MVT::f64*/13, 6/*#Ops*/, 5, 4, 3, 2, 6, 7,
                // Src: (fadd:{ *:[f64] } (VOP3Mods:{ *:[f64] } f64:{ *:[f64] }:$src1, i32:{ *:[i32] }:$src1_modifiers), (VOP3Mods0:{ *:[f64] } f64:{ *:[f64] }:$src0, i32:{ *:[i32] }:$src0_modifiers, i1:{ *:[i1] }:$clamp, i32:{ *:[i32] }:$omod)) - Complexity = -973
                // Dst: (V_ADD_F64_pseudo_e64:{ *:[f64] } i32:{ *:[i32] }:$src0_modifiers, f64:{ *:[f64] }:$src0, i32:{ *:[i32] }:$src1_modifiers, f64:{ *:[f64] }:$src1, i1:{ *:[i1] }:$clamp, i32:{ *:[i32] }:$omod)
/*529371*/     /*Scope*/ 17, /*->529389*/
/*529372*/      OPC_CheckPatternPredicate, 64, // (Subtarget->getGeneration() <= AMDGPUSubtarget::GFX11)
/*529374*/      OPC_CheckComplexPat0, /*#*/0, // SelectVOP3Mods:$ #2 #3
/*529376*/      OPC_CheckComplexPat2, /*#*/1, // SelectVOP3Mods0:$ #4 #5 #6 #7
/*529378*/      OPC_MorphNodeTo1None, TARGET_VAL(AMDGPU::V_ADD_F64_e64),
                    /*MVT::f64*/13, 6/*#Ops*/, 5, 4, 3, 2, 6, 7,
                // Src: (fadd:{ *:[f64] } (VOP3Mods:{ *:[f64] } f64:{ *:[f64] }:$src1, i32:{ *:[i32] }:$src1_modifiers), (VOP3Mods0:{ *:[f64] } f64:{ *:[f64] }:$src0, i32:{ *:[i32] }:$src0_modifiers, i1:{ *:[i1] }:$clamp, i32:{ *:[i32] }:$omod)) - Complexity = -973
                // Dst: (V_ADD_F64_e64:{ *:[f64] } i32:{ *:[i32] }:$src0_modifiers, f64:{ *:[f64] }:$src0, i32:{ *:[i32] }:$src1_modifiers, f64:{ *:[f64] }:$src1, i1:{ *:[i1] }:$clamp, i32:{ *:[i32] }:$omod)
/*529389*/     0, /*End of Scope*/
/*529390*/    /*Scope*/ 37, /*->529428*/
/*529391*/     OPC_CheckType, /*MVT::v2f32*/109,
/*529393*/     OPC_CheckPredicate, 108, // Predicate_my_any_fadd
/*529395*/     OPC_CheckPatternPredicate, 104, // (Subtarget->hasPackedFP32Ops())
/*529397*/     OPC_CheckComplexPat, /*CP*/13, /*#*/0, // SelectVOP3PMods:$ #2 #3
/*529400*/     OPC_CheckComplexPat, /*CP*/13, /*#*/1, // SelectVOP3PMods:$ #4 #5
/*529403*/     OPC_EmitInteger, /*MVT::i1*/2, 0,  // 0 #6
/*529406*/     OPC_EmitInteger32, 0,  // 0 #7
/*529408*/     OPC_EmitInteger32, 0,  // 0 #8
/*529410*/     OPC_EmitInteger32, 0,  // 0 #9
/*529412*/     OPC_EmitInteger32, 0,  // 0 #10
/*529414*/     OPC_MorphNodeTo1None, TARGET_VAL(AMDGPU::V_PK_ADD_F32),
                   /*MVT::v2f32*/109, 9/*#Ops*/, 3, 2, 5, 4, 6, 7, 8, 9, 10,
               // Src: (fadd:{ *:[v2f32] } (VOP3PMods:{ *:[v2f32] } v2f32:{ *:[v2f32] }:$src0, i32:{ *:[i32] }:$src0_modifiers), (VOP3PMods:{ *:[v2f32] } v2f32:{ *:[v2f32] }:$src1, i32:{ *:[i32] }:$src1_modifiers))<<P:Predicate_my_any_fadd>> - Complexity = -978
               // Dst: (V_PK_ADD_F32:{ *:[v2f32] } i32:{ *:[i32] }:$src0_modifiers, v2f32:{ *:[v2f32] }:$src0, i32:{ *:[i32] }:$src1_modifiers, v2f32:{ *:[v2f32] }:$src1)
/*529428*/    /*Scope*/ 33, /*->529462*/
/*529429*/     OPC_CheckType, /*MVT::v2f16*/89,
/*529431*/     OPC_CheckComplexPat, /*CP*/13, /*#*/0, // SelectVOP3PMods:$ #2 #3
/*529434*/     OPC_CheckComplexPat, /*CP*/13, /*#*/1, // SelectVOP3PMods:$ #4 #5
/*529437*/     OPC_EmitInteger, /*MVT::i1*/2, 0,  // 0 #6
/*529440*/     OPC_EmitInteger32, 0,  // 0 #7
/*529442*/     OPC_EmitInteger32, 0,  // 0 #8
/*529444*/     OPC_EmitInteger32, 0,  // 0 #9
/*529446*/     OPC_EmitInteger32, 0,  // 0 #10
/*529448*/     OPC_MorphNodeTo1None, TARGET_VAL(AMDGPU::V_PK_ADD_F16),
                   /*MVT::v2f16*/89, 9/*#Ops*/, 3, 2, 5, 4, 6, 7, 8, 9, 10,
               // Src: (fadd:{ *:[v2f16] } (VOP3PMods:{ *:[v2f16] } v2f16:{ *:[v2f16] }:$src0, i32:{ *:[i32] }:$src0_modifiers), (VOP3PMods:{ *:[v2f16] } v2f16:{ *:[v2f16] }:$src1, i32:{ *:[i32] }:$src1_modifiers)) - Complexity = -979
               // Dst: (V_PK_ADD_F16:{ *:[v2f16] } i32:{ *:[i32] }:$src0_modifiers, v2f16:{ *:[v2f16] }:$src0, i32:{ *:[i32] }:$src1_modifiers, v2f16:{ *:[v2f16] }:$src1)
/*529462*/    0, /*End of Scope*/
/*529463*/   0, /*End of Scope*/
/*529464*/  0, /*End of Scope*/
  }; // Total Array size is 563301 bytes

  SelectCodeCommon(N, MatcherTable, sizeof(MatcherTable));
}
};

class AMDGPUISelDAGToDAGPass : public SelectionDAGISelPass {
public:
  AMDGPUISelDAGToDAGPass(TargetMachine &TM);

  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

class AMDGPUDAGToDAGISelLegacy : public SelectionDAGISelLegacy {
public:
  static char ID;

  AMDGPUDAGToDAGISelLegacy(TargetMachine &TM, CodeGenOptLevel OptLevel);

  bool runOnMachineFunction(MachineFunction &MF) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;
  StringRef getPassName() const override;
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUISELDAGTODAG_H
