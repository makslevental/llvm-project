//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is part of the WebAssembly Disassembler Emitter.
// It contains the implementation of the disassembler tables.
// Documentation for the disassembler emitter in general can be found in
// WebAssemblyDisassemblerEmitter.h.
//
//===----------------------------------------------------------------------===//

#include "WebAssemblyDisassemblerEmitter.h"
#include "Common/CodeGenInstruction.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/iterator.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Record.h"

constexpr int WebAssemblyInstructionTableSize = 256;

void llvm::emitWebAssemblyDisassemblerTables(
    raw_ostream &OS,
    ArrayRef<const CodeGenInstruction *> NumberedInstructions) {
  // First lets organize all opcodes by (prefix) byte. Prefix 0 is the
  // starting table.
  std::map<unsigned,
           std::map<unsigned, std::pair<unsigned, const CodeGenInstruction *>>>
      OpcodeTable;
  for (const auto &[Idx, CGI] :
       enumerate(make_pointee_range(NumberedInstructions))) {
    const Record &Def = *CGI.TheDef;
    if (!Def.getValue("Inst"))
      continue;
    const BitsInit &Inst = *Def.getValueAsBitsInit("Inst");
    unsigned Opc = static_cast<unsigned>(*Inst.convertInitializerToInt());
    if (Opc == 0xFFFFFFFF)
      continue; // No opcode defined.
    assert(Opc <= 0xFFFFFF);
    unsigned Prefix;
    if (Opc <= 0xFFFF) {
      Prefix = Opc >> 8;
      Opc = Opc & 0xFF;
    } else {
      Prefix = Opc >> 16;
      Opc = Opc & 0xFFFF;
    }
    auto &CGIP = OpcodeTable[Prefix][Opc];
    // All wasm instructions have a StackBased field of type string, we only
    // want the instructions for which this is "true".
    bool IsStackBased = Def.getValueAsBit("StackBased");
    if (!IsStackBased)
      continue;
    if (CGIP.second) {
      // We already have an instruction for this slot, so decide which one
      // should be the canonical one. This determines which variant gets
      // printed in a disassembly. We want e.g. "call" not "i32.call", and
      // "end" when we don't know if its "end_loop" or "end_block" etc.
      bool IsCanonicalExisting =
          CGIP.second->TheDef->getValueAsBit("IsCanonical");
      // We already have one marked explicitly as canonical, so keep it.
      if (IsCanonicalExisting)
        continue;
      bool IsCanonicalNew = Def.getValueAsBit("IsCanonical");
      // If the new one is explicitly marked as canonical, take it.
      if (!IsCanonicalNew) {
        // Neither the existing or new instruction is canonical.
        // Pick the one with the shortest name as heuristic.
        // Though ideally IsCanonical is always defined for at least one
        // variant so this never has to apply.
        if (CGIP.second->AsmString.size() <= CGI.AsmString.size())
          continue;
      }
    }
    // Set this instruction as the one to use.
    CGIP = {Idx, &CGI};
  }

  OS << R"(
#include "MCTargetDesc/WebAssemblyMCTargetDesc.h"

namespace {
enum EntryType : uint8_t { ET_Unused, ET_Prefix, ET_Instruction };

struct WebAssemblyInstruction {
  uint16_t Opcode;
  EntryType ET;
  uint8_t NumOperands;
  uint16_t OperandStart;
};
} // end anonymous namespace

)";

  std::vector<std::string> OperandTable, CurOperandList;
  // Output one table per prefix.
  for (const auto &[Prefix, Table] : OpcodeTable) {
    if (Table.empty())
      continue;
    OS << "constexpr WebAssemblyInstruction InstructionTable" << Prefix;
    OS << "[] = {\n";
    for (unsigned I = 0; I < WebAssemblyInstructionTableSize; I++) {
      auto InstIt = Table.find(I);
      if (InstIt != Table.end()) {
        // Regular instruction.
        assert(InstIt->second.second);
        auto &CGI = *InstIt->second.second;
        OS << "  // 0x";
        OS.write_hex(static_cast<unsigned long long>(I));
        OS << ": " << CGI.AsmString << "\n";
        OS << "  { " << InstIt->second.first << ", ET_Instruction, ";
        OS << CGI.Operands.OperandList.size() << ", ";
        // Collect operand types for storage in a shared list.
        CurOperandList.clear();
        for (auto &Op : CGI.Operands.OperandList) {
          assert(Op.OperandType != "MCOI::OPERAND_UNKNOWN");
          CurOperandList.push_back(Op.OperandType);
        }
        // See if we already have stored this sequence before. This is not
        // strictly necessary but makes the table really small.
        auto SearchI =
            std::search(OperandTable.begin(), OperandTable.end(),
                        CurOperandList.begin(), CurOperandList.end());
        OS << std::distance(OperandTable.begin(), SearchI);
        // Store operands if no prior occurrence.
        if (SearchI == OperandTable.end())
          llvm::append_range(OperandTable, CurOperandList);
      } else {
        auto PrefixIt = OpcodeTable.find(I);
        // If we have a non-empty table for it that's not 0, this is a prefix.
        if (PrefixIt != OpcodeTable.end() && I && !Prefix)
          OS << "  { 0, ET_Prefix, 0, 0";
        else
          OS << "  { 0, ET_Unused, 0, 0";
      }
      OS << "  },\n";
    }
    OS << "};\n\n";
  }
  // Create a table of all operands:
  OS << "constexpr uint8_t OperandTable[] = {\n";
  for (const auto &Op : OperandTable)
    OS << "  " << Op << ",\n";
  OS << "};\n\n";

  // Create a table of all extension tables:
  OS << R"(
constexpr struct {
  uint8_t Prefix;
  const WebAssemblyInstruction *Table;
} PrefixTable[] = {
)";

  for (const auto &[Prefix, Table] : OpcodeTable) {
    if (Table.empty() || !Prefix)
      continue;
    OS << "  { " << Prefix << ", InstructionTable" << Prefix << " },\n";
  }
  OS << "};\n";
}
