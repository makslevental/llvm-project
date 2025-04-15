from enum import Flag
from pprint import pprint


class SchedGroupMask(Flag):
    NONE = 0
    ALU = 1 << 0
    VALU = 1 << 1
    SALU = 1 << 2
    MFMA = 1 << 3
    VMEM = 1 << 4
    VMEM_READ = 1 << 5
    VMEM_WRITE = 1 << 6
    DS = 1 << 7
    DS_READ = 1 << 8
    DS_WRITE = 1 << 9
    TRANS = 1 << 10


# SCHED_BARRIER's mask describes which instruction types should be allowed to be scheduled across it.
all = {SchedGroupMask._value2member_map_[1 << i] for i in range(0, 10 + 1)}.union(
    {SchedGroupMask.NONE}
)
ALL = (
    SchedGroupMask.ALU.value
    | SchedGroupMask.VALU.value
    | SchedGroupMask.SALU.value
    | SchedGroupMask.MFMA.value
    | SchedGroupMask.VMEM.value
    | SchedGroupMask.VMEM_READ.value
    | SchedGroupMask.VMEM_WRITE.value
    | SchedGroupMask.DS.value
    | SchedGroupMask.DS_READ.value
    | SchedGroupMask.DS_WRITE.value
    | SchedGroupMask.TRANS.value
)
print(ALL)
pprint(all)

ls = {SchedGroupMask._value2member_map_[i] for i in [1, 8, 16, 32, 64, 128, 256, 512]}
pprint(all - ls)
