#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ..ir import IntegerAttr, IntegerType, register_attribute_builder
from ._amdgpu_ops_gen import *
from ._amdgpu_enum_gen import *


@register_attribute_builder("builtin.AMDGPU_DPPPerm")
def _amdgpu_dppperm(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.AMDGPU_MFMAPermB")
def _amdgpu_mfmapermb(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.AMDGPU_SchedBarrierOpOpt")
def _amdgpu_schedbarrieropopt(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))
