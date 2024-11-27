#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ..ir import IntegerAttr, IntegerType, register_attribute_builder
from ._nvgpu_ops_gen import *
from ._nvgpu_enum_gen import *
from .._mlir_libs._mlirDialectsNVGPU import *


@register_attribute_builder("builtin.RcpRoundingMode")
def _rcproundingmode(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.TensorMapInterleaveKind")
def _tensormapinterleavekind(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.TensorMapL2PromoKind")
def _tensormapl2promokind(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.TensorMapOOBKind")
def _tensormapoobkind(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.TensorMapSwizzleKind")
def _tensormapswizzlekind(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))
