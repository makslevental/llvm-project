#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ...ir import IntegerAttr, IntegerType, register_attribute_builder
from .._gpu_ops_gen import *
from .._gpu_enum_gen import *
from ..._mlir_libs._mlirDialectsGPU import *


@register_attribute_builder("builtin.GPU_AddressSpaceEnum")
def _gpu_addressspaceenum(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.GPU_AllReduceOperation")
def _gpu_allreduceoperation(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.GPU_CompilationTargetEnum")
def _gpu_compilationtargetenum(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.GPU_Dimension")
def _gpu_dimension(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.GPU_Prune2To4SpMatFlag")
def _gpu_prune2to4spmatflag(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.GPU_ShuffleMode")
def _gpu_shufflemode(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.GPU_SpGEMMWorkEstimationOrComputeKind")
def _gpu_spgemmworkestimationorcomputekind(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.GPU_TransposeMode")
def _gpu_transposemode(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.MMAElementWise")
def _mmaelementwise(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.MappingIdEnum")
def _mappingidenum(x, context):
    return IntegerAttr.get(IntegerType.get_signless(64, context=context), int(x))


@register_attribute_builder("builtin.ProcessorEnum")
def _processorenum(x, context):
    return IntegerAttr.get(IntegerType.get_signless(64, context=context), int(x))
