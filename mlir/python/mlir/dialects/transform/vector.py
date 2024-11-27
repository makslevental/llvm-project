#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ...ir import IntegerAttr, IntegerType, register_attribute_builder
from .._vector_transform_enum_gen import *
from .._vector_transform_ops_gen import *


@register_attribute_builder("builtin.VectorContractLoweringAttr")
def _vectorcontractloweringattr(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.VectorMultiReductionLoweringAttr")
def _vectormultireductionloweringattr(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.VectorTransferSplitAttr")
def _vectortransfersplitattr(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.VectorTransposeLoweringAttr")
def _vectortransposeloweringattr(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))
