#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ..ir import IntegerAttr, IntegerType, register_attribute_builder
from ._bufferization_ops_gen import *
from ._bufferization_enum_gen import *


@register_attribute_builder("builtin.LayoutMapOption")
def _layoutmapoption(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))
