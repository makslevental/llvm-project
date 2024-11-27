#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ..ir import IntegerAttr, IntegerType, register_attribute_builder
from ._vector_ops_gen import *
from ._vector_enum_gen import *


@register_attribute_builder("builtin.CombiningKind")
def _combiningkind(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.PrintPunctuation")
def _printpunctuation(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.Vector_IteratorType")
def _vector_iteratortype(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))
