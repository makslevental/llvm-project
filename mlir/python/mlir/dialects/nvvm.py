#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ..ir import IntegerAttr, IntegerType, register_attribute_builder
from ._nvvm_ops_gen import *
from ._nvvm_enum_gen import *


@register_attribute_builder("builtin.LoadCacheModifierKind")
def _loadcachemodifierkind(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.MMAB1Op")
def _mmab1op(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.MMAFrag")
def _mmafrag(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.MMAIntOverflow")
def _mmaintoverflow(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.MMALayout")
def _mmalayout(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.MMATypes")
def _mmatypes(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.MemScopeKind")
def _memscopekind(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.ProxyKind")
def _proxykind(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.ReduxKind")
def _reduxkind(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.SetMaxRegisterAction")
def _setmaxregisteraction(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.SharedSpace")
def _sharedspace(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.ShflKind")
def _shflkind(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.WGMMAScaleIn")
def _wgmmascalein(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.WGMMAScaleOut")
def _wgmmascaleout(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.WGMMATypes")
def _wgmmatypes(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))
