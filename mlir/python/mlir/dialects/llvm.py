#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ._llvm_ops_gen import *
from ._llvm_enum_gen import *
from .._mlir_libs._mlirDialectsLLVM import *
from ..ir import Value, IntegerAttr, IntegerType, register_attribute_builder
from ._ods_common import get_op_result_or_op_results as _get_op_result_or_op_results


def mlir_constant(value, *, loc=None, ip=None) -> Value:
    return _get_op_result_or_op_results(
        ConstantOp(res=value.type, value=value, loc=loc, ip=ip)
    )


@register_attribute_builder("builtin.AsmATTOrIntel")
def _asmattorintel(x, context):
    return IntegerAttr.get(IntegerType.get_signless(64, context=context), int(x))


@register_attribute_builder("builtin.AtomicBinOp")
def _atomicbinop(x, context):
    return IntegerAttr.get(IntegerType.get_signless(64, context=context), int(x))


@register_attribute_builder("builtin.AtomicOrdering")
def _atomicordering(x, context):
    return IntegerAttr.get(IntegerType.get_signless(64, context=context), int(x))


@register_attribute_builder("builtin.CConvEnum")
def _cconvenum(x, context):
    return IntegerAttr.get(IntegerType.get_signless(64, context=context), int(x))


@register_attribute_builder("builtin.Comdat")
def _comdat(x, context):
    return IntegerAttr.get(IntegerType.get_signless(64, context=context), int(x))


@register_attribute_builder("builtin.DIFlags")
def _diflags(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.DISubprogramFlags")
def _disubprogramflags(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.FCmpPredicate")
def _fcmppredicate(x, context):
    return IntegerAttr.get(IntegerType.get_signless(64, context=context), int(x))


@register_attribute_builder("builtin.FPExceptionBehaviorAttr")
def _fpexceptionbehaviorattr(x, context):
    return IntegerAttr.get(IntegerType.get_signless(64, context=context), int(x))


@register_attribute_builder("builtin.FastmathFlags")
def _fastmathflags(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.FramePointerKindEnum")
def _framepointerkindenum(x, context):
    return IntegerAttr.get(IntegerType.get_signless(64, context=context), int(x))


@register_attribute_builder("builtin.ICmpPredicate")
def _icmppredicate(x, context):
    return IntegerAttr.get(IntegerType.get_signless(64, context=context), int(x))


@register_attribute_builder("builtin.IntegerOverflowFlags")
def _integeroverflowflags(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.LLVM_DIEmissionKind")
def _llvm_diemissionkind(x, context):
    return IntegerAttr.get(IntegerType.get_signless(64, context=context), int(x))


@register_attribute_builder("builtin.LLVM_DINameTableKind")
def _llvm_dinametablekind(x, context):
    return IntegerAttr.get(IntegerType.get_signless(64, context=context), int(x))


@register_attribute_builder("builtin.LinkageEnum")
def _linkageenum(x, context):
    return IntegerAttr.get(IntegerType.get_signless(64, context=context), int(x))


@register_attribute_builder("builtin.ModRefInfoEnum")
def _modrefinfoenum(x, context):
    return IntegerAttr.get(IntegerType.get_signless(64, context=context), int(x))


@register_attribute_builder("builtin.RoundingModeAttr")
def _roundingmodeattr(x, context):
    return IntegerAttr.get(IntegerType.get_signless(64, context=context), int(x))


@register_attribute_builder("builtin.TailCallKindEnum")
def _tailcallkindenum(x, context):
    return IntegerAttr.get(IntegerType.get_signless(64, context=context), int(x))


@register_attribute_builder("builtin.UnnamedAddr")
def _unnamedaddr(x, context):
    return IntegerAttr.get(IntegerType.get_signless(64, context=context), int(x))


@register_attribute_builder("builtin.Visibility")
def _visibility(x, context):
    return IntegerAttr.get(IntegerType.get_signless(64, context=context), int(x))
