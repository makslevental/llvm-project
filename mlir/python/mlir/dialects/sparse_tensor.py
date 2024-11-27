#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ..ir import IntegerAttr, IntegerType, register_attribute_builder
from ._sparse_tensor_ops_gen import *
from ._sparse_tensor_enum_gen import *
from .._mlir_libs._mlirDialectsSparseTensor import *
from .._mlir_libs import _mlirSparseTensorPasses as _cextSparseTensorPasses


@register_attribute_builder("builtin.SparseTensorSortKindEnum")
def _sparsetensorsortkindenum(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.SparseTensorStorageSpecifierKindEnum")
def _sparsetensorstoragespecifierkindenum(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))
