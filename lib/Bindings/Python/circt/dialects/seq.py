#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Generated tablegen dialects end up in the mlir.dialects package for now.
from mlir.dialects._seq_ops_gen import *


# Create a computational register whose input is the given value, and is clocked
# by the given clock. If a reset is provided, the register will be reset by that
# signal. If a reset value is provided, the register will reset to that,
# otherwise it will reset to zero. If name is provided, the register will be
# named.
def reg(value, clock, reset=None, reset_value=None, name=None):
  import circt.dialects.hw as hw
  from mlir.ir import IntegerAttr
  value_type = value.type
  if reset:
    if not reset_value:
      zero = IntegerAttr.get(value_type, 0)
      reset_value = hw.ConstantOp(value_type, zero).result
    return CompRegOp(value_type,
                     value,
                     clock,
                     reset=reset,
                     reset_value=reset_value,
                     name=name).result
  else:
    return CompRegOp(value_type, value, clock, name=name).result
