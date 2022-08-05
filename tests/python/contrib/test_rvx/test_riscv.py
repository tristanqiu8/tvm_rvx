import os
import tvm
import tvm.relay as relay
from tvm.contrib import utils
from tvm.contrib.download import download_testdata

data_dtype = "uint8"

x = relay.var("x", shape=(1, 4), dtype=data_dtype)
y = relay.var("y", shape=(1, 4), dtype=data_dtype)
z = relay.qnn.op.add(
        lhs=x,
        rhs=y,
        lhs_scale=relay.const(0.25, "float32"),
        lhs_zero_point=relay.const(127, "int32"),
        rhs_scale=relay.const(0.25, "float32"),
        rhs_zero_point=relay.const(127, "int32"),
        output_scale=relay.const(0.25, "float32"),
        output_zero_point=relay.const(127, "int32"),
    )

func = relay.Function([x, y], z)
mod = tvm.IRModule.from_expr(func)
mod = relay.transform.InferType()(mod)

target = tvm.target.riscv_cpu()
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target, params=None)

tmp = utils.tempdir()
lib_fname = tmp.relpath("net.tar")
lib.export_library(lib_fname)