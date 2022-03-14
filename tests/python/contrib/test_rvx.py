import tvm
import tvm.relay.testing
from tvm.relay.testing import run_infer_type
from tvm import relay
from tvm.relay.op.contrib import rvx
import numpy as np


def get_qnn_func(
    data_shape,
    data_dtype,
    kernel_shape,
    kernel_dtype,
    input_zero_point,
    kernel_zero_point,
    input_scale,
    kernel_scale,
    kernel_size,
    padding,
    strides,
    dilation,
    data_layout,
    kernel_layout,
    out_dtype,
    channels=None,
    groups=1,
):
    if isinstance(input_zero_point, (int, float)):
        input_zero_point = relay.const(input_zero_point, "int32")
    if isinstance(kernel_zero_point, (int, float)):
        kernel_zero_point = relay.const(kernel_zero_point, "int32")

    data = relay.var("data", shape=data_shape, dtype=data_dtype)
    kernel = relay.var("kernel", shape=kernel_shape, dtype=kernel_dtype)

    func = relay.qnn.op.conv2d(
        data,
        kernel,
        input_zero_point=input_zero_point,
        kernel_zero_point=kernel_zero_point,
        input_scale=relay.const(input_scale, "float32"),
        kernel_scale=relay.const(kernel_scale, "float32"),
        kernel_size=kernel_size,
        strides=strides,
        dilation=dilation,
        padding=padding,
        out_dtype=out_dtype,
        groups=groups,
        channels=channels,
        data_layout=data_layout,
        kernel_layout=kernel_layout,
    )

    mod = relay.Function(relay.analysis.free_vars(func), func)
    mod = tvm.IRModule.from_expr(mod)
    return mod


def check_partition(mod, params=None):
    def check_rvx_used(mod):
        num_rvx_subgraphs = sum(
        [1 if "rvx" in gv.name_hint else 0 for gv in mod.get_global_vars()]
        )
        assert num_rvx_subgraphs >= 1
    mod = rvx.partition_for_rvx(mod, params)
    print(mod.astext(show_meta_data=False))
    check_rvx_used(mod)
    
    
def check_conv2d():
    data_shape = (2, 1, 2, 4)
    data_dtype = "int8"
    kernel_shape = (3, 1, 2, 2)
    kernel_dtype = "int8"
    func = get_qnn_func(
        data_shape=data_shape,
        data_dtype=data_dtype,
        kernel_shape=kernel_shape,
        kernel_dtype=kernel_dtype,
        input_zero_point=0,
        kernel_zero_point=0,
        input_scale=1.0,
        kernel_scale=1.0,
        kernel_size=(2, 2),
        padding=(0, 0),
        strides=(1, 1),
        dilation=(1, 1),
        data_layout="NCHW",
        kernel_layout="OIHW",
        out_dtype="int32",
    )
    print(func.astext(show_meta_data=False))
    check_partition(func)
    
if __name__ == "__main__":
    check_conv2d()