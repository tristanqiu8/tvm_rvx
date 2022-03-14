# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, unused-argument
"""RVX Hardware supported operators and patterns"""
import logging

import tvm.ir
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name

from ...dataflow_pattern import wildcard, is_op
from .register import register_pattern_table

logger = logging.getLogger("RVX")

def make_conv_pattern(with_bias, with_relu, with_pool, relu_first=True):
    data = wildcard()
    weight = wildcard()
    bias = wildcard()
    in_zp = wildcard()
    in_scale = wildcard()
    kernel_zp = wildcard()
    kernel_scale = wildcard()
    reqt_in_scale = wildcard()
    reqt_in_zp = wildcard()
    reqt_out_scale = wildcard()
    reqt_out_zp = wildcard()
    conv = is_op("qnn.conv2d")(
        data,
        weight,
        in_zp,
        kernel_zp,
        in_scale,
        kernel_scale
    )
    # add possible nn.bias_add
    if with_bias:
        conv_out = is_op("nn.bias_add")(
            conv,
            bias
        )
    else:
        conv_out = conv
    # add qnn.requantize
    conv_req = is_op("qnn.requantize")(
        conv_out,
        reqt_in_scale,
        reqt_in_zp,
        reqt_out_scale,
        reqt_out_zp
    )
    if relu_first:
        if with_relu:
            relu_out = is_op("clip")(conv_req)
        else:
            relu_out = conv_out

        if with_pool:
            pool_out = is_op("nn.max_pool2d")(relu_out)
        else:
            pool_out = relu_out
        return pool_out
    else:
        if with_pool:
            pool_out = is_op("nn.max_pool2d")(conv_req)
        else:
            pool_out = conv_out

        if with_relu:
            relu_out = is_op("clip")(pool_out)
        else:
            relu_out = pool_out
        return relu_out

# make_conv_pattern(with_bias, with_relu, with_pool, relu_first=True)
conv_wob_woa_wop_pat = ("rvx.conv_wob_woa_wop", make_conv_pattern(False, False, False))
conv_wb_woa_wop_pat = ("rvx.conv_wb_woa_wop", make_conv_pattern(True, False, False))
conv_wob_wa_wop_pat = ("rvx.conv_wob_wa_wop", make_conv_pattern(False, True, False))
conv_wb_wa_wop_pat = ("rvx.conv_wob_wa_wop", make_conv_pattern(True, True, False))
conv_wob_woa_wp_pat = ("rvx.conv_wob_woa_wp", make_conv_pattern(False, False, True))
conv_wb_woa_wp_pat = ("rvx.conv_wb_woa_wp", make_conv_pattern(True, False, True))
conv_wob_wa_wp_pat = ("rvx.conv_wob_wa_wp", make_conv_pattern(False, True, True))
conv_wb_wa_wp_pat = ("rvx.conv_wb_wa_wp", make_conv_pattern(True, True, True))
conv_wob_wp_wa_pat = ("rvx.conv_wob_wp_wa", make_conv_pattern(False, True, True, False))
conv_wb_wp_wa_pat = ("rvx.conv_wb_wp_wa", make_conv_pattern(True, True, True, False))

@register_pattern_table("rvx")
def pattern_table():
    """Create rvx patterns.

    Returns
    -------
    rvx_patterns : List[rvx_pattern]
        Created patterns.
    """
    patterns = [conv_wb_wa_wp_pat, conv_wb_wp_wa_pat,
                conv_wb_wa_wop_pat, conv_wb_woa_wp_pat, conv_wob_wa_wp_pat, conv_wob_wp_wa_pat,
                conv_wb_woa_wop_pat, conv_wob_wa_wop_pat, conv_wob_woa_wp_pat,
                conv_wob_woa_wop_pat]
    return patterns

def partition_for_rvx(mod, params=None):
    """Partition the graph greedily offloading supported operators to RVX.

    Parameters
    ----------
    mod : Module
        The module to run passes on.
    params : Optional[Dict[str, NDArray]]
        Constant input parameters.
    Returns
    -------
    mod : Module
        Annotated and partitioned module.
    """
    if params:
        mod["main"] = bind_params_by_name(mod["main"], params)
    seq = tvm.transform.Sequential(
        [
            transform.InferType(),
            transform.MergeComposite(pattern_table()),
            transform.AnnotateTarget("rvx"),
            transform.MergeCompilerRegions(),
            transform.PartitionGraph(),
        ]
    )

    return seq(mod)