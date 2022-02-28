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
"""RVX library supported operators and patterns"""
import logging

import tvm.ir
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name

from ...dataflow_pattern import wildcard, is_op
from .register import register_pattern_table

logger = logging.getLogger("RVX")

def make_conv_pattern(with_relu, with_pool, relu_first):
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
    conv_out = is_op("qnn.conv2d")(
        data,
        weight,
        in_zp,
        kernel_zp,
        in_scale,
        kernel_scale
    )
    if relu_first:
        if with_relu:
            relu_out = is_op("clip")(conv_out)
        else:
            relu_out = conv_out

        if with_pool:
            pool_out = is_op("nn.max_pool2d")(relu_out)
        else:
            pool_out = relu_out
        return pool_out
    else:
        if with_pool:
            pool_out = is_op("nn.max_pool2d")(conv_out)
        else:
            pool_out = conv_out

        if with_relu:
            relu_out = is_op("clip")(pool_out)
        else:
            relu_out = pool_out
        return relu_out

@register_pattern_table("rvx")
def pattern_table():
    """Create rvx patterns.

    Returns
    -------
    rvx_patterns : List[rvx_pattern]
        Created patterns.
    """
    rvx_patterns = []
    return rvx_patterns