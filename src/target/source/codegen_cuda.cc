/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file codegen_cuda.cc
 */

#include "codegen_cuda.h"

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/function.h>
#include <tvm/tir/index_map.h>
#include <tvm/tir/stmt_functor.h>

#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include "../../tir/transforms/ir_utils.h"
#include "literal/cuda_half_t.h"
#include "literal/cuda_int8_t.h"
#include "ptx.h"

namespace tvm {
namespace codegen {

std::string GetFP8Type(DataType type) {
  std::stringstream stream;
  int32_t lanes = type.lanes();
  std::string vec;
  if (type.is_scalar()) {
    vec = "";
  } else if (lanes == 2) {
    vec = "x2";
  } else if (lanes == 4) {
    vec = "x4";
  } else if (lanes == 8) {
    vec = "x8";
  } else if (lanes == 16) {
    vec = "x16";
  } else {
    LOG(FATAL) << "Only support scalar and vector types of width (2, 4, 8, 16) for FP8";
  }
  stream << "__nv_fp8";
  std::string suffix;
  if (type.code() == DataType::kFloat8_e4m3fn) {
    suffix = "_e4m3";
  } else if (type.code() == DataType::kFloat8_e5m2) {
    suffix = "_e5m2";
  } else if (type.code() == DataType::kFloat8_e8m0fnu) {
    suffix = "_e8m0";
  } else {
    LOG(FATAL) << "Unsupported FP8 type in CUDA codegen";
  }
  stream << vec << suffix;
  return stream.str();
}

std::string GetFP6Type(DataType type) {
  std::stringstream stream;
  int32_t lanes = type.lanes();
  std::string vec;
  if (type.is_scalar()) {
    vec = "";
  } else if (lanes == 2) {
    vec = "x2";
  } else if (lanes == 4) {
    vec = "x4";
  } else if (lanes == 8) {
    vec = "x8";
  } else if (lanes == 16) {
    vec = "x16";
  } else {
    LOG(FATAL) << "Only support scalar and vector types of width (2, 4) for FP6";
  }
  stream << "__nv_fp6";
  std::string suffix;
  if (type.code() == DataType::kFloat6_e2m3fn) {
    suffix = "_e2m3";
  } else if (type.code() == DataType::kFloat6_e3m2fn) {
    suffix = "_e3m2";
  } else {
    LOG(FATAL) << "Unsupported FP6 type in CUDA codegen";
  }
  stream << vec << suffix;
  return stream.str();
}

std::string GetFP4Type(DataType type) {
  std::stringstream stream;
  int32_t lanes = type.lanes();
  std::string vec;
  if (type.is_scalar()) {
    vec = "";
  } else if (lanes == 2) {
    vec = "x2";
  } else if (lanes == 4) {
    vec = "x4";
  } else if (lanes == 8) {
    vec = "x8";
  } else if (lanes == 16) {
    vec = "x16";
  } else {
    LOG(FATAL) << "Only support scalar and vector types of width (2, 4) for FP4";
  }
  stream << "__nv_fp4";
  std::string suffix;
  if (type.code() == DataType::kFloat4_e2m1fn) {
    suffix = "_e2m1";
  } else {
    LOG(FATAL) << "Unsupported FP4 type in CUDA codegen";
  }
  stream << vec << suffix;
  return stream.str();
}

CodeGenCUDA::CodeGenCUDA() { restrict_keyword_ = "__restrict__"; }

void CodeGenCUDA::Init(bool output_ssa) {
  CodeGenC::Init(output_ssa);
  vid_global_barrier_state_ = name_supply_->FreshName(runtime::symbol::tvm_global_barrier_state);
  vid_global_barrier_expect_ = name_supply_->FreshName("__barrier_expect");
  ICHECK_EQ(vid_global_barrier_state_, runtime::symbol::tvm_global_barrier_state);
}

void CodeGenCUDA::PrintFunctionSignature(const String& function_name, const PrimFunc& func,
                                         std::ostream& os) {
  auto calling_conv =
      func->GetAttr<Integer>(tvm::attr::kCallingConv, Integer(tvm::CallingConv::kDefault));
  if (calling_conv == CallingConv::kDeviceKernelLaunch) {
    os << "extern \"C\" __global__ ";
  } else if (calling_conv == CallingConv::kDefault) {
    os << "extern \"C\" __device__ ";
  } else {
    LOG(FATAL) << "Unsupported calling convention for cuda codegen: " << calling_conv;
  }
  CodeGenC::PrintFunctionSignature(function_name, func, os);
}

class ThreadIdxExtractor : public tir::StmtVisitor {
 private:
  void VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == tir::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      if (iv->var->name_hint == "threadIdx.x" || iv->thread_tag == "threadIdx.x") {
        threadIdx_x_ext = op->value;
      }
      if (iv->var->name_hint == "threadIdx.y" || iv->thread_tag == "threadIdx.y") {
        threadIdx_y_ext = op->value;
      }
      if (iv->var->name_hint == "threadIdx.z" || iv->thread_tag == "threadIdx.z") {
        threadIdx_z_ext = op->value;
      }
    }
    StmtVisitor::VisitStmt_(op);
  }

 public:
  PrimExpr threadIdx_x_ext = Integer(1);
  PrimExpr threadIdx_y_ext = Integer(1);
  PrimExpr threadIdx_z_ext = Integer(1);
};

void CodeGenCUDA::PrintExtraAttrs(const PrimFunc& f, std::ostream& os) {
  ThreadIdxExtractor extractor;
  extractor(f->body);
  arith::Analyzer analyzer;
  PrimExpr threadIdx_ext = analyzer.Simplify(extractor.threadIdx_x_ext * extractor.threadIdx_y_ext *
                                             extractor.threadIdx_z_ext);
  if (const IntImmNode* const threadIdx_ext_int = threadIdx_ext.as<IntImmNode>()) {
    if (threadIdx_ext_int->value == 1) {
      // unable to extract the number of threads per block, hence directly return
      return;
    }
    os << " __launch_bounds__(" << threadIdx_ext_int->value << ")";
  }
}

std::string CodeGenCUDA::Finish() {
  decl_stream << "#include <cuda.h>\n";

  if (enable_fp16_) {
    decl_stream << "#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)\n";
    decl_stream << "#include <cuda_fp16.h>\n";
    decl_stream << "__device__ half max"
                << "(half a, half b)\n"
                << "{\n  return __hgt(__half(a), __half(b)) ? a : b;\n}\n";
    decl_stream << "__device__ half min(half a, half b)\n"
                << "{\n  return __hlt(__half(a), __half(b)) ? a : b;\n}\n";
    decl_stream << "#else\n";
    decl_stream << _cuda_half_t_def;
    decl_stream << "#endif\n\n";

    decl_stream << _cuda_half_util;
  }

  if (enable_bf16_) {
    decl_stream << "#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)\n";
    decl_stream << "#include <cuda_bf16.h>\n";
    decl_stream << "__device__ nv_bfloat16 max"
                << "(nv_bfloat16 a, nv_bfloat16 b)\n"
                << "{\n  return __hgt(a, b) ? a : b;\n}\n";
    decl_stream << "__device__ nv_bfloat16 min(nv_bfloat16 a, nv_bfloat16 b)\n"
                << "{\n  return __hlt(a, b) ? a : b;\n}\n";
    decl_stream << "#endif\n\n";
    decl_stream << _cuda_bfloat16_util;
  }

  if (enable_fp8_) {
    decl_stream << "#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)\n";
    decl_stream << "#include <cuda_fp8.h>\n";
    decl_stream << "using fp8_e4_t = __nv_fp8_e4m3;\n";
    decl_stream << "using fp8_e4x2_t = __nv_fp8x2_e4m3;\n";
    decl_stream << "using fp8_e4x4_t = __nv_fp8x4_e4m3;\n";
    decl_stream << "struct fp8_e4x8_t {\n fp8_e4_t data[8]; \n};\n";
    decl_stream << "struct fp8_e4x16_t {\n fp8_e4_t data[16]; \n};\n";
    decl_stream << "using fp8_e5_t = __nv_fp8_e5m2;\n";
    decl_stream << "using fp8_e5x2_t = __nv_fp8x2_e5m2;\n";
    decl_stream << "using fp8_e5x4_t = __nv_fp8x4_e5m2;\n";
    decl_stream << "struct fp8_e5x8_t {\n fp8_e5_t data[8]; \n};\n";
    decl_stream << "struct fp8_e5x16_t {\n fp8_e5_t data[16]; \n};\n";
    decl_stream << "using fp8_e8_t = __nv_fp8_e8m0;\n";
    decl_stream << "using fp8_e8x2_t = __nv_fp8x2_e8m0;\n";
    decl_stream << "using fp8_e8x4_t = __nv_fp8x4_e8m0;\n";
    decl_stream << "struct fp8_e8x8_t {\n fp8_e8_t data[8]; \n};\n";
    decl_stream << "struct fp8_e8x16_t {\n fp8_e8_t data[16]; \n};\n";
    decl_stream << "#endif\n\n";
  }

  if (enable_fp6_) {
    decl_stream << "#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)\n";
    decl_stream << "#include <cuda_fp6.h>\n";
    decl_stream << "using fp6_e2_t = __nv_fp6_e2m3;\n";
    decl_stream << "using fp6_e2x2_t = __nv_fp6x2_e2m3;\n";
    decl_stream << "using fp6_e2x4_t = __nv_fp6x4_e2m3;\n";
    decl_stream << "struct fp6_e2x8_t {\n fp6_e2_t data[8]; \n};\n";
    decl_stream << "struct fp6_e2x16_t {\n fp6_e2_t data[16]; \n};\n";
    decl_stream << "using fp6_e3_t = __nv_fp6_e3m2;\n";
    decl_stream << "using fp6_e3x2_t = __nv_fp6x2_e3m2;\n";
    decl_stream << "using fp6_e3x4_t = __nv_fp6x4_e3m2;\n";
    decl_stream << "struct fp6_e3x8_t {\n fp6_e3_t data[8]; \n};\n";
    decl_stream << "struct fp6_e3x16_t {\n fp6_e3_t data[16]; \n};\n";
    decl_stream << "#endif\n\n";
  }

  if (enable_fp4_) {
    decl_stream << "#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)\n";
    decl_stream << "#include <cuda_fp4.h>\n";
    decl_stream << "using fp4_e2_t = __nv_fp4_e2m1;\n";
    decl_stream << "using fp4_e2x2_t = __nv_fp4x2_e2m1;\n";
    decl_stream << "using fp4_e2x4_t = __nv_fp4x4_e2m1;\n";
    decl_stream << "struct fp4_e2x8_t {\n fp4_e2_t data[8]; \n};\n";
    decl_stream << "struct fp4_e2x16_t {\n fp4_e2_t data[16]; \n};\n";
    decl_stream << "#endif\n\n";
  }
  declare_vector_type_extensions(decl_stream, enable_fp16_, enable_bf16_, enable_fp8_, enable_fp4_);

  if (enable_warp_shuffle_) {
    decl_stream << _cuda_warp_intrinsic_util;
  }

  if (enable_int8_) {
    decl_stream << "#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 610)\n";
    decl_stream << "#include <sm_61_intrinsics.h>\n";
    decl_stream << _cuda_int8_t_def;
    decl_stream << "#endif\n";
  }

  if (need_math_constants_h_) {
    decl_stream << "#include <math_constants.h>\n";
  }

  if (need_mma_h_) {
    decl_stream << "#include <mma.h>\n";
  }

  if (need_cast_smem_ptr_to_int_) {
    decl_stream << "__forceinline__ __device__ unsigned int\n";
    decl_stream << "cast_smem_ptr_to_int(const void* const smem_ptr)\n";
    decl_stream << "{\n";
    decl_stream << "  unsigned int smem_int;\n";
    decl_stream << "  asm volatile (\"{ .reg .u64 smem_int; cvta.to.shared.u64 smem_int, %1; "
                   "cvt.u32.u64 %0, smem_int; }\"\n";
    decl_stream << "    : \"=r\"(smem_int) : \"l\"(smem_ptr));\n";
    decl_stream << "  return smem_int;\n";
    decl_stream << "}\n";
  }

  decl_stream << "\n#if (((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || \\\n";
  decl_stream << "     (__CUDACC_VER_MAJOR__ > 11))\n";
  decl_stream << "#define TVM_ENABLE_L2_PREFETCH 1\n";
  decl_stream << "#else\n";
  decl_stream << "#define TVM_ENABLE_L2_PREFETCH 0\n";
  decl_stream << "#endif\n";

  decl_stream << "#include <cstdint>\n";
  decl_stream << "using uint = unsigned int;\n";
  decl_stream << "using uchar = unsigned char;\n";
  decl_stream << "using ushort = unsigned short;\n";

  return CodeGenC::Finish();
}

void CodeGenCUDA::VisitStmt_(const tir::ForNode* op) {
  ICHECK(is_const_int(op->min, 0));
  if (op->kind == tir::ForKind::kUnrolled) {
    PrintIndent();
    stream << "#pragma unroll\n";
  }
  CodeGenC::VisitStmt_(op);
}

void CodeGenCUDA::BindThreadIndex(const IterVar& iv) {
  ICHECK(!var_idmap_.count(iv->var.get()));
  var_idmap_[iv->var.get()] = CastFromTo(iv->thread_tag, DataType::UInt(32), iv->var.dtype());
}

void CodeGenCUDA::PrintType(DataType t, std::ostream& os) {  // NOLINT(*)
  int lanes = t.lanes();
  if (t.is_handle()) {
    ICHECK(t.is_scalar()) << "do not yet support vector types";
    os << "void*";
    return;
  }

  if (t.is_void()) {
    os << "void";
    return;
  }

  bool fail = false;
  if (t.is_float()) {
    switch (t.bits()) {
      case 16:
        enable_fp16_ = true;
        if (t.is_scalar()) {
          os << "half";
        } else if (lanes <= 8) {
          ICHECK_EQ(lanes % 2, 0) << "Only support an even number of lanes for half type";
          if (lanes <= 4) {
            os << "half" << lanes;
          } else {
            os << "uint" << lanes / 2;
          }
        } else {
          fail = true;
        }
        break;
      case 32:
        if (lanes <= 4) {
          os << "float";
        } else if (lanes <= 8) {
          // Emit CUDA code to access fp32 vector elements for 4 < lanes <= 8.
          //
          // float8 is stored as ulonglong4
          //
          // f8.v1 is emitted as *(float2*)(&(ul4.x)).x
          // f8.v2 is emitted as *(float2*)(&(ul4.x)).y
          //
          ICHECK_EQ(lanes % 2, 0) << "only support even lane for float type with lanes > 4";
          os << "ulonglong" << lanes / 2;
        } else {
          fail = true;
        }
        break;
      case 64:
        os << "double";
        break;
      default:
        fail = true;
        break;
    }
    if (!fail && (t.is_scalar() || t.bits() == 16)) return;
    if (!fail && (lanes > 4 && lanes <= 8 && t.bits() == 32)) return;
    if (!fail && (lanes >= 2 && lanes <= 4)) {
      os << lanes;
      return;
    }
  } else if (t.is_bfloat16()) {
    enable_bf16_ = true;
    if (t.is_scalar()) {
      os << "nv_bfloat16";
    } else if (lanes <= 8) {
      ICHECK_EQ(lanes % 2, 0) << "only support even lane for bfloat16 type";
      if (lanes <= 4) {
        os << "nv_bfloat16" << lanes;
      } else {
        os << "uint" << lanes / 2;
      }
    } else {
      fail = true;
    }
    if (!fail) return;
  } else if (t.is_float8()) {
    enable_fp8_ = true;
    if (t.lanes() <= 4) {
      os << GetFP8Type(t);
    } else {
      os << "uint" << t.lanes() / 4;
    }
    return;
  } else if (t.is_float6()) {
    enable_fp6_ = true;
    if (t.lanes() <= 4) {
      os << GetFP6Type(t);
    } else {
      fail = true;
    }
    return;
  } else if (t.is_float4()) {
    enable_fp4_ = true;
    if (t.lanes() <= 4) {
      os << GetFP4Type(t);
    } else {
      fail = true;
    }
    return;
  } else if (t == DataType::Bool()) {
    os << "bool";
    return;
  } else if (t.is_vector_bool()) {
    // CUDA does not support bool vectors.
    // Use ushort vectors to represent instead.
    int n = t.lanes();
    if (n <= 4) {
      os << "ushort" << n;
      return;
    }
  } else if (t.is_uint() || t.is_int()) {
    if (t.is_uint()) {
      os << "u";
    }
    switch (t.bits()) {
      case 1: {
        if (t.is_scalar()) {
          os << "int";
          return;
        } else if (t.lanes() == 8) {
          os << "int8_t";
          return;
        } else if (t.lanes() == 16) {
          os << "int16_t";
          return;
        } else if (t.lanes() == 32) {
          os << "int";
          return;
        } else {
          LOG(FATAL) << "Cannot convert type " << t << " to CUDA type!";
        }
      }
      case 4: {
        if (t.is_scalar()) {
          os << "int";
          return;
        } else if (t.lanes() == 4) {
          os << "int16_t";
          return;
        } else if (t.lanes() == 8) {
          // directly 8 4-bit int in integer.
          os << "int";
          return;
        } else if (t.lanes() == 16) {
          os << "int2";
          return;
        } else if (t.lanes() == 32) {
          os << "int4";
          return;
        } else if (t.lanes() == 64) {
          os << "int8";
          return;
        } else {
          LOG(FATAL) << "Cannot convert type " << t << " to CUDA type!";
        }
      }
      case 8: {
        if (t.lanes() == 4) {
          // directly 4 8 bit int in integer.
          enable_int8_ = true;

          // We use int for int8x4 instead of char4 because using char4 is
          // likely to produce extra instructions to pack four int8 elements
          // into 32-bit data.
          os << "int";
          return;
        } else if (t.lanes() == 8) {
          enable_int8_ = true;
          os << "int2";
          return;
        } else if (t.lanes() == 16) {
          enable_int8_ = true;
          os << "int4";
          return;
        } else if (!t.is_uint() && t.is_scalar()) {
          os << "signed char";
          break;
        } else {
          os << "char";
          break;
        }
      }
      case 16: {
        if (t.is_scalar()) {
          os << "short";
        } else if (t.lanes() <= 4) {
          os << "short" << lanes;
        } else if (t.lanes() <= 8) {
          // Emit CUDA code to access int16 vector elements.
          //
          // short4 is stored as int2
          //
          // s4.x is emitted as *(short2*)(&(i2.x)).x
          // s4.y is emitted as *(short2*)(&(i2.x)).y
          // s4.z is emitted as *(short2*)(&(i2.y)).x
          // s4.w is emitted as *(short2*)(&(i2.y)).y
          //
          ICHECK_EQ(t.lanes() % 2, 0) << "only support even lane for shorT type with lanes > 4";
          os << "int" << t.lanes() / 2;
        } else {
          fail = true;
        }
        if (!fail) {
          return;
        }
        break;
      }
      case 32: {
        if (t.is_scalar()) {
          os << "int";
        } else if (t.lanes() <= 4) {
          os << "int" << t.lanes();
        } else if (t.lanes() <= 8) {
          // Emit CUDA code to access int32 vector elements for 4 < lanes <= 8.
          //
          // int8 is stored as longlong4
          //
          // i8.v1 is emitted as *(int2*)(&(l4.x)).x
          // i8.v2 is emitted as *(int2*)(&(l4.x)).y
          //
          ICHECK_EQ(lanes % 2, 0) << "only support even lane for int32 type with lanes > 4";
          os << "longlong" << lanes / 2;
        } else {
          fail = true;
        }
        if (!fail) {
          return;
        }
        break;
      }
      case 64: {
        if (t.is_scalar()) {
          os << "int64_t";
        } else if (t.lanes() == 2) {
          os << "longlong2";
        } else if (t.lanes() == 3) {
          os << "longlong3";
        } else if (t.lanes() == 4) {
          os << "longlong4";
        }
        return;
      }
      default:
        fail = true;
        break;
    }
    if (!fail && lanes == 1) {
      return;
    }
    if (!fail && (lanes >= 2 && lanes <= 4)) {
      os << lanes;
      return;
    }
  }
  LOG(FATAL) << "Cannot convert type " << t << " to CUDA type";
}

void CodeGenCUDA::PrintVecConstructor(DataType t, std::ostream& os) {
  os << "make_";
  PrintType(t, os);
}

void CodeGenCUDA::PrintVecBinaryOp(const std::string& op, DataType t, PrimExpr lhs, PrimExpr rhs,
                                   std::ostream& os) {  // NOLINT(*)
  // Declare the result.
  std::string sret = name_supply_->FreshName("_");
  this->PrintIndent();
  this->PrintType(t, stream);
  stream << ' ' << sret << ";\n";
  int ssa_scope = BeginScope();
  {
    // Unpack into individual ops.
    std::string vlhs = SSAGetID(PrintExpr(lhs), lhs.dtype());
    std::string vrhs = SSAGetID(PrintExpr(rhs), rhs.dtype());

    for (int i = 0, lanes = t.lanes(); i < lanes; ++i) {
      std::ostringstream value_temp;
      if (isalpha(op[0])) {
        value_temp << op << "(";
        PrintVecElemLoad(vlhs, lhs.dtype(), i, value_temp);
        value_temp << ", ";
        PrintVecElemLoad(vrhs, rhs.dtype(), i, value_temp);
        value_temp << ")";
      } else {
        value_temp << "(";
        PrintVecElemLoad(vlhs, lhs.dtype(), i, value_temp);
        value_temp << op;
        PrintVecElemLoad(vrhs, rhs.dtype(), i, value_temp);
        value_temp << ")";
      }
      PrintVecElemStore(sret, t, i, value_temp.str());
    }
  }
  EndScope(ssa_scope);
  os << sret;
}

void CodeGenCUDA::PrintVecElemLoad(const std::string& vec, DataType t, int i,
                                   std::ostream& os) {  // NOLINT(*)
  if (t.is_scalar()) {
    os << vec;
    return;
  }

  static const char access[] = {'x', 'y', 'z', 'w'};
  ICHECK(i >= 0 && i < (t.bits() == 8 ? 16 : (t.bits() == 16 || t.bits() == 32) ? 8 : 4));
  if (t.bits() == 8 && (t.is_int() || t.is_uint())) {
    std::string type_name = t.is_int() ? "char" : "unsigned char";
    if (t.lanes() == 2 || t.lanes() == 3) {
      os << vec << "." << access[i % t.lanes()];
    } else {
      std::string ac = t.lanes() == 4 ? vec : (vec + "." + access[i / 4]);
      os << "((" << type_name << ")(" << ac << " >> " << i % 4 * 8 << "))";
    }
  } else if (t.is_float16()) {
    if (t.lanes() <= 4) {
      os << vec << "." << access[i];
    } else {
      os << "((half2*)(&(" << vec << "." << access[i / 2] << ")))->" << access[i % 2];
    }
  } else if (t.is_bfloat16()) {
    if (t.lanes() <= 4) {
      os << vec << "." << access[i];
    } else {
      os << "((nv_bfloat162*)(&(" << vec << "." << access[i / 2] << ")))->" << access[i % 2];
    }
  } else if (t.lanes() > 4 && t.lanes() <= 8) {
    std::string type_name;
    if (t.bits() == 16) {
      if (t.is_int()) {
        type_name = "short";
      } else if (t.is_uint()) {
        type_name = "ushort";
      }
    } else if (t.bits() == 32) {
      if (t.is_int()) {
        type_name = "int";
      } else if (t.is_uint()) {
        type_name = "uint";
      } else if (t.is_float()) {
        type_name = "float";
      }
    }
    ICHECK(!type_name.empty());
    os << "((" << type_name << "2*)(&(" << vec << "." << access[i / 2] << ")))->" << access[i % 2];
  } else if (t.is_float4_e2m1fn()) {
    os << "([](__nv_fp4_storage_t v) { __nv_fp4_e2m1 t; t.__x = v; return t; })((" << vec
       << ".__x >> " << i * 4 << ") & 0xF)";
  } else {
    os << vec << "." << access[i];
  }
}

void CodeGenCUDA::PrintVecElemStore(const std::string& vec, DataType t, int i,
                                    const std::string& value) {
  this->PrintIndent();
  static const char access[] = {'x', 'y', 'z', 'w'};
  ICHECK(i >= 0 && i < (t.bits() == 8 ? 16 : (t.bits() == 16 || t.bits() == 32) ? 8 : 4));
  if (t.bits() == 8 && (t.is_int() || t.is_uint())) {
    if (t.lanes() == 2 || t.lanes() == 3) {
      stream << vec << '.' << access[i % t.lanes()] << "="
             << "(" << value << ");\n";
    } else {
      std::string ac = t.lanes() == 4 ? vec : (vec + "." + access[i / 4]);
      stream << ac << "=";
      // Do not read the first undef lane.
      if (i != 0) {
        stream << ac << " & ~(0x000000ff << " << i % 4 * 8 << ") |";
      }
      stream << "(" << value << " << " << i % 4 * 8 << ");\n";
    }
  } else if (t.is_float16()) {
    if (t.lanes() <= 4) {
      stream << vec << "." << access[i] << " = " << value << ";\n";
    } else {
      stream << "((half2*)(&(" << vec << "." << access[i / 2] << ")))->" << access[i % 2] << " = "
             << value << ";\n";
    }

  } else if (t.is_bfloat16()) {
    if (t.lanes() <= 4) {
      stream << vec << "." << access[i] << " = " << value << ";\n";
    } else {
      stream << "((nv_bfloat162*)(&(" << vec << "." << access[i / 2] << ")))->" << access[i % 2]
             << " = " << value << ";\n";
    }
  } else if (t.lanes() > 4 && t.lanes() <= 8) {
    std::string type_name;
    if (t.bits() == 16) {
      if (t.is_int()) {
        type_name = "short";
      } else if (t.is_uint()) {
        type_name = "ushort";
      }
    } else if (t.bits() == 32) {
      if (t.is_int()) {
        type_name = "int";
      } else if (t.is_uint()) {
        type_name = "uint";
      } else if (t.is_float()) {
        type_name = "float";
      }
    }
    ICHECK(!type_name.empty());
    stream << "((" << type_name << "2*)(&(" << vec << "." << access[i / 2] << ")))->"
           << access[i % 2] << " = " << value << ";\n";
  } else {
    stream << vec << "." << access[i] << " = " << value << ";\n";
  }
}

void CodeGenCUDA::PrintStorageSync(const CallNode* op) {
  const std::string& sync = op->args[0].as<StringImmNode>()->value;
  if (sync == "warp") {
    // DO nothing.
  } else if (sync == "shared" || sync == "shared.dyn") {
    this->PrintIndent();
    this->stream << "__syncthreads();\n";
  } else if (sync == "global") {
    if (!need_global_barrier_) {
      need_global_barrier_ = true;
      this->decl_stream << "extern \"C\" __device__ unsigned " << vid_global_barrier_state_
                        << ";\n";
    }
    // global synchronizer
    std::string is_load = PrintExpr(op->args[1]);
    std::string num_blocks = PrintExpr(op->args[2]);
    this->PrintIndent();
    // In theory only threadfence is needed
    // but we observed problems with only threadfence
    this->stream << "__threadfence_system();\n";
    this->PrintIndent();
    this->stream << "if (" << is_load << ") {\n";
    int wb = this->BeginScope();
    this->PrintIndent();
    this->stream << "atomicAdd(&" << vid_global_barrier_state_ << ", 1);\n";
    this->PrintIndent();
    std::string ptr = name_supply_->FreshName("pf");
    this->stream << "volatile unsigned* " << ptr << " = &" << vid_global_barrier_state_ << ";\n";
    this->PrintIndent();
    this->stream << vid_global_barrier_expect_ << " += " << num_blocks << ";\n";
    this->PrintIndent();
    this->stream << "while (" << ptr << "[0] < " << vid_global_barrier_expect_ << ");\n";
    this->EndScope(wb);
    this->PrintIndent();
    this->stream << "}\n";
    this->PrintIndent();
    this->stream << "__syncthreads();\n";
  }
}

void CodeGenCUDA::PrintStorageScope(const std::string& scope, std::ostream& os) {  // NOLINT(*)
  ICHECK_NE(scope, "global") << "Cannot allocate global memory when targeting CUDA. You must pass "
                                "all global arrays as input instead";
  if (scope == "shared") {
    os << "__shared__ ";
  } else if (scope == "shared.dyn") {
    os << "extern __shared__ ";
  }
}

std::string CodeGenCUDA::CastFromTo(std::string value, DataType from, DataType target) {
  if (from == target) return value;
  std::ostringstream os;
  os << "((";
  this->PrintType(target, os);
  os << ")";
  if (from.is_float16() && (target.is_int() || target.is_uint()) && target.bits() == 8) {
    os << "(";
    if (target.is_uint()) {
      os << "u";
    }
    os << "int)";
  }
  os << value << ")";
  return os.str();
}

void CodeGenCUDA::VisitExpr_(const CastNode* op, std::ostream& os) {
  DataType from_ty = op->value.dtype();
  DataType target_ty = op->dtype;
  ICHECK_EQ(target_ty.lanes(), from_ty.lanes());

  // Emit simple C-style type conversion.
  if (from_ty.is_scalar()) return CodeGenC::VisitExpr_(op, os);

  if (target_ty.code() == DataType::kFloat8_e3m4 || target_ty.code() == DataType::kFloat8_e4m3 ||
      target_ty.code() == DataType::kFloat8_e4m3b11fnuz ||
      target_ty.code() == DataType::kFloat8_e4m3fn ||
      target_ty.code() == DataType::kFloat8_e4m3fnuz ||
      target_ty.code() == DataType::kFloat8_e5m2 ||
      target_ty.code() == DataType::kFloat8_e5m2fnuz ||
      target_ty.code() == DataType::kFloat8_e8m0fnu ||
      target_ty.code() == DataType::kFloat4_e2m1fn ||

      from_ty.code() == DataType::kFloat8_e3m4 || from_ty.code() == DataType::kFloat8_e4m3 ||
      from_ty.code() == DataType::kFloat8_e4m3b11fnuz ||
      from_ty.code() == DataType::kFloat8_e4m3fn || from_ty.code() == DataType::kFloat8_e4m3fnuz ||
      from_ty.code() == DataType::kFloat8_e5m2 || from_ty.code() == DataType::kFloat8_e5m2fnuz ||
      from_ty.code() == DataType::kFloat8_e8m0fnu || from_ty.code() == DataType::kFloat4_e2m1fn) {
    std::ostringstream val;
    if (target_ty.code() == DataType::kBFloat && target_ty.lanes() == 2) {
      val << "cast_to_nv_bfloat162(" << PrintExpr(op->value) << ")";
    } else {
      val << "(";
      PrintType(target_ty, val);
      val << ")(" << PrintExpr(op->value) << ")";
    }
    os << val.str();
    return;
  }

  // We could emit make_float4 like calls, but the emitted code looks
  // too compact to read. Emit this as vectorized unary ops.
  std::string sret = name_supply_->FreshName("_");
  this->PrintIndent();
  this->PrintType(target_ty, stream);
  stream << ' ' << sret << ";\n";
  {
    std::string src = SSAGetID(PrintExpr(op->value), from_ty);
    for (int i = 0, lanes = from_ty.lanes(); i < lanes; ++i) {
      std::ostringstream val;
      val << "(";
      PrintType(target_ty.element_of(), val);
      val << ")(";
      PrintVecElemLoad(src, from_ty, i, val);
      val << ")";
      PrintVecElemStore(sret, target_ty, i, val.str());
    }
  }
  os << sret;
}

void CodeGenCUDA::PrintCallExtern(Type ret_type, String global_symbol, const Array<PrimExpr>& args,
                                  bool skip_first_arg, std::ostream& os) {  // NOLINT(*)
  DataType ret_dtype = GetRuntimeDataType(ret_type);
  if (ret_dtype.is_fixed_length_vector()) {
    //
    // Emit an unsupported vector call
    //
    // v = intrin_f((float4*)A[0], (float4*)B[0])
    //
    // as
    //
    // float4 __ret;
    // {
    //   float4 __arg0 = ((float4*)A)[0];
    //   float4 __arg1 = ((float4*)B)[0];
    //   __ret.x = intrin_f(__arg0.x, __arg1.x);
    //   __ret.y = intrin_f(__arg0.y, __arg1.y);
    //   __ret.z = intrin_f(__arg0.z, __arg1.z);
    //   __ret.w = intrin_f(__arg0.w, __arg1.w);
    // }
    // v = __ret;
    //
    // Declare the result vector.
    std::string sret = name_supply_->FreshName("_");
    this->PrintIndent();
    this->PrintType(ret_dtype, stream);
    stream << ' ' << sret << ";\n";
    {
      // Load arguments.
      std::vector<std::string> sargs;
      size_t arg_begin = static_cast<size_t>(skip_first_arg);
      for (size_t i = arg_begin; i < args.size(); ++i) {
        std::string val = SSAGetID(PrintExpr(args[i]), args[i].dtype());
        sargs.push_back(std::move(val));
      }

      // Emit a scalar call for each lane.
      for (int i = 0; i < ret_dtype.lanes(); ++i) {
        std::ostringstream scall;
        scall << global_symbol << "(";
        for (size_t j = 0; j < sargs.size(); ++j) {
          if (j > 0) scall << ", ";
          PrintVecElemLoad(sargs[j], args[arg_begin + j].dtype(), i, scall);
        }
        scall << ")";
        PrintVecElemStore(sret, ret_dtype, i, scall.str());
      }
    }
    os << sret;
  } else {
    CodeGenC::PrintCallExtern(ret_type, global_symbol, args, skip_first_arg, os);
  }
}

void CodeGenCUDA::VisitExpr_(const CallNode* op, std::ostream& os) {
  if (auto opt_call_opt = op->op.as<Op>()) {
    Op call_op = opt_call_opt.value();
    // This is only for backward compatibility with __shfl_{up/down}.
    // A macro will be used to replace *_sync calls to legacy ones.
    if (op_need_warp_shuffle_.get(call_op, false)) {
      enable_warp_shuffle_ = true;
    }
  }

  if (op->op.same_as(builtin::tvm_fill_fragment())) {
    need_mma_h_ = true;
    ICHECK_EQ(op->args.size(), 6U);
    os << "nvcuda::wmma::fill_fragment(";
    this->PrintExpr(op->args[0], os);
    os << "[";
    this->PrintExpr(op->args[4], os);
    os << "], ";
    this->PrintExpr(op->args[5], os);
    os << ")";
  } else if (op->op.same_as(builtin::tvm_load_matrix_sync())) {
    need_mma_h_ = true;
    ICHECK_EQ(op->args.size(), 8U);
    os << "nvcuda::wmma::load_matrix_sync(";
    this->PrintExpr(op->args[0], os);
    os << "[";
    this->PrintExpr(op->args[4], os);
    os << "], ";
    this->PrintExpr(op->args[5], os);
    os << ", ";
    this->PrintExpr(op->args[6], os);
    os << ")";
  } else if (op->op.same_as(builtin::tvm_store_matrix_sync())) {
    need_mma_h_ = true;
    ICHECK_EQ(op->args.size(), 8U);
    os << "nvcuda::wmma::store_matrix_sync(";
    this->PrintExpr(op->args[5], os);
    os << ", ";
    this->PrintExpr(op->args[0], os);
    os << "[";
    this->PrintExpr(op->args[4], os);
    os << "], ";
    this->PrintExpr(op->args[6], os);
    if (const StringImmNode* str = op->args[7].as<StringImmNode>()) {
      os << ", nvcuda::wmma::mem_" << str->value;
    } else {
      LOG(FATAL) << "Invalid parameters";
    }
    os << ")";
  } else if (op->op.same_as(builtin::tvm_mma_sync())) {
    need_mma_h_ = true;
    ICHECK_EQ(op->args.size(), 8U);
    os << "nvcuda::wmma::mma_sync(";
    for (int i = 0; i < 4; ++i) {
      this->PrintExpr(op->args[i * 2], os);
      os << "[";
      this->PrintExpr(op->args[i * 2 + 1], os);
      os << "]" << ((i < 3) ? ", " : ")");
    }
  } else if (op->op.same_as(builtin::tvm_bmma_sync())) {
    need_mma_h_ = true;
    ICHECK_EQ(op->args.size(), 8U);
    os << "nvcuda::wmma::bmma_sync(";
    for (int i = 0; i < 4; ++i) {
      this->PrintExpr(op->args[i * 2], os);
      os << "[";
      this->PrintExpr(op->args[i * 2 + 1], os);
      os << "]" << ((i < 3) ? ", " : ")");
    }
  } else if (op->op.same_as(builtin::ptx_mma())) {
    // arg 0: shape: mXnXkX
    // arg 1: A layout: row/col
    // arg 2: B layout: row/col
    // arg 3: A precision: fp16, fp64, ...
    // arg 4: B precision: fp16, fp64, ...
    // arg 5: C precision: fp32, fp64, ...
    // arg 6: A multiplicand
    // arg 7: A multiplicand index
    // arg 8: B multiplicand
    // arg 9: B multiplicand index
    // arg 10: C accumulator
    // arg 11: C accumulator index
    // arg 12: saturate
    // arg 13: (optional) 1-bit operator (xor or and)
    ICHECK(op->args.size() == 13U || op->args.size() == 14U);
    std::string shape = Downcast<StringImm>(op->args[0])->value;
    std::string A_layout = Downcast<StringImm>(op->args[1])->value;
    std::string B_layout = Downcast<StringImm>(op->args[2])->value;
    std::string A_dtype = Downcast<StringImm>(op->args[3])->value;
    std::string B_dtype = Downcast<StringImm>(op->args[4])->value;
    std::string C_dtype = Downcast<StringImm>(op->args[5])->value;
    std::string a_ref = this->PrintExpr(op->args[6]);
    std::string a_bias = this->PrintExpr(op->args[7]);
    std::string b_ref = this->PrintExpr(op->args[8]);
    std::string b_bias = this->PrintExpr(op->args[9]);
    std::string c_ref = this->PrintExpr(op->args[10]);
    std::string c_bias = this->PrintExpr(op->args[11]);
    bool saturate = Downcast<Bool>(op->args[12])->value;
    std::string bit_op = op->args.size() > 13 ? Downcast<StringImm>(op->args[13])->value : "";
    std::string asm_code =
        PrintMMAAssembly(shape, A_layout, B_layout, A_dtype, B_dtype, C_dtype, a_ref, a_bias, b_ref,
                         b_bias, c_ref, c_bias, "", "", "", bit_op, false, saturate);

    this->stream << asm_code;
  } else if (op->op.same_as(builtin::ptx_mma_sp())) {
    // arg 0: shape: mXnXkX
    // arg 1: A layout: row/col
    // arg 2: B layout: row/col
    // arg 3: A precision: fp16, fp32, ...
    // arg 4: B precision: fp16, fp32, ...
    // arg 5: C precision: fp16, fp32, ...
    // arg 6: A multiplicand pointer
    // arg 7: A multiplicand index
    // arg 8: B multiplicand pointer
    // arg 9: B multiplicand index
    // arg 10: C accumulator pointer
    // arg 11: C accumulator index
    // arg 12: metadata
    // arg 13: metadata index
    // arg 14: sparse_selector
    // arg 15: saturate
    ICHECK_EQ(op->args.size(), 16U);
    std::string shape = Downcast<StringImm>(op->args[0])->value;
    std::string A_layout = Downcast<StringImm>(op->args[1])->value;
    std::string B_layout = Downcast<StringImm>(op->args[2])->value;
    std::string A_dtype = Downcast<StringImm>(op->args[3])->value;
    std::string B_dtype = Downcast<StringImm>(op->args[4])->value;
    std::string C_dtype = Downcast<StringImm>(op->args[5])->value;
    std::string a_ref = this->PrintExpr(op->args[6]);
    std::string a_offset = this->PrintExpr(op->args[7]);
    std::string b_ref = this->PrintExpr(op->args[8]);
    std::string b_offset = this->PrintExpr(op->args[9]);
    std::string c_ref = this->PrintExpr(op->args[10]);
    std::string c_offset = this->PrintExpr(op->args[11]);
    std::string metadata = this->PrintExpr(op->args[12]);
    std::string metadata_offset = this->PrintExpr(op->args[13]);
    std::string sparse_selector = this->PrintExpr(op->args[14]);
    bool saturate = Downcast<Bool>(op->args[15])->value;
    std::string asm_code = PrintMMAAssembly(
        shape, A_layout, B_layout, A_dtype, B_dtype, C_dtype, a_ref, a_offset, b_ref, b_offset,
        c_ref, c_offset, metadata, metadata_offset, sparse_selector, "", true, saturate);
    this->stream << asm_code;
  } else if (op->op.same_as(builtin::ptx_ldmatrix())) {
    // arg 0: whether the matrix is loaded in column major format or not.
    // arg 1: number of matrices to load.
    // arg 2: The data type in the matrix, .b16 is the only accepted data type.
    // arg 3: pointer to local buffer.
    // arg 4: The offset of the element to store in the local buffer.
    // arg 5: pointer to the shared memory buffer to load.
    // arg 6: The offset of the start element of the row to load in shared memory.
    ICHECK_EQ(op->args.size(), 7U);
    bool trans = Downcast<Bool>(op->args[0])->value;
    int num = Downcast<Integer>(op->args[1])->value;
    std::string type = Downcast<StringImm>(op->args[2])->value;
    std::string local_ptr = this->PrintExpr(op->args[3]);
    std::string local_elem_offset = this->PrintExpr(op->args[4]);
    std::string smem_ptr = this->PrintExpr(op->args[5]);
    if (trans && op->dtype.bits() == 8) {
      // Since ldmatrix assumes that a matrix element is 16 bit, it cannot properly transpose an
      // int8 matrix.
      std::string smem_stride = this->PrintExpr(op->args[6]);
      ICHECK(num == 4);
      os << "for (int i = 0; i < 16; ++i) {\n";
      os << local_ptr << "[" + local_elem_offset + " + i] = " << smem_ptr
         << "[(i % 8) / 4 * " + smem_stride + " * 16 + (threadIdx.x % 4) * 4 * " + smem_stride +
                "+ (i % 4) * " + smem_stride + " + threadIdx.x / 4 +  (i / 8) * 8];\n";
      os << "}\n";
    } else {
      std::string smem_elem_offset = this->PrintExpr(op->args[6]);
      need_cast_smem_ptr_to_int_ = true;
      this->stream << PrintLoadMatrixAssembly(trans, num, type, local_ptr, local_elem_offset,
                                              smem_ptr, smem_elem_offset);
    }
  } else if (op->op.same_as(builtin::mma_store())) {
    int m = Downcast<Integer>(op->args[0])->value;
    int n = Downcast<Integer>(op->args[1])->value;
    std::string dst = this->PrintExpr(op->args[2]);
    std::string src = this->PrintExpr(op->args[3]);
    std::string src_offset = this->PrintExpr(op->args[4]);
    PrimExpr stride = op->args[5];

    ICHECK(m == 16 && n == 16) << "Only m == 16 && n == 16 case supported for now";

    // Each thread in a warp holds a certain number of elements of an MMA output.
    // For example, if we compute a 16x16 tile using MMA, each thread holds 8 elements
    // in its registers. So conceptually, a warp memory is organized as a 32x8 block.
    // A map from a 16x16 tile to a 32x8 block of memory is specified by the index map below.

    // To store the 32x8 output back to a 16x16 tile in shared or global memory, we invert this map
    // to determine the output location for each 8 element.

    const auto index_map_func =
        tvm::ffi::Function::GetGlobal("tir.index_map.shared_16x16_to_ldmatrix_32x8_layout");
    ICHECK(index_map_func.has_value());

    arith::Analyzer analyzer;
    auto inverse_index_map =
        IndexMap::FromFunc(2, *index_map_func).Inverse({Range(0, m), Range(0, n)}, &analyzer);
    auto indices_16x16 = inverse_index_map->final_indices;

    // "//" and "%" in the index map are translated to FloorDiv/Mod, but the plain Div/Mod are fine.
    // FloorDiv/Mod are supposed to be lowered before they reach codegen, so manually replace them
    // to the plain ones here.
    class LowerFloorDivMod : public ExprMutator {
     public:
      PrimExpr VisitExpr_(const FloorDivNode* op) {
        return tir::Div(this->VisitExpr(op->a), this->VisitExpr(op->b));
      }
      PrimExpr VisitExpr_(const FloorModNode* op) {
        return tir::Mod(this->VisitExpr(op->a), this->VisitExpr(op->b));
      }
    };

    auto dst_ind = LowerFloorDivMod()(indices_16x16[0] * stride + indices_16x16[1]);

    var_idmap_[inverse_index_map->initial_indices[0].get()] = "threadIdx.x";
    var_idmap_[inverse_index_map->initial_indices[1].get()] = "local_id";

    os << "for (int local_id = 0; local_id < 8; ++local_id) {\n";
    os << dst << "[" + this->PrintExpr(dst_ind) + "] = " << src << "[" << src_offset
       << " + local_id];\n";
    os << "}\n";

  } else if (op->op.same_as(builtin::mma_fill())) {
    std::string num_elem = this->PrintExpr(op->args[0]);
    std::string dst = this->PrintExpr(op->args[1]);
    std::string dst_offset = this->PrintExpr(op->args[2]);

    os << "for (int i = 0; i < " << num_elem << "; ++i) {\n";
    os << dst << "[" << dst_offset << " + i] = 0.0;";
    os << "}\n";
  } else if (op->op.same_as(builtin::ptx_cp_async())) {
    std::string dst = this->PrintExpr(op->args[0]);
    std::string dst_offset = this->PrintExpr(op->args[1]);
    std::string src = this->PrintExpr(op->args[2]);
    std::string src_offset = this->PrintExpr(op->args[3]);
    std::string size = this->PrintExpr(op->args[4]);
    need_cast_smem_ptr_to_int_ = true;
    // use size of argument list to indicate whether or not to use predicated cp.async
    if (op->args.size() == 5) {
      this->stream << PrintCpAsyncAssembly(dst, dst_offset, src, src_offset, size);
    } else {
      this->stream << PrintPredicatedCpAsyncAssembly(dst, dst_offset, src, src_offset, size,
                                                     this->PrintExpr(op->args[5]));
    }
  } else if (op->op.same_as(builtin::ptx_cp_async_bulk())) {
    need_cast_smem_ptr_to_int_ = true;
    std::string dst = this->PrintExpr(op->args[0]);
    std::string dst_offset = this->PrintExpr(op->args[1]);
    std::string src = this->PrintExpr(op->args[2]);
    std::string src_offset = this->PrintExpr(op->args[3]);
    std::string size = this->PrintExpr(op->args[4]);
    int barrier_id = Downcast<IntImm>(op->args[5])->value;
    CHECK(barrier_id < barrier_count_);
    std::string barrier = barrier_name_ + "[" + std::to_string(barrier_id) + "]";
    this->stream << PrintCpAsyncBulkAsm(dst, dst_offset, src, src_offset, size, barrier);
  } else if (op->op.same_as(builtin::ptx_commit_group())) {
    this->stream << "__asm__ __volatile__(\"cp.async.commit_group;\");\n\n";
  } else if (op->op.same_as(builtin::ptx_wait_group())) {
    int n = Downcast<IntImm>(op->args[0])->value;
    this->stream << "__asm__ __volatile__(\"cp.async.wait_group " << n << ";\");\n\n";
  } else if (op->op.same_as(builtin::ptx_cp_async_barrier())) {
    need_cast_smem_ptr_to_int_ = true;
    int barrier_id = Downcast<IntImm>(op->args[0])->value;
    CHECK(barrier_id < barrier_count_);
    std::string barrier = barrier_name_ + "[" + std::to_string(barrier_id) + "]";
    this->stream << PrintCpAsyncBarrierAsm(barrier);
  } else if (op->op.same_as(builtin::ptx_init_barrier_thread_count())) {
    need_cast_smem_ptr_to_int_ = true;
    int barrier_id = Downcast<IntImm>(op->args[0])->value;
    CHECK(barrier_id < barrier_count_);
    std::string barrier = barrier_name_ + "[" + std::to_string(barrier_id) + "]";
    std::string thread_count = this->PrintExpr(op->args[1]);
    this->stream << PrintInitBarrierThreadCountAsm(barrier, thread_count);
  } else if (op->op.same_as(builtin::ptx_arrive_barrier())) {
    need_cast_smem_ptr_to_int_ = true;
    int barrier_id = Downcast<IntImm>(op->args[0])->value;
    CHECK(barrier_id < barrier_count_);
    std::string barrier = barrier_name_ + "[" + std::to_string(barrier_id) + "]";
    this->stream << PrintArriveBarrierAsm(barrier);
  } else if (op->op.same_as(builtin::ptx_arrive_barrier_expect_tx())) {
    need_cast_smem_ptr_to_int_ = true;
    int barrier_id = Downcast<IntImm>(op->args[0])->value;
    CHECK(barrier_id < barrier_count_);
    std::string barrier = barrier_name_ + "[" + std::to_string(barrier_id) + "]";
    std::string byte_count = this->PrintExpr(op->args[1]);
    this->stream << PrintArriveBarrierExpectTxAsm(barrier, byte_count);
  } else if (op->op.same_as(builtin::ptx_wait_barrier())) {
    need_cast_smem_ptr_to_int_ = true;
    int barrier_id = Downcast<IntImm>(op->args[0])->value;
    CHECK(barrier_id < barrier_count_);
    std::string barrier = barrier_name_ + "[" + std::to_string(barrier_id) + "]";
    this->stream << PrintWaitBarrierAsm(barrier);
  } else if (op->op.same_as(builtin::create_barriers())) {
    CHECK_EQ(barrier_count_, -1);
    int barrier_count = Downcast<IntImm>(op->args[0])->value;
    // pad barrier alignment to avoid runtime alignment errors
    CHECK_EQ(barrier_alignment_bytes_ % sizeof(uint64_t), 0);
    int barrier_alignment_count = barrier_alignment_bytes_ / sizeof(uint64_t);
    if (barrier_count % barrier_alignment_count != 0) {
      barrier_count = ((barrier_count / barrier_alignment_count) + 1) * barrier_alignment_count;
    }
    barrier_count_ = barrier_count;
    this->stream << "__shared__ __align__(" << barrier_alignment_bytes_ << ") uint64_t "
                 << barrier_name_ << "[" << barrier_count << "];\n";
    this->stream << "for (int i = 0; i < " << barrier_count << "; ++i) { " << barrier_name_
                 << "[i] = 0; }\n";
  } else if (op->op.same_as(builtin::ptx_ldg32())) {
    /*
    asm volatile (
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %2, 0;\n"
        // " @p ld.global.nc.f32 %0, [%1];}\n"t
        " @p ld.global.nc.L2::128B.f32 %0, [%1];}\n"
        : "=f"(reg)
        : "l"(addr), "r"((int)guard)
    );
    */

    // get local
    std::string reg = this->PrintExpr(op->args[0]);
    // get guard
    std::string guard = this->PrintExpr(op->args[1]);
    const BufferLoadNode* addr_buffer = op->args[2].as<BufferLoadNode>();
    std::string global_addr = this->PrintExpr(addr_buffer->indices[0]);
    std::string global_buffer = this->PrintExpr(addr_buffer->buffer->data);
    std::string local_addr = this->PrintExpr(op->args[3]);
    this->stream << "asm volatile (\n";
    this->stream << "\"{.reg .pred p;\\n\"\n";
    this->stream << "\" setp.ne.b32 p, %2, 0;\\n\"\n";
    this->stream << "\" @!p mov.b32 %0, 0;\\n\"\n";
    this->stream << "\" @p ld.global.nc.f32 %0, [%1];}\\n\"\n";
    // stream << "\" @p ld.global.nc.L2::128B.f32 %0, [%1];}\\n\"\n" ;
    stream << ": \"=f\"(" << reg << "[" << local_addr << "]"
           << ")\n";
    stream << ": \"l\"((void*)(" << global_buffer << "+" << global_addr << ")), \"r\"((int)"
           << guard << ")\n";
    stream << ");\n";
  } else if (op->op.same_as(builtin::reinterpret())) {
    DataType tgt_dtype = op->dtype;
    DataType src_dtype = op->args[0]->dtype;
    PrimExpr value = op->args[0];

    // Handle float4_e2m1fn reinterpret
    if (!src_dtype.is_float4_e2m1fn() && !tgt_dtype.is_float4_e2m1fn()) {
      return CodeGenC::VisitExpr_(op, os);
    }
    if (src_dtype == tgt_dtype ||
        tgt_dtype.lanes() * tgt_dtype.bits() == src_dtype.lanes() * src_dtype.bits()) {
      return CodeGenC::VisitExpr_(op, os);
    }
    CHECK_EQ(tgt_dtype.lanes(), src_dtype.lanes())
        << "E2M1 float4 reinterpret expects source and target to have the same number of lanes. "
        << "Source dtype: " << src_dtype << ", Target dtype: " << tgt_dtype;
    CHECK_EQ(tgt_dtype.bytes(), src_dtype.bytes())
        << "E2M1 float4 reinterpret expects source and target to have the same number of bytes. "
        << "Source dtype: " << src_dtype << ", Target dtype: " << tgt_dtype;

    int lanes = tgt_dtype.lanes();

    int ssa_scope = BeginScope();
    if (lanes == 1) {
      // The case of lane=1 is same as the normal reinterpret,
      // except that we allow the src and dst dtype to have different number of bits.
      std::string rhs = SSAGetID(PrintExpr(value), src_dtype);
      os << "(*(";
      this->PrintType(tgt_dtype, os);
      os << " *)(&(" << rhs << ")))";
    } else if (lanes == 2) {
      if (tgt_dtype.is_float4_e2m1fn()) {
        // We view the source as an uint16, and then extract bits of two fp4 numbers,
        // and finally reinterpret the result as fp4x2.
        value = tir::Call(DataType::UInt(16), tir::builtin::reinterpret(), {value});
        tir::Var temp_var("temp_var", DataType::UInt(16));
        value = tir::Let(
            temp_var, value,
            tir::Cast(DataType::UInt(8), (temp_var & IntImm(DataType::UInt(16), 0xF)) |
                                             ((temp_var >> 4) & IntImm(DataType::UInt(16), 0xF0))));
      } else {
        value = tir::Cast(DataType::UInt(16),
                          tir::Call(DataType::UInt(8), tir::builtin::reinterpret(), {value}));
        tir::Var temp_var("temp_var", DataType::UInt(16));
        value = tir::Let(temp_var, value,
                         (temp_var & IntImm(DataType::UInt(16), 0xF)) |
                             ((temp_var & IntImm(DataType::UInt(16), 0xF0)) << 4));
      }
      os << PrintExpr(tir::Call(tgt_dtype, tir::builtin::reinterpret(), {value}));
    } else if (lanes == 4) {
      if (tgt_dtype.is_float4_e2m1fn()) {
        // We view the source as an uint32, and then extract bits of four fp4 numbers,
        // and finally reinterpret the result as fp4x4.
        value = tir::Call(DataType::UInt(32), tir::builtin::reinterpret(), {value});
        tir::Var temp_var("temp_var", DataType::UInt(32));
        value = tir::Let(temp_var, value,
                         tir::Cast(DataType::UInt(16),
                                   (temp_var & IntImm(DataType::UInt(32), 0xF)) |
                                       ((temp_var >> 4) & IntImm(DataType::UInt(32), 0xF0)) |
                                       ((temp_var >> 8) & IntImm(DataType::UInt(32), 0xF00)) |
                                       ((temp_var >> 12) & IntImm(DataType::UInt(32), 0xF000))));
      } else {
        value = tir::Cast(DataType::UInt(32),
                          tir::Call(DataType::UInt(16), tir::builtin::reinterpret(), {value}));
        tir::Var temp_var("temp_var", DataType::UInt(32));
        value = tir::Let(temp_var, value,
                         (temp_var & IntImm(DataType::UInt(32), 0xF)) |
                             ((temp_var & IntImm(DataType::UInt(32), 0xF0)) << 4) |
                             ((temp_var & IntImm(DataType::UInt(32), 0xF00)) << 8) |
                             ((temp_var & IntImm(DataType::UInt(32), 0xF000)) << 12));
      }
      os << PrintExpr(tir::Call(tgt_dtype, tir::builtin::reinterpret(), {value}));
    } else {
      LOG(FATAL) << "Invalid number of lanes for float4_e2m1fn reinterpret: " << lanes;
    }
    EndScope(ssa_scope);
  } else if (op->op.same_as(builtin::thread_return())) {
    os << "return";
  } else {
    CodeGenC::VisitExpr_(op, os);
  }
}

void CodeGenCUDA::VisitStmt_(const AttrStmtNode* op) {
  if (op->attr_key == tir::attr::fragment_shape) {
    const VarNode* buffer = op->node.as<VarNode>();
    const StringImmNode* shape_str = op->value.as<StringImmNode>();
    fragment_shapes[buffer] = shape_str->value;
  } else if (op->attr_key == tir::attr::fragment_layout) {
    const VarNode* buffer = op->node.as<VarNode>();
    const StringImmNode* layout_str = op->value.as<StringImmNode>();
    fragment_layouts[buffer] = layout_str->value;
  } else if (op->attr_key == tir::attr::async_commit_queue_scope) {
    const IntImmNode* queue_id = op->value.as<IntImmNode>();
    ICHECK(queue_id && queue_id->value == 0) << "For CUDA, the index of an async queue must be 0.";
    this->VisitStmt(op->body);
    auto commit_group = Call(DataType::Void(), builtin::ptx_commit_group(), {});
    this->VisitExpr(commit_group, this->stream);
    return;
  } else if (op->attr_key == tir::attr::async_wait_queue_scope) {
    auto wait_attrs = GetAsyncWaitAttributes(op);
    auto queue_id = wait_attrs.first.as<IntImmNode>();
    ICHECK(queue_id && queue_id->value == 0) << "For CUDA, the index of an async queue must be 0.";
    auto wait_cnt = wait_attrs.second;
    auto wait_group = Call(DataType::Void(), builtin::ptx_wait_group(), {wait_cnt});
    this->VisitExpr(wait_group, this->stream);
    auto inner = op->body.as<AttrStmtNode>();
    ICHECK(inner);
    this->VisitStmt(inner->body);
    return;
  }
  CodeGenC::VisitStmt_(op);
}

void CodeGenCUDA::VisitStmt_(const AllocateNode* op) {
  ICHECK(!is_zero(op->condition));
  std::string vid = AllocVarID(op->buffer_var.get());

  this->PrintIndent();
  std::string scope = GetPtrStorageScope(op->buffer_var);
  const VarNode* buffer = op->buffer_var.as<VarNode>();
  if (scope.find("wmma.") == 0) {
    if (scope == "wmma.matrix_a" || scope == "wmma.matrix_b") {
      ICHECK(op->dtype == DataType::Float(16) || op->dtype == DataType::Int(8) ||
             op->dtype == DataType::UInt(8) || op->dtype == DataType::Int(4) ||
             op->dtype == DataType::UInt(4) || op->dtype == DataType::Int(1) ||
             op->dtype == DataType::BFloat(16))
          << "Matrix_a and matrix_b only support half or char or unsigned char "
          << "or uint4 or int4 or int1 type for now";
    } else {
      ICHECK(op->dtype == DataType::Float(16) || op->dtype == DataType::Float(32) ||
             op->dtype == DataType::Int(32))
          << "Accumulator only support half, float and int type for now";
    }
    PrintWmmaScope(scope, op->dtype, buffer, stream);
  } else {
    PrintStorageScope(scope, stream);
    PrintType(op->dtype, stream);
  }

  if (scope == "shared.dyn") {
    stream << ' ' << vid << "[];\n";
  } else {
    size_t constant_size = op->ConstantAllocationSize();
    ICHECK_GT(constant_size, 0) << "Can only handle constant size stack allocation for now";

    if (scope.find("wmma.") == 0) {
      constant_size = GetWmmaFragmentSize(scope, buffer, constant_size);
    }
    if ((op->dtype == DataType::Int(4) || op->dtype == DataType::UInt(4) ||
         op->dtype == DataType::Int(1)) &&
        scope == "shared") {
      constant_size = constant_size / (32 / op->dtype.bits());
    }
    stream << ' ' << vid << '[' << constant_size << "];\n";
  }

  RegisterHandleType(op->buffer_var.get(), op->dtype);
  this->PrintStmt(op->body);
}

void CodeGenCUDA::VisitStmt_(const EvaluateNode* op) {
  if (is_const_int(op->value)) return;
  const CallNode* call = op->value.as<CallNode>();
  if (call && call->op.same_as(builtin::tvm_global_barrier_kinit())) {
    PrintIndent();
    stream << "__shared__ unsigned " << vid_global_barrier_expect_ << ";\n";
    PrintIndent();
    stream << "if (threadIdx.x == 0) {\n";
    PrintIndent();
    stream << "  " << vid_global_barrier_expect_ << " = 0;\n";
    PrintIndent();
    stream << "}\n";
  } else {
    CodeGenC::VisitStmt_(op);
  }
}

void CodeGenCUDA::VisitExpr_(const RampNode* op, std::ostream& os) {
  int lanes = op->dtype.lanes();
  CHECK_LE(lanes, 4) << "ValueError: Ramp of more than 4 lanes is not allowed.";
  PrintVecConstructor(op->dtype, os);
  os << "(";
  for (int i = 0; i < lanes; i++) {
    os << "(" << PrintExpr(op->base) << ")"
       << "+(" << PrintExpr(op->stride) << "*" << i << ")";
    if (i != lanes - 1) os << ", ";
  }
  os << ")";
}

void CodeGenCUDA::VisitExpr_(const BroadcastNode* op, std::ostream& os) {  // NOLINT(*)
  int lanes = op->dtype.lanes();
  if ((op->dtype.is_int() || op->dtype.is_uint()) && op->dtype.bits() == 8 && lanes == 4) {
    // make_int8x4
    const int64_t* p = as_const_int(op->value);
    ICHECK(p);
    int64_t v = *p & 0xFF;
    v = (v << 24) | (v << 16) | (v << 8) | v;
    if (op->dtype.is_uint()) {
      os << "(uint)" << v;
    } else {
      os << "(int)" << v;
    }
    return;
  }

  if (op->dtype.is_float16()) {
    std::string v = PrintExpr(op->value);
    PrintVecConstructor(op->dtype, os);
    os << '(';
    if (lanes <= 4) {
      for (int i = 0; i < lanes / 2; ++i) {
        if (i != 0) os << ", ";
        os << v << ", " << v;
      }
    } else {
      for (int i = 0; i < lanes / 2; ++i) {
        if (i != 0) os << ", ";
        os << "__pack_half2(" << v << ", " << v << ")";
      }
    }
    os << ')';
    return;
  }

  if (op->dtype.is_bfloat16()) {
    std::string v = PrintExpr(op->value);
    PrintVecConstructor(op->dtype, os);
    os << '(';
    if (lanes > 4) {
      for (int i = 0; i < lanes / 2; ++i) {
        if (i != 0) os << ", ";
        os << "__pack_nv_bfloat162(" << v << ", " << v << ")";
      }
    } else {
      for (int i = 0; i < lanes; ++i) {
        if (i != 0) os << ", ";
        os << v;
      }
    }
    os << ')';
    return;
  }

  if (op->dtype.is_float8() || op->dtype.is_float4()) {
    int lanes = op->dtype.lanes();
    ICHECK(lanes == 1 || lanes == 2 || lanes == 4);
    std::string v = PrintExpr(op->value);
    // Implicit conversion from float back to fp8
    PrintType(op->dtype, os);
    os << "(make_float" << lanes << "(";
    for (int i = 0; i < lanes; ++i) {
      if (i != 0) os << ", ";
      os << "static_cast<float>(" << v << ")";
    }
    os << "))";
    return;
  }

  if ((op->dtype.is_int() || op->dtype.is_uint()) && op->dtype.bits() == 4) {
    bool fail = false;
    const int64_t* p = as_const_int(op->value);
    ICHECK(p);
    int64_t v = *p & 0xF;

    if (lanes == 4) {
      v = (v << 12) | (v << 8) | (v << 4) | v;
      if (op->dtype.is_uint()) {
        os << "(uint16_t)" << v;
      } else {
        os << "(int16_t)" << v;
      }
    } else {
      v = (v << 28) | (v << 24) | (v << 20) | (v << 16) | (v << 12) | (v << 8) | (v << 4) | v;
      if (lanes == 8) {
        if (op->dtype.is_uint()) {
          os << "(uint)" << v;
        } else {
          os << "(int)" << v;
        }
      } else if (lanes == 16 || lanes == 32) {
        PrintVecConstructor(op->dtype, os);
        os << '(';
        for (int i = 0; i < lanes / 8; ++i) {
          if (i != 0) os << ", ";
          if (op->dtype.is_uint()) {
            os << "(uint)" << v;
          } else {
            os << "(int)" << v;
          }
        }
        os << ')';
      } else {
        fail = true;
      }
    }

    if (!fail) {
      return;
    }
  }

  std::string v = PrintExpr(op->value);
  PrintVecConstructor(op->dtype, os);
  os << '(';
  for (int i = 0; i < lanes; ++i) {
    if (i != 0) os << ", ";
    os << v;
  }
  os << ')';
}

void CodeGenCUDA::VisitExpr_(const SelectNode* op, std::ostream& os) {
  // Non-vector cases.
  if (!op->dtype.is_fixed_length_vector()) {
    CodeGenC::VisitExpr_(op, os);
    return;
  }

  // Codegen vector condition case by serializing the select op.
  ICHECK(op->false_value->dtype == op->dtype && op->true_value->dtype == op->dtype &&
         op->dtype.lanes() == op->condition.dtype().lanes());

  std::string r_var = name_supply_->FreshName("_");
  this->PrintIndent();
  this->PrintType(op->dtype, stream);
  stream << ' ' << r_var << ";\n";
  {
    std::string c_var = SSAGetID(PrintExpr(op->condition), op->dtype);
    std::string t_var = SSAGetID(PrintExpr(op->true_value), op->dtype);
    std::string f_var = SSAGetID(PrintExpr(op->false_value), op->dtype);

    // The condition is stored as an ushort vector.
    int lanes = op->dtype.lanes();
    DataType memory_ty(DataType::TypeCode::kUInt, 16, lanes);

    for (int i = 0; i < lanes; ++i) {
      std::ostringstream item;
      item << "(bool(";
      PrintVecElemLoad(c_var, memory_ty, i, item);
      item << ")?";
      PrintVecElemLoad(t_var, op->dtype, i, item);
      item << ':';
      PrintVecElemLoad(f_var, op->dtype, i, item);
      item << ')';
      PrintVecElemStore(r_var, op->dtype, i, item.str());
    }
  }
  os << r_var;
}

inline void PrintConst(const FloatImmNode* op, std::ostream& os, CodeGenCUDA* p) {  // NOLINT(*)
  // Type code is kBFloat
  if (op->dtype.is_bfloat16()) {
    os << "__float2bfloat16_rn";
    os << '(' << std::scientific << op->value << 'f' << ')';
    return;
  }
  // Type code is kFloat8_e5m2 or kE4M4Float
  if (op->dtype.is_float8() || op->dtype.is_float4()) {
    p->PrintType(op->dtype, os);
    os << '(' << std::scientific << op->value << 'f' << ')';
    return;
  }
  // Type code is kFloat
  switch (op->dtype.bits()) {
    case 64: {
      std::ostringstream temp;
      if (std::isinf(op->value)) {
        if (op->value < 0) {
          temp << "-";
        }
        temp << "CUDART_INF";
        p->need_math_constants_h_ = true;
      } else if (std::isnan(op->value)) {
        temp << "CUDART_NAN";
        p->need_math_constants_h_ = true;
      } else {
        temp << std::fixed << std::setprecision(15) << op->value;
      }
      p->MarkConst(temp.str());
      os << temp.str();
      break;
    }
    case 32: {
      std::ostringstream temp;
      if (std::isinf(op->value)) {
        if (op->value < 0) {
          temp << "-";
        }
        temp << "CUDART_INF_F";
        p->need_math_constants_h_ = true;
      } else if (std::isnan(op->value)) {
        temp << "CUDART_NAN_F";
        p->need_math_constants_h_ = true;
      } else {
        temp << std::scientific << op->value << 'f';
      }
      p->MarkConst(temp.str());
      os << temp.str();
      break;
    }
    case 16: {
      os << "__float2half_rn" << '(';
      FloatImm const_f32 = FloatImm(DataType::Float(32), op->value);
      PrintConst(const_f32.get(), os, p);
      os << ')';
      break;
    }
    default:
      LOG(FATAL) << "Bad bit-width for float: " << op->dtype << "\n";
  }
}

void CodeGenCUDA::VisitExpr_(const FloatImmNode* op, std::ostream& os) {  // NOLINT(*)
  PrintConst(op, os, this);
}

void CodeGenCUDA::PrintWmmaScope(const std::string& scope, DataType t, const VarNode* variable,
                                 std::ostream& os) {
  std::stringstream type;
  PrintType(t, type);
  ICHECK(fragment_shapes.count(variable))
      << "Cannot find shape of the wmma fragment " << variable->name_hint;
  std::string shape_str = fragment_shapes.at(variable);
  if ((t.is_int() || t.is_uint()) && t.bits() < 8 && t.lanes() == 1) {
    type.str(std::string());
    if (t.is_int()) {
      if (t.bits() == 4) {
        type << "nvcuda::wmma::experimental::precision::s4";
      } else if (t.bits() == 1) {
        type << "nvcuda::wmma::experimental::precision::b1";
      } else {
        LOG(FATAL) << "Unhandled interger type for wmma fragment!";
      }
    } else if (t.is_uint()) {
      if (t.bits() == 4) {
        type << "nvcuda::wmma::experimental::precision::u4";
      } else {
        LOG(FATAL) << "Unhandled interger type for wmma fragment!";
      }
    }
  }
  if (scope == "wmma.matrix_a") {
    need_mma_h_ = true;
    std::string layout_str = fragment_layouts[variable];
    ICHECK_NE(layout_str, "") << "Layout must be defined for matrix_a";
    os << "nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, " << shape_str << ", " << type.str()
       << ", nvcuda::wmma::" << layout_str << ">";
  } else if (scope == "wmma.matrix_b") {
    need_mma_h_ = true;
    std::string layout_str = fragment_layouts[variable];
    ICHECK_NE(layout_str, "") << "Layout must be defined for matrix_b";
    os << "nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, " << shape_str << ", " << type.str()
       << ", nvcuda::wmma::" << layout_str << ">";
  } else if (scope == "wmma.accumulator") {
    need_mma_h_ = true;
    os << "nvcuda::wmma::fragment<nvcuda::wmma::accumulator, " << shape_str << ", " << type.str()
       << ">";
  }
}

int stoi(const std::string& str) {
  try {
    return std::stoi(str);
  } catch (std::invalid_argument& e) {
    LOG(FATAL) << "Cannot convert \"" << str << "\" to int";
    throw;
  }
}

int32_t CodeGenCUDA::GetWmmaFragmentSize(const std::string& scope, const VarNode* variable,
                                         int32_t size) {
  ICHECK(fragment_shapes.count(variable))
      << "Cannot find shape of the wmma fragment " << variable->name_hint;
  std::string shape_str = fragment_shapes.at(variable);
  std::pair<int32_t, int32_t> dim = GetWmmaFragmentDimSize(shape_str, scope);
  if (dim.first * dim.second != 0)
    return size / dim.first / dim.second;
  else
    return 0;
}

void CodeGenCUDA::HandleVolatileLoads(const std::string& value, const BufferLoadNode* op,
                                      std::ostream& os) {
  // Cast away volatile qualifier for fp16 types. That is, only loads and
  // stores are volatile. The loaded objects are not marked as volatile.
  //
  if ((op->dtype.is_float16() || op->dtype.is_bfloat16()) && IsVolatile(op->buffer->data.get())) {
    os << "(";
    PrintType(op->dtype, os);
    os << ")(" << value << ")";
  } else {
    os << value;
  }
}

void CodeGenCUDA::PrintVecElemLoadExpr(DataType t, int i, const std::string& value,
                                       std::ostream& os) {
  ICHECK_GT(t.lanes(), 1);
  if (t.bits() == 8 && (t.is_int() || t.is_uint())) {
    if (!(t.lanes() == 2 || t.lanes() == 3)) {
      if (i != 0) {
        os << "|";
      }
      os << "((0x000000ff << " << i * 8 << ") & (" << value << " << " << i * 8 << "))";
      return;
    }
  }

  if (t.is_float16()) {
    if (i == 0) {
      PrintVecConstructor(t, os);
      os << '(';
    }
    if (i == t.lanes() - 1) {
      os << value << ")";
    } else {
      os << value << ",";
    }
    return;
  }

  if (t.is_bfloat16()) {
    if (i == 0) {
      PrintVecConstructor(t, os);
      os << '(';
    }
    if (i == t.lanes() - 1) {
      os << value << ")";
    } else {
      os << value << ",";
    }
    return;
  }

  if (i == 0) {
    PrintVecConstructor(t, os);
    os << "(";
  }
  os << value;
  if (i != t.lanes() - 1) {
    os << ",";
  } else {
    os << ")";
  }
  return;
}

}  // namespace codegen
}  // namespace tvm
