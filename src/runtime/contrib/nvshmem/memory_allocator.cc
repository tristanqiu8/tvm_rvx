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
#include <nvshmem.h>
#include <nvshmemx.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/memory/memory_manager.h>

#include <thread>

#include "../../cuda/cuda_common.h"
#include "../../disco/utils.h"
#include "../../memory/pooled_allocator.h"

namespace tvm {
namespace runtime {

using tvm::runtime::memory::Buffer;
using tvm::runtime::memory::PooledAllocator;

/*!
 * \brief The memory allocator of NVSHMEM.
 * Overriding PooledAllocator for efficient memory management.
 */
class NVSHMEMAllocator final : public PooledAllocator {
 public:
  explicit NVSHMEMAllocator() : PooledAllocator() {}

  ~NVSHMEMAllocator() { PooledAllocator::ReleaseAll(); }

  void Clear() final { PooledAllocator::ReleaseAll(); }

  bool AllowMemoryScope(const std::string& mem_scope) const final {
    // The allowed memory scope of NVSHMEM is "nvshmem";
    return mem_scope == "nvshmem";
  }

  /*! \brief Return the global NVSHMEM singleton allocator. */
  static NVSHMEMAllocator* Global() {
    static NVSHMEMAllocator* allocator = new NVSHMEMAllocator();
    return allocator;
  }

  NDArray Empty(ffi::Shape shape, DataType dtype, Device device) {
    class NVSHMEMAlloc {
     public:
      explicit NVSHMEMAlloc(Buffer buffer) : buffer_(buffer) {}
      void AllocData(DLTensor* tensor) { tensor->data = buffer_.data; }
      void FreeData(DLTensor* tensor) { NVSHMEMAllocator::Global()->Free(buffer_); }

     private:
      Buffer buffer_;
    };

    Buffer buffer = PooledAllocator::Alloc(device, shape, dtype, String("nvshmem"));
    return NDArray::FromNDAlloc(NVSHMEMAlloc(buffer), shape, dtype, device);
  }

 private:
  void* DeviceAllocDataSpace(Device dev, size_t size, size_t alignment,
                             DLDataType type_hint) final {
    ICHECK_EQ(dev.device_type, DLDeviceType::kDLCUDA)
        << "nvshmem can only allocate cuda device memory space.";
    ICHECK(type_hint.code == DLDataTypeCode::kDLInt || type_hint.code == DLDataTypeCode::kDLUInt ||
           type_hint.code == DLDataTypeCode::kDLFloat)
        << "nvshmem can only allocate tensor with int, usingned int or float data types.";
    return nvshmem_align(alignment, size);
  }

  void DeviceFreeDataSpace(Device dev, void* ptr) final { nvshmem_free(ptr); }
};

NDArray NVSHMEMEmpty(ffi::Shape shape, DataType dtype, Device device) {
  return NVSHMEMAllocator::Global()->Empty(shape, dtype, UseDefaultDeviceIfNone(device));
}

TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("runtime.disco.nvshmem.empty", NVSHMEMEmpty);
});

void NVSHMEMFinalize() {
  NVSHMEMAllocator::Global()->Clear();
  nvshmem_finalize();
}

TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("runtime.disco.nvshmem.finalize_nvshmem", NVSHMEMFinalize);
});

}  // namespace runtime
}  // namespace tvm
