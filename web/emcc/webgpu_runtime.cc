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

/*
 * \file webgpu_runtime.cc
 * \brief WebGPU runtime based on the TVM JS.
 */

// configurations for tvm logging
#define TVM_LOG_STACK_TRACE 0
#define TVM_LOG_DEBUG 0
#define TVM_LOG_CUSTOMIZE 1
#define TVM_FFI_ALWAYS_LOG_BEFORE_THROW 1
#define DMLC_USE_LOGGING_LIBRARY <tvm/runtime/logging.h>

#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/device_api.h>

#include <iostream>
#include <string>

#include "../../src/runtime/meta_data.h"
#include "../../src/runtime/workspace_pool.h"

namespace tvm {
namespace runtime {

/*! \brief Thread local workspace */
class WebGPUThreadEntry {
 public:
  /*! \brief thread local pool*/
  WorkspacePool pool;
  /*! \brief constructor */
  WebGPUThreadEntry();
  // get the threadlocal workspace
  static WebGPUThreadEntry* ThreadLocal();
};

// All the implementations are redirectly to the JS side.
class WebGPUDeviceAPI : public DeviceAPI {
 public:
  WebGPUDeviceAPI() {
    auto fp = tvm::ffi::Function::GetGlobal("wasm.WebGPUDeviceAPI");
    CHECK(fp.has_value()) << "Cannot find wasm.WebGPUContext in the env";
    auto getter = ffi::TypedFunction<ffi::Function(std::string)>(*fp);
    alloc_space_ = getter("deviceAllocDataSpace");
    free_space_ = getter("deviceFreeDataSpace");
    copy_to_gpu_ = getter("deviceCopyToGPU");
    copy_from_gpu_ = getter("deviceCopyFromGPU");
    copy_within_gpu_ = getter("deviceCopyWithinGPU");
  }

  void SetDevice(Device dev) final {}
  void GetAttr(Device dev, DeviceAttrKind kind, ffi::Any* rv) final {
    if (kind == kExist) {
      *rv = 1;
    }
  }

  void* AllocDataSpace(Device dev, size_t nbytes, size_t alignment, DLDataType type_hint) final {
    double ptr_number = alloc_space_(nbytes);
    return reinterpret_cast<void*>(static_cast<int64_t>(ptr_number));
  }

  void FreeDataSpace(Device dev, void* ptr) final { return free_space_(ptr); }

 protected:
  void CopyDataFromTo(const void* from, size_t from_offset, void* to, size_t to_offset, size_t size,
                      Device dev_from, Device dev_to, DLDataType type_hint,
                      TVMStreamHandle stream) final {
    if (static_cast<int>(dev_from.device_type) == kDLWebGPU &&
        static_cast<int>(dev_to.device_type) == kDLWebGPU) {
      CHECK_EQ(dev_from.device_id, dev_to.device_id);
      copy_within_gpu_(const_cast<void*>(from), from_offset, to, to_offset, size);
    } else if (static_cast<int>(dev_from.device_type) == kDLWebGPU &&
               dev_to.device_type == kDLCPU) {
      void* to_ptr = static_cast<uint8_t*>(to) + to_offset;
      copy_from_gpu_(const_cast<void*>(from), from_offset, to_ptr, size);
    } else if (dev_from.device_type == kDLCPU &&
               static_cast<int>(dev_to.device_type) == kDLWebGPU) {
      void* from_ptr = static_cast<uint8_t*>(const_cast<void*>(from)) + from_offset;
      copy_to_gpu_(from_ptr, to, to_offset, size);
    } else {
      LOG(FATAL) << "expect copy from/to WebGPU or between WebGPU";
    }
  }

 public:
  TVMStreamHandle CreateStream(Device dev) final { LOG(FATAL) << "Not implemented"; }

  void FreeStream(Device dev, TVMStreamHandle stream) final { LOG(FATAL) << "Not implemented"; }

  void SyncStreamFromTo(Device dev, TVMStreamHandle event_src, TVMStreamHandle event_dst) {
    LOG(FATAL) << "Not implemented";
  }

  void StreamSync(Device dev, TVMStreamHandle stream) final {
    static auto func = tvm::ffi::Function::GetGlobal("__asyncify.WebGPUWaitForTasks");
    ICHECK(func.has_value()) << "Stream sync inside c++ only supported in asyncify mode";
    (*func)();
  }

  void SetStream(Device dev, TVMStreamHandle stream) final { LOG(FATAL) << "Not implemented"; }

  TVMStreamHandle GetCurrentStream(Device dev) final { LOG(FATAL) << "Not implemented"; }

  void* AllocWorkspace(Device dev, size_t size, DLDataType type_hint) final {
    return WebGPUThreadEntry::ThreadLocal()->pool.AllocWorkspace(dev, size);
  }

  void FreeWorkspace(Device dev, void* data) final {
    WebGPUThreadEntry::ThreadLocal()->pool.FreeWorkspace(dev, data);
  }

  static WebGPUDeviceAPI* Global() {
    static WebGPUDeviceAPI* inst = new WebGPUDeviceAPI();
    return inst;
  }

 private:
  // NOTE: js return number as double.
  ffi::TypedFunction<double(int64_t nbytes)> alloc_space_;
  ffi::TypedFunction<void(void* ptr)> free_space_;
  ffi::TypedFunction<void(void* from, void* to, int64_t to_offset, int64_t nbytes)> copy_to_gpu_;
  ffi::TypedFunction<void(void* from, int64_t from_offset, void* to, int64_t nbytes)>
      copy_from_gpu_;
  ffi::TypedFunction<void(void* from, int64_t from_offset, void* to, int64_t to_offset,
                          int64_t nbytes)>
      copy_within_gpu_;
};

typedef dmlc::ThreadLocalStore<WebGPUThreadEntry> WebGPUThreadStore;

WebGPUThreadEntry::WebGPUThreadEntry()
    : pool(static_cast<DLDeviceType>(kDLWebGPU), WebGPUDeviceAPI::Global()) {}

WebGPUThreadEntry* WebGPUThreadEntry::ThreadLocal() {
  static thread_local WebGPUThreadEntry inst = WebGPUThreadEntry();
  return &inst;
}

class WebGPUModuleNode final : public runtime::ModuleNode {
 public:
  explicit WebGPUModuleNode(std::unordered_map<std::string, std::string> smap,
                            std::unordered_map<std::string, FunctionInfo> fmap)
      : smap_(smap), fmap_(fmap) {
    auto fp = tvm::ffi::Function::GetGlobal("wasm.WebGPUCreateShader");
    CHECK(fp.has_value());
    create_shader_ = *fp;
  }

  const char* type_key() const final { return "webgpu"; }

  ffi::Function GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) final {
    // special function
    if (name == "webgpu.get_fmap") {
      return ffi::Function([this](ffi::PackedArgs args, ffi::Any* rv) {
        std::ostringstream os;
        dmlc::JSONWriter writer(&os);
        writer.Write(fmap_);
        *rv = os.str();
      });
    } else if (name == "webgpu.get_shader") {
      return ffi::Function([this](ffi::PackedArgs args, ffi::Any* rv) {
        auto name = args[0].cast<std::string>();
        auto it = smap_.find(name);
        ICHECK(it != smap_.end()) << "Cannot find code " << name;
        *rv = it->second;
      });
    } else if (name == "webgpu.update_prebuild") {
      return ffi::Function([this](ffi::PackedArgs args, ffi::Any* rv) {
        auto name = args[0].cast<std::string>();
        ffi::Function func = args[1].cast<ffi::Function>();
        prebuild_[name] = func;
      });
    }
    // check prebuild cache
    auto prebuild_it = prebuild_.find(name);
    if (prebuild_it != prebuild_.end()) {
      return prebuild_it->second;
    }

    auto it = smap_.find(name);
    if (it != smap_.end()) {
      FunctionInfo info = fmap_.at(name);
      info.name = name;
      std::ostringstream os;
      dmlc::JSONWriter writer(&os);
      info.Save(&writer);
      return create_shader_(os.str(), it->second);
    } else {
      return ffi::Function(nullptr);
    }
  }

  int GetPropertyMask() const final { return ModulePropertyMask::kBinarySerializable; };

  void SaveToBinary(dmlc::Stream* stream) final { LOG(FATAL) << "Not implemented"; }

  String GetSource(const String& format) final {
    // can only return source code.
    return source_;
  }

 private:
  // code table
  std::unordered_map<std::string, std::string> smap_;
  // function information table.
  std::unordered_map<std::string, FunctionInfo> fmap_;
  // The source
  std::string source_;
  // prebuild_ functions
  std::unordered_map<std::string, ffi::Function> prebuild_;
  // Callback to get the GPU function.
  ffi::TypedFunction<ffi::Function(std::string finfo, std::string shader)> create_shader_;
};

Module WebGPUModuleLoadBinary(void* strm) {
  dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);
  std::unordered_map<std::string, std::string> smap;
  std::unordered_map<std::string, FunctionInfo> fmap;

  stream->Read(&fmap);
  stream->Read(&smap);
  return Module(make_object<WebGPUModuleNode>(smap, fmap));
}

// for now webgpu is hosted via a vulkan module.
TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("runtime.module.loadbinary_webgpu", WebGPUModuleLoadBinary)
      .def_packed("device_api.webgpu", [](ffi::PackedArgs args, ffi::Any* rv) {
        DeviceAPI* ptr = WebGPUDeviceAPI::Global();
        *rv = static_cast<void*>(ptr);
      });
});

}  // namespace runtime
}  // namespace tvm
