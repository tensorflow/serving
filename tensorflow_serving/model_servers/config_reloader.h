/* Copyright 2018 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_SERVING_MODEL_SERVERS_CONFIG_LOADER_H_
#define TENSORFLOW_SERVING_MODEL_SERVERS_CONFIG_LOADER_H_

#include <string>
#include <utility>
#include <thread>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_serving/config/model_server_config.pb.h"
#include "tensorflow_serving/core/source.h"
#include "tensorflow_serving/core/storage_path.h"
#include "tensorflow_serving/sources/storage_path/file_system_storage_path_source.h"

namespace tensorflow {
namespace serving {

class ConfigReloader {
 public:
  ConfigReloader(const std::string& file_path, bool stop_loading = false)
      : file_path_(file_path), stop_loading_(stop_loading) {}
  virtual ~ConfigReloader() {
    if (loading_thread_) {
      loading_thread_->join();
    }
  }
  ConfigReloader(const ConfigReloader&) = delete;
  ConfigReloader& operator=(const ConfigReloader&) = delete;

  virtual bool Start();
  virtual bool Stop();
  const string& GetFilePath() { return file_path_; }
  void SetReloadFunc(std::function<void()> reload_func) {
    reload_func_ = reload_func;
  }

 protected:
  const int32_t kInterval{ 1 };
  volatile bool exit_;
  volatile bool stop_loading_;
  std::unique_ptr<std::thread> loading_thread_;
  std::string file_path_;
  time_t last_modified_time_{ 0 };
  std::function<void()> reload_func_;

  virtual void Run();
  void Reload();
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_MODEL_SERVERS_CONFIG_LOADER_H_
