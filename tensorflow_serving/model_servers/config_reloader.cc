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

#include "tensorflow_serving/model_servers/config_reloader.h"

#include <string>
#include <utility>
#include <thread>
#include <sys/stat.h>

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

bool ConfigReloader::Start() {
  exit_ = false;
  stop_loading_ = false;
  loading_thread_.reset(new std::thread(std::bind(&ConfigReloader::Run, this)));
  return true;
}

bool ConfigReloader::Stop() {
  exit_ = true;
  stop_loading_ = true;
  return true;
}

void ConfigReloader::Run() {
  while (!exit_) {
    Reload();
    // sleep
    std::this_thread::sleep_for(std::chrono::seconds(kInterval));
  }
}

void ConfigReloader::Reload() {
  if (!Env::Default()->FileExists(file_path_).ok()) {
    LOG(INFO) << "Config file not exist: " << file_path_;
    return;
  }
  struct stat file_stat;
  if (-1 == stat(file_path_.c_str(), &file_stat)) {
    LOG(INFO) << "Failed to stat: " << file_path_;
    return;
  }
  time_t modifiled_time = file_stat.st_mtime;
  if (modifiled_time <= last_modified_time_) {
    return;
  }
  last_modified_time_ = modifiled_time;
  if (reload_func_) {
    reload_func_();
  }
}

}  // namespace serving
}  // namespace tensorflow
