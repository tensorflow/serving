/* Copyright 2016 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_SERVING_MODEL_SERVERS_PLATFORM_CAFFE_H_
#define TENSORFLOW_SERVING_MODEL_SERVERS_PLATFORM_CAFFE_H_

#include "tensorflow_serving/model_servers/model_platform_types.h"

#include "tensorflow_serving/servables/caffe/predict_impl.h"
#include "tensorflow_serving/servables/caffe/caffe_source_adapter.h"
#include "tensorflow_serving/servables/caffe/caffe_py_util.h"

template <>
struct ModelPlatformTraits<Caffe> {
  using SourceAdapter = tensorflow::serving::CaffeSourceAdapter;
  using SourceAdapterConfig = tensorflow::serving::CaffeSourceAdapterConfig;
  using PredictImpl = tensorflow::serving::CaffePredictImpl;

  static const char* name() { return "caffe"; }
  static constexpr bool defined = true;

  static void GlobalInit(int argc, char** argv) {
    tensorflow::serving::CaffeGlobalInit(&argc, &argv);
  }

  static tensorflow::Status ConfigureSourceAdapter(
      bool enable_batching, SourceAdapterConfig* config) {
    auto bundle_cfg = config->mutable_config();
    if (enable_batching) {
      tensorflow::serving::BatchingParameters* batching_parameters =
          bundle_cfg->mutable_batching_parameters();
      batching_parameters->mutable_thread_pool_name()->set_value(
          "model_server_batch_threads");
    }
    if (tensorflow::serving::IsPyCaffeAvailable()) {
      bundle_cfg->set_enable_py_caffe(true);
      bundle_cfg->add_python_path(
          ".runfiles/tf_serving/tensorflow_serving/servables/caffe/pycaffe");
    }

    return tensorflow::Status::OK();
  }

  template <typename TAdapter>
  static tensorflow::Status CreateSourceAdapter(
      const SourceAdapterConfig& config, std::unique_ptr<TAdapter>* adapter) {
    std::unique_ptr<SourceAdapter> typed_adapter;
    const auto status = SourceAdapter::Create(config, &typed_adapter);

    if (status.ok()) {
      *adapter = std::move(typed_adapter);
    }
    return status;
  }
};

#endif  // TENSORFLOW_SERVING_MODEL_SERVERS_PLATFORM_CAFFE_H_