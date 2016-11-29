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

#include "tensorflow_serving/servables/caffe/caffe_simple_servers.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow_serving/core/servable_handle.h"
#include "tensorflow_serving/servables/caffe/caffe_session_bundle.h"
#include "tensorflow_serving/test_util/test_util.h"

namespace tensorflow {
namespace serving {
namespace {

class SimpleServersTest : public ::testing::Test {
 protected:
  SimpleServersTest()
      : test_data_path_(test_util::TestSrcDirPath(
          "servables/caffe/test_data/mnist_pretrained_caffe")) {
  }

  const string test_data_path_;
};

TEST_F(SimpleServersTest, Basic) {
  std::unique_ptr<Manager> manager;
  const Status status = simple_servers::CreateSingleCaffeModelManagerFromBasePath(
      test_data_path_, &manager);
  TF_CHECK_OK(status);
  // We wait until the manager starts serving the servable.
  // TODO(b/25545570): Use the waiter api when it's ready.
  while (manager->ListAvailableServableIds().empty()) {
    Env::Default()->SleepForMicroseconds(1000);
  }
  ServableHandle<CaffeSessionBundle> bundle;
  const Status handle_status =
      manager->GetServableHandle(ServableRequest::Latest("default"), &bundle);
  
  TF_CHECK_OK(handle_status);
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
