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

#include "tensorflow_serving/servables/caffe/caffe_session_bundle_factory.h"
#include "tensorflow_serving/servables/caffe/test_data/mnist_sample.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <algorithm>

#include "google/protobuf/wrappers.pb.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/types.h"

#include "tensorflow_serving/resources/resource_values.h"
#include "tensorflow_serving/servables/caffe/caffe_session_bundle_config.pb.h"
#include "tensorflow_serving/servables/caffe/caffe_session_bundle.h"
#include "tensorflow_serving/servables/caffe/caffe_py_util.h"
#include "tensorflow_serving/test_util/test_util.h"

namespace tensorflow {
namespace serving {
namespace {

using test_util::EqualsProto;

class CaffeSessionBundleFactoryTest : public ::testing::Test {
 protected:
  CaffeSessionBundleFactoryTest()
      : export_dir_(test_util::TestSrcDirPath(
            "servables/caffe/test_data/mnist_pretrained_caffe/00000023")) {
    input_sample_ = mnist_sample_28x28();
  }

  // Test data path, to be initialized to point at an export of half-plus-two.
  const string export_dir_;
  std::vector<float> input_sample_;

  // Test that a SessionBundle handles a single request for the half plus two
  // model properly. The request has size=2, for batching purposes.
  void TestSingleRequest(CaffeSessionBundle* bundle) {
    ASSERT_EQ(28 * 28, input_sample_.size());
    gtl::ArraySlice<float> input_slice(input_sample_);

    std::vector<std::pair<string, Tensor>> inputs{
        {"images", test::AsTensor(input_slice, {1, 1, 28, 28})}};
    std::vector<Tensor> outputs;

    TF_ASSERT_OK(bundle->session->Run(inputs, {"scores"}, {}, &outputs));
    {
      ASSERT_EQ(outputs.size(), 1);
      ASSERT_EQ(outputs[0].NumElements(), 10);
      // convert result to tensor
      {
        Tensor expected_output =
            test::AsTensor<float>({0, 0, 0, 0, 0, 0, 0, 1, 0, 0}, {1, 10});
        test::ExpectTensorNear<float>(expected_output, outputs[0], 0.05);
      }
    }
  }
};

inline string TestPyDirPath() {
  return test_util::TestSrcDirPath("servables/caffe/pycaffe");
}

class CaffeSessionBundleFactoryPyTest : public ::testing::Test {
 protected:
  CaffeSessionBundleFactoryPyTest()
      : export_dir_(test_util::TestSrcDirPath(
            "servables/caffe/test_data/py_layers/00000001")),
        input_sample_(0) {
    input_sample_.resize(9 * 8, 7.0);
  }

  const string export_dir_;
  std::vector<float> input_sample_;

  // Test that a SessionBundle handles a single request for the 10**3
  // model correctly.
  void TestSingleRequest(CaffeSessionBundle* bundle) {
    ASSERT_EQ(9 * 8, input_sample_.size());
    gtl::ArraySlice<float> input_slice(input_sample_);

    std::vector<std::pair<string, Tensor>> inputs{
        {"data", test::AsTensor(input_slice, {1, 9, 8})}};
    std::vector<Tensor> outputs;

    TF_ASSERT_OK(bundle->session->Run(inputs, {"three"}, {}, &outputs));
    ASSERT_EQ(outputs.size(), 1);

    auto vec = outputs[0].flat<float>();
    ASSERT_EQ(vec.size(), input_sample_.size());

    for (int i = 0; i < input_sample_.size(); ++i) {
      ASSERT_EQ(vec(i), input_sample_[i] * pow(10, 3));
    }
  }
};

TEST_F(CaffeSessionBundleFactoryTest, Basic) {
  const CaffeSessionBundleConfig config;
  std::unique_ptr<CaffeSessionBundleFactory> factory;
  TF_ASSERT_OK(CaffeSessionBundleFactory::Create(config, &factory));
  std::unique_ptr<CaffeSessionBundle> bundle;
  TF_ASSERT_OK(factory->CreateSessionBundle(export_dir_, &bundle));
  TestSingleRequest(bundle.get());
}

TEST_F(CaffeSessionBundleFactoryTest, Batching) {
  CaffeSessionBundleConfig config;
  BatchingParameters* batching_params = config.mutable_batching_parameters();
  batching_params->mutable_max_batch_size()->set_value(2);
  batching_params->mutable_max_enqueued_batches()->set_value(INT_MAX);
  std::unique_ptr<CaffeSessionBundleFactory> factory;
  TF_ASSERT_OK(CaffeSessionBundleFactory::Create(config, &factory));
  std::unique_ptr<CaffeSessionBundle> bundle;
  TF_ASSERT_OK(factory->CreateSessionBundle(export_dir_, &bundle));
  CaffeSessionBundle* bundle_raw = bundle.get();

  // Run multiple requests concurrently. They should be executed as
  // 'kNumRequestsToTest/2' batches.
  {
    const int kNumRequestsToTest = 10;
    std::vector<std::unique_ptr<Thread>> request_threads;
    for (int i = 0; i < kNumRequestsToTest; ++i) {
      request_threads.push_back(
          std::unique_ptr<Thread>(Env::Default()->StartThread(
              ThreadOptions(), strings::StrCat("thread_", i),
              [this, bundle_raw] { this->TestSingleRequest(bundle_raw); })));
    }
  }
}

TEST_F(CaffeSessionBundleFactoryTest,
       EstimateResourceRequirementWithBadExport) {
  const CaffeSessionBundleConfig config;
  std::unique_ptr<CaffeSessionBundleFactory> factory;

  TF_ASSERT_OK(CaffeSessionBundleFactory::Create(config, &factory));
  ResourceAllocation resource_requirement;

  const Status status = factory->EstimateResourceRequirement(
      "/a/bogus/export/dir", &resource_requirement);

  EXPECT_FALSE(status.ok());
}

TEST_F(CaffeSessionBundleFactoryTest,
       EstimateResourceRequirementWithGoodExport) {
  const uint64 kVariableFileSize = 1724991 /* bytes (approx. 1.6 MB) */;
  const uint64 expected_ram_requirement =
      kVariableFileSize *
          CaffeSessionBundleFactory::kResourceEstimateRAMMultiplier +
      CaffeSessionBundleFactory::kResourceEstimateRAMPadBytes;

  ResourceAllocation want;
  ResourceAllocation::Entry* ram_entry = want.add_resource_quantities();
  ram_entry->set_quantity(expected_ram_requirement);

  Resource* ram_resource = ram_entry->mutable_resource();
  ram_resource->set_device(device_types::kMain);
  ram_resource->set_kind(resource_kinds::kRamBytes);

  const CaffeSessionBundleConfig config;
  std::unique_ptr<CaffeSessionBundleFactory> factory;
  TF_ASSERT_OK(CaffeSessionBundleFactory::Create(config, &factory));

  ResourceAllocation got;
  TF_ASSERT_OK(factory->EstimateResourceRequirement(export_dir_, &got));
  EXPECT_THAT(got, EqualsProto(want));
}

TEST_F(CaffeSessionBundleFactoryPyTest, PythonLayer) {
#ifndef WITH_PYTHON_LAYER
  // do nothing if pycaffe is not enabled
  ASSERT_FALSE(IsPyCaffeAvailable());
  LOG(WARNING) << "[CaffeSessionBundleFactoryPyTest] skipped because pycaffe "
                  "unavailable\n"
               << "    (compile with --define=caffe_python_layer=ON)";
#else
  ASSERT_TRUE(IsPyCaffeAvailable());

  CaffeSessionBundleConfig config;
  config.set_enable_py_caffe(true);
  config.add_python_path(TestPyDirPath());
  config.add_python_path(export_dir_);

  std::unique_ptr<CaffeSessionBundleFactory> factory;
  TF_ASSERT_OK(CaffeSessionBundleFactory::Create(config, &factory));

  std::unique_ptr<CaffeSessionBundle> bundle;
  TF_ASSERT_OK(factory->CreateSessionBundle(export_dir_, &bundle));

  TestSingleRequest(bundle.get());
#endif
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
