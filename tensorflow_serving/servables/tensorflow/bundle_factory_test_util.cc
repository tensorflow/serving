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

#include "tensorflow_serving/servables/tensorflow/bundle_factory_test_util.h"

#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow_serving/resources/resource_values.h"
#include "tensorflow_serving/test_util/test_util.h"

namespace tensorflow {
namespace serving {
namespace test_util {

namespace {

const char kTestSavedModelPath[] =
    "cc/saved_model/testdata/half_plus_two/00000123";
const char kTestSessionBundleExportPath[] =
    "session_bundle/testdata/half_plus_two/00000123";

}  // namespace

string GetTestSavedModelPath() {
  return test_util::TensorflowTestSrcDirPath(kTestSavedModelPath);
}

string GetTestSessionBundleExportPath() {
  return test_util::ContribTestSrcDirPath(kTestSessionBundleExportPath);
}

std::vector<string> GetTestSavedModelFiles() {
  const string dir = GetTestSavedModelPath();
  return {tensorflow::io::JoinPath(dir, kSavedModelAssetsDirectory, "foo.txt"),
          tensorflow::io::JoinPath(dir, kSavedModelFilenamePb),
          tensorflow::io::JoinPath(dir, kSavedModelVariablesFilename,
                                   "variables.data-00000-of-00001"),
          tensorflow::io::JoinPath(dir, kSavedModelVariablesFilename,
                                   "variables.index")};
}

std::vector<string> GetTestSessionBundleExportFiles() {
  const string dir = GetTestSessionBundleExportPath();
  return {tensorflow::io::JoinPath(dir, "export.meta"),
          tensorflow::io::JoinPath(dir, "export-00000-of-00001")};
}

uint64 GetTotalFileSize(const std::vector<string>& files) {
  uint64 total_file_size = 0;
  for (const string& file : files) {
    if (!(Env::Default()->IsDirectory(file).ok())) {
      uint64 file_size;
      TF_CHECK_OK(Env::Default()->GetFileSize(file, &file_size));
      total_file_size += file_size;
    }
  }
  return total_file_size;
}

SignatureDef GetTestSessionSignature() {
  SignatureDef signature;
  TensorInfo input;
  input.set_name("x:0");
  (*signature.mutable_inputs())["x"] = input;
  TensorInfo output;
  output.set_name("y:0");
  (*signature.mutable_outputs())["y"] = output;
  return signature;
}

void TestSingleRequest(Session* session) {
  Tensor input = test::AsTensor<float>({100.0f, 42.0f}, {2});
  // half plus two: output should be input / 2 + 2.
  Tensor expected_output =
      test::AsTensor<float>({100.0f / 2 + 2, 42.0f / 2 + 2}, {2});

  // Note that "x" and "y" are the actual names of the nodes in the graph.
  // The saved manifest binds these to "input" and "output" respectively, but
  // these tests are focused on the raw underlying session without bindings.
  const std::vector<std::pair<string, Tensor>> inputs = {{"x:0", input}};
  const std::vector<string> output_names = {"y:0"};
  const std::vector<string> empty_targets;
  std::vector<Tensor> outputs;

  TF_ASSERT_OK(session->Run(inputs, output_names, empty_targets, &outputs));

  ASSERT_EQ(1, outputs.size());
  const auto& single_output = outputs.at(0);
  test::ExpectTensorEqual<float>(expected_output, single_output);
}

void TestMultipleRequests(int num_requests, Session* session) {
  std::vector<std::unique_ptr<Thread>> request_threads;
  for (int i = 0; i < num_requests; ++i) {
    request_threads.push_back(
        std::unique_ptr<Thread>(Env::Default()->StartThread(
            ThreadOptions(), strings::StrCat("thread_", i),
            [session] { TestSingleRequest(session); })));
  }
}

ResourceAllocation GetExpectedResourceEstimate(double total_file_size) {
  // kResourceEstimateRAMMultiplier and kResourceEstimateRAMPadBytes should
  // match the constants defined in bundle_factory_util.cc.
  const double kResourceEstimateRAMMultiplier = 1.2;
  const int kResourceEstimateRAMPadBytes = 0;
  const uint64 expected_ram_requirement =
      total_file_size * kResourceEstimateRAMMultiplier +
      kResourceEstimateRAMPadBytes;
  ResourceAllocation resource_alloc;
  ResourceAllocation::Entry* ram_entry =
      resource_alloc.add_resource_quantities();
  Resource* ram_resource = ram_entry->mutable_resource();
  ram_resource->set_device(device_types::kMain);
  ram_resource->set_kind(resource_kinds::kRamBytes);
  ram_entry->set_quantity(expected_ram_requirement);
  return resource_alloc;
}

}  // namespace test_util
}  // namespace serving
}  // namespace tensorflow
