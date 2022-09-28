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

#include <queue>

#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/threadpool_options.h"
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
const char kTestTfLiteModelPath[] =
    "servables/tensorflow/testdata/saved_model_half_plus_two_tflite/00000123";
const char kTestMLMDSavedModelPath[] =
    "servables/tensorflow/testdata/half_plus_two_mlmd/00000123";

}  // namespace

string GetTestSavedModelPath() {
  return test_util::TensorflowTestSrcDirPath(kTestSavedModelPath);
}

string GetTestMLMetadataSavedModelPath() {
  return test_util::TensorflowTestSrcDirPath(kTestMLMDSavedModelPath);
}

string GetTestSessionBundleExportPath() {
  return test_util::TestSrcDirPath(kTestSessionBundleExportPath);
}

string GetTestTfLiteModelPath() {
  return test_util::TestSrcDirPath(kTestTfLiteModelPath);
}

std::vector<string> GetTestSessionBundleExportFiles() {
  const string dir = GetTestSessionBundleExportPath();
  return {tensorflow::io::JoinPath(dir, "export.meta"),
          tensorflow::io::JoinPath(dir, "export-00000-of-00001")};
}

std::vector<string> GetTestSavedModelBundleExportFiles() {
  const string dir = GetTestSavedModelPath();
  return {
      tensorflow::io::JoinPath(dir, "saved_model.pb"),
      tensorflow::io::JoinPath(dir, "assets/foo.txt"),
      tensorflow::io::JoinPath(dir, "variables/variables.index"),
      tensorflow::io::JoinPath(dir, "variables/variables.data-00000-of-00001")};
}

uint64_t GetTotalFileSize(const std::vector<string>& files) {
  uint64_t total_file_size = 0;
  for (const string& file : files) {
    if (!(Env::Default()->IsDirectory(file).ok()) &&
        Env::Default()->FileExists(file).ok()) {
      uint64_t file_size;
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

void TestSingleRequest(Session* session, int input_batch_size) {
  Tensor input(DT_FLOAT, TensorShape({input_batch_size}));
  test::FillIota<float>(&input, 100.0f);
  // half plus two: output should be input / 2 + 2.
  Tensor expected_output(DT_FLOAT, TensorShape({input_batch_size}));
  test::FillFn<float>(&expected_output,
                      [](int i) -> float { return (100.0f + i) / 2 + 2; });

  // Note that "x" and "y" are the actual names of the nodes in the graph.
  // The saved manifest binds these to "input" and "output" respectively, but
  // these tests are focused on the raw underlying session without bindings.
  const std::vector<std::pair<string, Tensor>> inputs = {{"x:0", input}};
  const std::vector<string> output_names = {"y:0"};
  const std::vector<string> empty_targets;
  std::vector<Tensor> outputs;

  RunMetadata run_metadata;
  TF_ASSERT_OK(session->Run(RunOptions{}, inputs, output_names, empty_targets,
                            &outputs, &run_metadata,
                            thread::ThreadPoolOptions{}));

  ASSERT_EQ(1, outputs.size());
  const auto& single_output = outputs.at(0);
  test::ExpectTensorEqual<float>(expected_output, single_output);
}

void TestMultipleRequests(Session* session, int num_requests,
                          int input_batch_size) {
  std::vector<std::unique_ptr<Thread>> request_threads;
  request_threads.reserve(num_requests);
  for (int i = 0; i < num_requests; ++i) {
    request_threads.push_back(
        std::unique_ptr<Thread>(Env::Default()->StartThread(
            ThreadOptions(), strings::StrCat("thread_", i),
            [session, input_batch_size] {
              TestSingleRequest(session, input_batch_size);
            })));
  }
}

ResourceAllocation GetExpectedResourceEstimate(double total_file_size) {
  // kResourceEstimateRAMMultiplier and kResourceEstimateRAMPadBytes should
  // match the constants defined in bundle_factory_util.cc.
  const double kResourceEstimateRAMMultiplier = 1.2;
  const int kResourceEstimateRAMPadBytes = 0;
  const uint64_t expected_ram_requirement =
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

void CopyDirOrDie(const string& src_dir, const string& dst_dir) {
  int64_t u_files = 0;
  int64_t u_dirs = 0;
  if (Env::Default()->IsDirectory(dst_dir).ok()) {
    TF_ASSERT_OK(Env::Default()->DeleteRecursively(dst_dir, &u_files, &u_dirs));
  }
  TF_ASSERT_OK(Env::Default()->RecursivelyCreateDir(dst_dir));
  std::queue<std::string> dirs_to_copy;
  dirs_to_copy.push(src_dir);
  while (!dirs_to_copy.empty()) {
    const string dir = dirs_to_copy.front();
    dirs_to_copy.pop();
    std::vector<std::string> children;
    TF_ASSERT_OK(Env::Default()->GetChildren(dir, &children));
    for (const string& child : children) {
      const string child_path = io::JoinPath(dir, child);
      StringPiece remainder = child_path;
      CHECK(str_util::ConsumePrefix(&remainder, src_dir));
      if (Env::Default()->IsDirectory(child_path).ok()) {
        TF_ASSERT_OK(
            Env::Default()->CreateDir(io::JoinPath(dst_dir, remainder)));
        dirs_to_copy.push(child_path);
      } else {
        TF_ASSERT_OK(Env::Default()->CopyFile(
            child_path, io::JoinPath(dst_dir, remainder)));
      }
    }
  }
}

}  // namespace test_util
}  // namespace serving
}  // namespace tensorflow
