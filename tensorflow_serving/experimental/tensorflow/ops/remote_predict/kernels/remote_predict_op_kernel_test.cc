/* Copyright 2020 Google Inc. All Rights Reserved.

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
#include "tensorflow_serving/experimental/tensorflow/ops/remote_predict/kernels/remote_predict_op_kernel.h"

#include "absl/status/status.h"
#include "absl/time/time.h"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#include "tensorflow_serving/experimental/tensorflow/ops/remote_predict/cc/ops/remote_predict_op.h"

namespace tensorflow {
namespace serving {
namespace {

// Empty mock rpc class.
class MockRpc {};

// Mock class for RemotePredict Op kernel test.
class MockPredictionService {
 public:
  static absl::Status Create(const string& target_address,
                             std::unique_ptr<MockPredictionService>* service) {
    service->reset(new MockPredictionService(target_address));
    return ::absl::OkStatus();
  }

  StatusOr<MockRpc*> CreateRpc(absl::Duration max_rpc_deadline) {
    return new MockRpc;
  }

  // The model_name in request determines response and/or status.
  void Predict(MockRpc* rpc, PredictRequest* request, PredictResponse* response,
               std::function<void(absl::Status status)> callback);

  static constexpr char kGoodModel[] = "good_model";
  static constexpr char kBadModel[] = "bad_model";

 private:
  MockPredictionService(const string& target_address);
};

constexpr char MockPredictionService::kGoodModel[];
constexpr char MockPredictionService::kBadModel[];

typedef google::protobuf::Map<tensorflow::string, tensorflow::TensorProto> AliasTensorMap;

MockPredictionService::MockPredictionService(const string& target_address) {}

void MockPredictionService::Predict(
    MockRpc* rpc, PredictRequest* request, PredictResponse* response,
    std::function<void(absl::Status status)> callback) {
  // Use model name to specify the behavior of each test.
  std::string model_name = request->model_spec().name();
  if (model_name == kGoodModel) {
    *(response->mutable_model_spec()) = request->model_spec();
    AliasTensorMap& inputs = *request->mutable_inputs();
    AliasTensorMap& outputs = *response->mutable_outputs();
    outputs["output0"] = inputs["input0"];
    outputs["output1"] = inputs["input1"];
    callback(::absl::OkStatus());
  }

  if (model_name == kBadModel) {
    callback(absl::Status(absl::StatusCode::kAborted, "Aborted"));
  }
}

REGISTER_KERNEL_BUILDER(Name("TfServingRemotePredict").Device(DEVICE_CPU),
                        RemotePredictOp<MockPredictionService>);

using RemotePredict = ops::TfServingRemotePredict;

// Use model_name to specify the behavior of different tests.
::tensorflow::Status RunRemotePredict(
    const string& model_name, std::vector<Tensor>* outputs,
    const DataTypeSlice& output_types = {DT_INT32, DT_INT32},
    const absl::optional<::absl::Duration> deadline = absl::nullopt,
    bool fail_on_rpc_error = true,
    const string& target_address = "target_address",
    int64_t target_model_version = -1, const string& signature_name = "") {
  const Scope scope = Scope::DisabledShapeInferenceScope();
  // Model_name will decide the result of the RPC.
  auto input_tensor_aliases = ops::Const(
      scope.WithOpName("input_tensor_aliases"), {"input0", "input1"});
  auto input_tensors0 = ops::Const(scope.WithOpName("input_tensors0"), {1, 2});
  auto input_tensors1 = ops::Const(scope.WithOpName("input_tensors1"), {3, 4});
  auto output_tensor_aliases = ops::Const(
      scope.WithOpName("output_tensor_aliases"), {"output0", "output1"});
  std::vector<Output> fetch_outputs;
  RemotePredict::Attrs attrs = RemotePredict::Attrs()
                                   .TargetAddress(target_address)
                                   .ModelName(model_name)
                                   .SignatureName(signature_name);

  if (target_model_version >= 0) {
    attrs = attrs.ModelVersion(target_model_version);
  }
  if (deadline.has_value()) {
    attrs = attrs.MaxRpcDeadlineMillis(absl::ToInt64Seconds(deadline.value()) *
                                       1000);
  }
  attrs = attrs.FailOpOnRpcError(fail_on_rpc_error);

  auto remote_predict = RemotePredict(
      scope, input_tensor_aliases, {input_tensors0, input_tensors1},
      output_tensor_aliases, output_types, attrs);

  fetch_outputs = {remote_predict.status_code,
                   remote_predict.status_error_message};
  fetch_outputs.insert(fetch_outputs.end(),
                       remote_predict.output_tensors.begin(),
                       remote_predict.output_tensors.end());
  TF_RETURN_IF_ERROR(scope.status());

  ClientSession session(scope);
  return session.Run(fetch_outputs, outputs);
}

TEST(RemotePredictTest, TestSimple) {
  std::vector<Tensor> outputs;
  TF_ASSERT_OK(RunRemotePredict(
      /*model_name=*/MockPredictionService::kGoodModel, &outputs));
  ASSERT_EQ(4, outputs.size());
  // Checks whether the status code is 0 and there is no error message.
  EXPECT_EQ(0, outputs[0].scalar<int>()());
  EXPECT_EQ("", outputs[1].scalar<tensorflow::tstring>()());
  test::ExpectTensorEqual<int>(outputs[2], test::AsTensor<int>({1, 2}));
  test::ExpectTensorEqual<int>(outputs[3], test::AsTensor<int>({3, 4}));
}

TEST(RemotePredictTest, TestRpcError) {
  std::vector<Tensor> outputs;
  const auto status = RunRemotePredict(
      /*model_name=*/MockPredictionService::kBadModel, &outputs);
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(error::Code::ABORTED, status.code());
  EXPECT_THAT(status.error_message(), ::testing::HasSubstr("Aborted"));
}

TEST(RemotePredictTest, TestRpcErrorReturnStatus) {
  std::vector<Tensor> outputs;
  // Specifying output_types to float solves
  // "MemorySanitizer: use-of-uninitialized-value"
  const auto status = RunRemotePredict(
      /*model_name=*/MockPredictionService::kBadModel, &outputs,
      {DT_FLOAT, DT_FLOAT}, /*deadline=*/absl::nullopt,
      /*fail_on_rpc_error=*/false);

  EXPECT_TRUE(status.ok());
  EXPECT_EQ(static_cast<int>(error::Code::ABORTED), outputs[0].scalar<int>()());
  EXPECT_EQ("Aborted", outputs[1].scalar<tensorflow::tstring>()());
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
