/* Copyright 2026 Google Inc. All Rights Reserved.

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

#include "tensorflow_serving/servables/tensorflow/tfrt_servable.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/cc/saved_model/reader.h"
#include "tensorflow/cc/saved_model/signature_constants.h"
#include "tensorflow/cc/saved_model/tag_constants.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/status.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/tfrt/runtime/runtime.h"
#include "tensorflow/core/tfrt/saved_model/saved_model.h"
#include "tensorflow_serving/apis/model.pb.h"
#include "tensorflow_serving/apis/predict.pb.h"
#include "tensorflow_serving/config/model_server_config.pb.h"
#include "tensorflow_serving/config/platform_config.pb.h"
#include "tensorflow_serving/core/aspired_version_policy.h"
#include "tensorflow_serving/core/availability_preserving_policy.h"
#include "tensorflow_serving/core/servable_handle.h"
#include "tensorflow_serving/model_servers/model_platform_types.h"
#include "tensorflow_serving/model_servers/server_core.h"
#include "tensorflow_serving/servables/tensorflow/saved_model_config.pb.h"
#include "tensorflow_serving/servables/tensorflow/servable.h"
#include "tensorflow_serving/servables/tensorflow/test_util/mock_tfrt_saved_model.h"
#include "tensorflow_serving/servables/tensorflow/tfrt_saved_model_source_adapter.pb.h"
#include "tensorflow_serving/test_util/test_util.h"

namespace tensorflow {
namespace serving {
namespace {

using ::testing::_;
using ::testing::HasSubstr;
using ::testing::Return;
using ::testing::ReturnRef;

constexpr char kTestModelName[] = "test_model";
constexpr int kTestModelVersion = 123;

class TfrtSavedModelServableTest : public ::testing::Test {
 public:
  static void SetUpTestSuite() {
    tfrt_stub::SetGlobalRuntime(
        tfrt_stub::Runtime::Create(/*num_inter_op_threads=*/4));

    ModelServerConfig config;
    auto model_config = config.mutable_model_config_list()->add_config();
    model_config->set_name(kTestModelName);
    model_config->set_base_path(
        test_util::TestSrcDirPath("servables/tensorflow/testdata/"
                                  "saved_model_half_plus_two_tf2_cpu"));
    model_config->set_model_platform(kTensorFlowModelPlatform);

    ServerCore::Options options;
    options.model_server_config = config;
    PlatformConfigMap platform_config_map;
    ::google::protobuf::Any source_adapter_config;
    TfrtSavedModelSourceAdapterConfig saved_model_bundle_source_adapter_config;
    source_adapter_config.PackFrom(saved_model_bundle_source_adapter_config);
    (*(*platform_config_map
            .mutable_platform_configs())[kTensorFlowModelPlatform]
          .mutable_source_adapter_config()) = source_adapter_config;
    options.platform_config_map = platform_config_map;
    options.aspired_version_policy =
        std::unique_ptr<AspiredVersionPolicy>(new AvailabilityPreservingPolicy);
    // Reduce the number of initial load threads to be num_load_threads to avoid
    // timing out in tests.
    options.num_initial_load_threads = options.num_load_threads;
    TF_ASSERT_OK(ServerCore::Create(std::move(options), &server_core_));
  }

  static void TearDownTestSuite() { server_core_.reset(); }

 protected:
  ServableHandle<Servable> GetServable() {
    ModelSpec model_spec;
    model_spec.set_name(kTestModelName);
    ServableHandle<Servable> servable;
    TF_CHECK_OK(server_core_->GetServableHandle(model_spec, &servable));
    return servable;
  }

  static PredictRequest CreateRequest(float x) {
    PredictRequest request;
    request.mutable_model_spec()->set_name(kTestModelName);
    request.mutable_model_spec()->mutable_version()->set_value(
        kTestModelVersion);
    TensorProto& input = (*request.mutable_inputs())["x"];
    input.set_dtype(tensorflow::DT_FLOAT);
    input.add_float_val(x);
    return request;
  }

 private:
  static std::unique_ptr<ServerCore> server_core_;
};

std::unique_ptr<ServerCore> TfrtSavedModelServableTest::server_core_;

// A model without streaming ops must return its regular outputs through
// PredictStreamed as a single final response.
TEST_F(TfrtSavedModelServableTest,
       PredictStreamedReturnsRegularOutputsForNonStreamingModel) {
  ServableHandle<Servable> servable = GetServable();

  std::vector<absl::StatusOr<PredictResponse>> responses;
  absl::StatusOr<std::unique_ptr<PredictStreamedContext>> context =
      servable->PredictStreamed(
          Servable::RunOptions(),
          [&responses](absl::StatusOr<PredictResponse> response) {
            responses.push_back(std::move(response));
          });
  TF_ASSERT_OK(context.status());

  PredictRequest request = CreateRequest(2.0f);
  TF_ASSERT_OK((*context)->ProcessRequest(&request));
  TF_ASSERT_OK((*context)->Close());

  ASSERT_EQ(responses.size(), 1);
  TF_ASSERT_OK(responses[0].status());
  const PredictResponse& response = *responses[0];
  EXPECT_EQ(response.model_spec().name(), kTestModelName);
  EXPECT_EQ(response.model_spec().signature_name(),
            kDefaultServingSignatureDefKey);
  EXPECT_EQ(response.model_spec().version().value(), kTestModelVersion);
  ASSERT_TRUE(response.outputs().contains("y"));
  // y = 0.5 * x + 2.
  ASSERT_EQ(response.outputs().at("y").float_val_size(), 1);
  EXPECT_FLOAT_EQ(response.outputs().at("y").float_val(0), 3.0f);
}

// A handshake message followed by the payload yields exactly one response for
// the payload.
TEST_F(TfrtSavedModelServableTest,
       PredictStreamedWithHandshakeReturnsOneResponse) {
  ServableHandle<Servable> servable = GetServable();

  std::vector<absl::StatusOr<PredictResponse>> responses;
  absl::StatusOr<std::unique_ptr<PredictStreamedContext>> context =
      servable->PredictStreamed(
          Servable::RunOptions(),
          [&responses](absl::StatusOr<PredictResponse> response) {
            responses.push_back(std::move(response));
          });
  TF_ASSERT_OK(context.status());

  PredictRequest handshake;
  handshake.mutable_model_spec()->set_name(kTestModelName);
  handshake.mutable_request_options()
      ->mutable_handshake()
      ->set_estimated_payload_bytes(64);
  TF_ASSERT_OK((*context)->ProcessRequest(&handshake));
  EXPECT_TRUE(responses.empty());

  PredictRequest request = CreateRequest(4.0f);
  TF_ASSERT_OK((*context)->ProcessRequest(&request));
  TF_ASSERT_OK((*context)->Close());

  ASSERT_EQ(responses.size(), 1);
  TF_ASSERT_OK(responses[0].status());
  ASSERT_EQ(responses[0]->outputs().at("y").float_val_size(), 1);
  EXPECT_FLOAT_EQ(responses[0]->outputs().at("y").float_val(0), 4.0f);
}

// SavedModelImpl::LoadSavedModel moves GraphDef out of MetaGraphDef into
// TfrtGraphExecutionState. Verify that SupportsStreamedOutputs still detects
// PwStreamResults in the function library and installs streamed_output_callback
// (causing TFRT to reject the non-streaming 'serving_default' signature).
TEST_F(TfrtSavedModelServableTest,
       PredictStreamedSavedModelImplDetectsPwStreamResultsInFunctionLibrary) {
  const std::string saved_model_dir = test_util::TestSrcDirPath(
      "servables/tensorflow/testdata/saved_model_half_plus_two_tf2_cpu/"
      "00000123");
  MetaGraphDef meta_graph_def;
  TF_ASSERT_OK(ReadMetaGraphDefFromSavedModel(
      saved_model_dir, {kSavedModelTagServe}, &meta_graph_def));

  FunctionDef* fn =
      meta_graph_def.mutable_graph_def()->mutable_library()->add_function();
  fn->mutable_signature()->set_name("streaming_fn");
  NodeDef* node = fn->add_node_def();
  node->set_name("pw_stream_results");
  node->set_op("PwStreamResults");

  tfrt_stub::SavedModel::Options options(tfrt_stub::GetGlobalRuntime());
  absl::StatusOr<std::unique_ptr<tfrt_stub::SavedModel>> saved_model =
      tfrt_stub::SavedModelImpl::LoadSavedModel(
          options, std::move(meta_graph_def), saved_model_dir);
  TF_ASSERT_OK(saved_model.status());
  ASSERT_TRUE((*saved_model)
                  ->GetMetaGraphDef()
                  .graph_def()
                  .library()
                  .function()
                  .empty());

  TfrtSavedModelConfig config;
  SavedModelConfig model_config;
  TfrtSavedModelServable servable(kTestModelName, kTestModelVersion, config,
                                  model_config, *std::move(saved_model),
                                  /*thread_pool_factory=*/nullptr);

  std::vector<absl::StatusOr<PredictResponse>> responses;
  absl::StatusOr<std::unique_ptr<PredictStreamedContext>> context =
      servable.PredictStreamed(
          Servable::RunOptions(),
          [&responses](absl::StatusOr<PredictResponse> response) {
            responses.push_back(std::move(response));
          });
  TF_ASSERT_OK(context.status());

  PredictRequest request = CreateRequest(2.0f);
  absl::Status status = (*context)->ProcessRequest(&request);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(), HasSubstr("does not support streaming"));
  EXPECT_TRUE(responses.empty());
}

// Verify that SupportsStreamedOutputs detects a PwStreamResults node in the
// top-level Graph of TfrtGraphExecutionState after LoadSavedModel moves
// GraphDef out of MetaGraphDef.
TEST_F(TfrtSavedModelServableTest,
       PredictStreamedSavedModelImplDetectsPwStreamResultsInTopLevelGraph) {
  const std::string saved_model_dir = test_util::TestSrcDirPath(
      "servables/tensorflow/testdata/saved_model_half_plus_two_tf2_cpu/"
      "00000123");
  MetaGraphDef meta_graph_def;
  TF_ASSERT_OK(ReadMetaGraphDefFromSavedModel(
      saved_model_dir, {kSavedModelTagServe}, &meta_graph_def));

  // Register an empty 0-arg FunctionDef named "PwStreamResults" (with no
  // internal node_defs) so ConvertGraphDefToGraph resolves the op definition
  // for the top-level NodeDef without triggering the function library check.
  FunctionDef* fn =
      meta_graph_def.mutable_graph_def()->mutable_library()->add_function();
  fn->mutable_signature()->set_name("PwStreamResults");
  NodeDef* node = meta_graph_def.mutable_graph_def()->add_node();
  node->set_name("pw_stream_results");
  node->set_op("PwStreamResults");

  tfrt_stub::SavedModel::Options options(tfrt_stub::GetGlobalRuntime());
  absl::StatusOr<std::unique_ptr<tfrt_stub::SavedModel>> saved_model =
      tfrt_stub::SavedModelImpl::LoadSavedModel(
          options, std::move(meta_graph_def), saved_model_dir);
  TF_ASSERT_OK(saved_model.status());
  ASSERT_TRUE((*saved_model)->GetMetaGraphDef().graph_def().node().empty());

  TfrtSavedModelConfig config;
  SavedModelConfig model_config;
  TfrtSavedModelServable servable(kTestModelName, kTestModelVersion, config,
                                  model_config, *std::move(saved_model),
                                  /*thread_pool_factory=*/nullptr);

  std::vector<absl::StatusOr<PredictResponse>> responses;
  absl::StatusOr<std::unique_ptr<PredictStreamedContext>> context =
      servable.PredictStreamed(
          Servable::RunOptions(),
          [&responses](absl::StatusOr<PredictResponse> response) {
            responses.push_back(std::move(response));
          });
  TF_ASSERT_OK(context.status());

  PredictRequest request = CreateRequest(2.0f);
  absl::Status status = (*context)->ProcessRequest(&request);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(), HasSubstr("does not support streaming"));
  EXPECT_TRUE(responses.empty());
}

// A streaming model that emits responses through streamed_output_callback and
// also has regular signature return tensors must only emit the streamed
// responses, not an extra final response from RunPredict.
TEST_F(TfrtSavedModelServableTest,
       PredictStreamedStreamingModelEmitsOnlyStreamedOutputs) {
  auto mock_saved_model =
      std::make_unique<::testing::NiceMock<test_util::MockSavedModel>>();

  MetaGraphDef meta_graph_def;
  NodeDef* stream_node = meta_graph_def.mutable_graph_def()->add_node();
  stream_node->set_name("pw_stream_results");
  stream_node->set_op("PwStreamResults");
  SignatureDef& sig_def =
      (*meta_graph_def.mutable_signature_def())[kDefaultServingSignatureDefKey];
  (*sig_def.mutable_inputs())["x"].set_name("x:0");
  (*sig_def.mutable_outputs())["y"].set_name("y:0");
  ON_CALL(*mock_saved_model, GetMetaGraphDef())
      .WillByDefault(ReturnRef(meta_graph_def));

  tfrt_stub::internal::Signature internal_sig;
  internal_sig.input_names = {"x"};
  internal_sig.input_specs = {tfrt_stub::TensorSpec(tensorflow::DT_FLOAT)};
  internal_sig.output_names = {"y"};
  internal_sig.output_specs = {tfrt_stub::TensorSpec(tensorflow::DT_FLOAT)};
  tfrt_stub::FunctionMetadata func_metadata(&internal_sig);
  EXPECT_CALL(*mock_saved_model,
              GetFunctionMetadata(kDefaultServingSignatureDefKey))
      .WillRepeatedly(Return(func_metadata));

  EXPECT_CALL(*mock_saved_model, Run(_, kDefaultServingSignatureDefKey, _, _))
      .WillOnce([](const tfrt_stub::SavedModel::RunOptions& run_options,
                   absl::string_view func_name, absl::Span<const Tensor> inputs,
                   std::vector<Tensor>* outputs) {
        EXPECT_TRUE(run_options.streamed_output_callback != nullptr);
        if (run_options.streamed_output_callback) {
          Tensor chunk1(tensorflow::DT_FLOAT, TensorShape({1}));
          chunk1.flat<float>()(0) = 10.0f;
          run_options.streamed_output_callback({{"y", chunk1}});

          Tensor chunk2(tensorflow::DT_FLOAT, TensorShape({1}));
          chunk2.flat<float>()(0) = 20.0f;
          run_options.streamed_output_callback({{"y", chunk2}});
        }
        Tensor final_output(tensorflow::DT_FLOAT, TensorShape({1}));
        final_output.flat<float>()(0) = 99.0f;
        outputs->push_back(final_output);
        return absl::OkStatus();
      });

  TfrtSavedModelConfig config;
  SavedModelConfig model_config;
  TfrtSavedModelServable servable(kTestModelName, kTestModelVersion, config,
                                  model_config, std::move(mock_saved_model),
                                  /*thread_pool_factory=*/nullptr);

  std::vector<absl::StatusOr<PredictResponse>> responses;
  absl::StatusOr<std::unique_ptr<PredictStreamedContext>> context =
      servable.PredictStreamed(
          Servable::RunOptions(),
          [&responses](absl::StatusOr<PredictResponse> response) {
            responses.push_back(std::move(response));
          });
  TF_ASSERT_OK(context.status());

  PredictRequest request = CreateRequest(2.0f);
  TF_ASSERT_OK((*context)->ProcessRequest(&request));
  TF_ASSERT_OK((*context)->Close());

  ASSERT_EQ(responses.size(), 2);
  TF_ASSERT_OK(responses[0].status());
  EXPECT_FLOAT_EQ(responses[0]->outputs().at("y").float_val(0), 10.0f);
  TF_ASSERT_OK(responses[1].status());
  EXPECT_FLOAT_EQ(responses[1]->outputs().at("y").float_val(0), 20.0f);
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
