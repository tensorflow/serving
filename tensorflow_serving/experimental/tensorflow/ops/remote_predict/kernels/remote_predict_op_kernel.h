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
#ifndef TENSORFLOW_SERVING_EXPERIMENTAL_TENSORFLOW_OPS_REMOTE_PREDICT_KERNELS_REMOTE_PREDICT_OP_KERNEL_H_
#define TENSORFLOW_SERVING_EXPERIMENTAL_TENSORFLOW_OPS_REMOTE_PREDICT_KERNELS_REMOTE_PREDICT_OP_KERNEL_H_

#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "google/protobuf/wrappers.pb.h"
#include "google/protobuf/map.h"
#include "absl/flags/declare.h"
#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/time/time.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/batching_util/warmup.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/core/protobuf/named_tensor.pb.h"
#include "tensorflow_serving/apis/model.pb.h"
#include "tensorflow_serving/apis/predict.pb.h"

ABSL_DECLARE_FLAG(bool, remote_predict_op_use_tensor_content);

ABSL_DECLARE_FLAG(bool, remote_predict_bypass_rpc_during_warmup);

namespace tensorflow {
namespace serving {

inline absl::Status SetTensorToZero(Tensor* tensor) {
  DataType dtype = tensor->dtype();
  if (dtype == DT_STRING) {
    auto flat = tensor->flat<tstring>();
    for (int i = 0; i < flat.size(); ++i) {
      flat(i) = "";
    }
    return absl::OkStatus();
  }

  if (tensorflow::DataTypeIsNumeric(dtype) || dtype == DT_BOOL) {
    if (tensor->TotalBytes() > 0) {
      std::memset(tensor->data(), 0, tensor->TotalBytes());
    }
    return absl::OkStatus();
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "Unsupported type for zero initialization: ", DataTypeString(dtype)));
}

typedef google::protobuf::Map<tensorflow::string, tensorflow::TensorProto> AliasTensorMap;

// Remote Predict Op kernel implementation class templated on different
// PredictionServiceStubTypes.
template <typename PredictionServiceStubType>
class RemotePredictOp : public AsyncOpKernel {
 public:
  explicit RemotePredictOp(OpKernelConstruction* context)
      : AsyncOpKernel(context) {
    string target_address;
    OP_REQUIRES_OK(context,
                   context->GetAttr("target_address", &target_address));
    OP_REQUIRES_OK(context, context->GetAttr("model_name", &model_name_));
    OP_REQUIRES_OK(context, context->GetAttr("model_version", &model_version_));
    OP_REQUIRES_OK(context, context->GetAttr("max_rpc_deadline_millis",
                                             &max_rpc_deadline_millis_));
    OP_REQUIRES_OK(context, context->GetAttr("fail_op_on_rpc_error",
                                             &fail_op_on_rpc_error_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("signature_name", &signature_name_));
    if (context->HasAttr("output_shapes")) {
      if (context->GetAttr("output_shapes", &output_shapes_).ok() &&
          !output_shapes_.empty()) {
        has_output_shapes_ = true;
      }
    }
    absl::Status prediction_service_status =
        PredictionServiceStubType::Create(target_address, &prediction_service_);
    OP_REQUIRES(context, prediction_service_status.ok(),
                tensorflow::Status(static_cast<::absl::StatusCode>(
                                       prediction_service_status.code()),
                                   prediction_service_status.message()));
  }

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
    if (absl::GetFlag(FLAGS_remote_predict_bypass_rpc_during_warmup) &&
        IsWarmup(context)) {
      int expected_outputs = context->num_outputs() - 2;
      if (has_output_shapes_ && output_shapes_.size() == expected_outputs) {
        OutputZeroFilledTensors(context, std::move(done));
        return;
      } else {
        LOG(WARNING)
            << "Cannot bypass RPC during warmup: missing or mismatched output "
               "shapes. Falling back to remote call. has_output_shapes: "
            << has_output_shapes_ << " expected_outputs: " << expected_outputs
            << " shapes_size: "
            << (has_output_shapes_ ? output_shapes_.size() : 0);
        // Fall through to remote call.
      }
    }

    // Get the input tensor alias names.
    const auto& input_tensor_aliases = context->input(0).flat<tstring>();

    // Get the input tensors.
    OpInputList input_tensors;
    OP_REQUIRES_OK_ASYNC(
        context, context->input_list("input_tensors", &input_tensors), done);
    // Get the output tensor alias names.
    // Directly index to output_tensor_aliases by moving past all the input
    // before it, including the input_tensor_aliases and input_tensors.
    auto output_tensor_aliases =
        context->input(1 + input_tensors.size()).flat<tstring>();

    // Build the PredictRequest.
    PredictRequest* request = new PredictRequest();

    request->mutable_model_spec()->set_name(model_name_);

    request->mutable_model_spec()->set_signature_name(signature_name_);

    if (model_version_ >= 0) {
      request->mutable_model_spec()->mutable_version()->set_value(
          model_version_);
    }

    AliasTensorMap& inputs = *request->mutable_inputs();
    for (int i = 0; i < input_tensor_aliases.size(); ++i) {
      tensorflow::TensorProto proto;
      if (absl::GetFlag(FLAGS_remote_predict_op_use_tensor_content)) {
        input_tensors[i].AsProtoTensorContent(&proto);
      } else {
        input_tensors[i].AsProtoField(&proto);
      }
      inputs[input_tensor_aliases(i)] = proto;
    }

    for (int i = 0; i < output_tensor_aliases.size(); ++i) {
      request->add_output_filter(tensorflow::string(output_tensor_aliases(i)));
    }

    PredictResponse* response = new PredictResponse();

    auto rpc_or = prediction_service_->CreateRpc(
        absl::Milliseconds(max_rpc_deadline_millis_));
    OP_REQUIRES_ASYNC(context, rpc_or.ok(),
                      tensorflow::Status(static_cast<::absl::StatusCode>(
                                             rpc_or.status().code()),
                                         rpc_or.status().message()),
                      [&]() {
                        delete request;
                        delete response;
                        done();
                      });
    auto rpc = rpc_or.value();
    auto callback = [this, context, rpc, request, response,
                     output_tensor_aliases, done](const absl::Status& status) {
      PostProcessResponse(context, response, status, fail_op_on_rpc_error_,
                          output_tensor_aliases, [&]() {
                            delete rpc;
                            delete request;
                            delete response;
                            done();
                          });
    };
    // Make the RPC call.
    prediction_service_->Predict(rpc, request, response, callback);
  }

  void PostProcessResponse(OpKernelContext* context, PredictResponse* response,
                           const absl::Status& rpc_status,
                           bool fail_op_on_rpc_error,
                           TTypes<const tstring>::Flat output_tensor_aliases,
                           DoneCallback rpc_done) {
    auto rpc_cleaner = gtl::MakeCleanup([&] { rpc_done(); });
    Tensor* status_code;
    OP_REQUIRES_OK_ASYNC(
        context, context->allocate_output(0, TensorShape({}), &status_code),
        rpc_cleaner.release());
    status_code->scalar<int>()() = static_cast<int>(rpc_status.code());
    Tensor* status_error_message;
    OP_REQUIRES_OK_ASYNC(
        context,
        context->allocate_output(1, TensorShape({}), &status_error_message),
        rpc_cleaner.release());
    status_error_message->scalar<tstring>()() = rpc_status.message();
    OpOutputList output_tensors_list;
    OP_REQUIRES_OK_ASYNC(
        context, context->output_list("output_tensors", &output_tensors_list),
        rpc_cleaner.release());
    // Process the response.
    if (!rpc_status.ok()) {
      if (fail_op_on_rpc_error) {
        OP_REQUIRES_OK_ASYNC(context,
                             tensorflow::Status(static_cast<::absl::StatusCode>(
                                                    rpc_status.code()),
                                                rpc_status.message()),
                             rpc_cleaner.release());
      } else {
        // Allocate some empty output for the output_tensors.
        for (int i = 0; i < output_tensors_list.size(); ++i) {
          Tensor* unused;
          OP_REQUIRES_OK_ASYNC(
              context,
              output_tensors_list.allocate(i, TensorShape({}), &unused),
              rpc_cleaner.release());
        }
        return;
      }
    }
    OP_REQUIRES_ASYNC(
        context, output_tensors_list.size() == output_tensor_aliases.size(),
        errors::Internal(
            "Response doesn't have the right number of outputs; actual: ",
            output_tensors_list.size(),
            " expected: ", output_tensor_aliases.size()),
        rpc_cleaner.release());
    AliasTensorMap& outputs = *response->mutable_outputs();
    for (int i = 0; i < output_tensor_aliases.size(); i++) {
      Tensor output_tensor;
      OP_REQUIRES_ASYNC(
          context, output_tensor.FromProto(outputs[output_tensor_aliases(i)]),
          errors::Internal("Response tensor proto: ",
                           tensorflow::string(output_tensor_aliases(i)),
                           " cannot be converted back to a tensor."),
          rpc_cleaner.release());
      output_tensors_list.set(i, output_tensor);
    }
  }

  bool IsWarmup(OpKernelContext* context) {
    const auto* metadata = context->session_metadata();
    if (metadata == nullptr) {
      LOG_FIRST_N(INFO, 10) << "RemotePredictOp: SessionMetadata is null";
      return false;
    }
    tensorflow::serving::WarmupStateRegistry::Key tf_key(metadata->name(),
                                                         metadata->version());
    bool tf_in_warmup =
        tensorflow::serving::GetGlobalWarmupStateRegistry().Lookup(tf_key) !=
        nullptr;

    LOG_FIRST_N(INFO, 10) << "RemotePredictOp: SessionMetadata name="
                          << metadata->name()
                          << " version=" << metadata->version()
                          << " tf_in_warmup=" << tf_in_warmup;
    return tf_in_warmup;
  }

  void OutputZeroFilledTensors(OpKernelContext* context, DoneCallback done) {
    OpInputList input_tensors;
    absl::Status status = context->input_list("input_tensors", &input_tensors);
    int64_t batch_size = -1;
    if (status.ok() && input_tensors.size() > 0 &&
        input_tensors[0].dims() > 0) {
      batch_size = input_tensors[0].dim_size(0);
    }

    LOG(INFO) << "Bypassing RPC during warmup with batch_size = " << batch_size
              << " and output shapes: "
              << absl::StrJoin(output_shapes_, ",",
                               [](std::string* out, const TensorShape& shape) {
                                 absl::StrAppend(out, shape.DebugString());
                               });
    auto cleaner = gtl::MakeCleanup([&] { done(); });

    Tensor* status_code = nullptr;
    status = context->allocate_output(0, TensorShape({}), &status_code);
    if (!status.ok()) {
      context->CtxFailure(status);
      return;
    }
    status_code->scalar<int>()() = 0;

    Tensor* status_error_message = nullptr;
    status =
        context->allocate_output(1, TensorShape({}), &status_error_message);
    if (!status.ok()) {
      context->CtxFailure(status);
      return;
    }
    status_error_message->scalar<tstring>()() = "";

    OpOutputList output_tensors_list;
    status = context->output_list("output_tensors", &output_tensors_list);
    if (!status.ok()) {
      context->CtxFailure(status);
      return;
    }

    for (int i = 0; i < output_shapes_.size(); ++i) {
      Tensor* output_tensor = nullptr;
      TensorShape shape = output_shapes_[i];
      if (batch_size != -1 && shape.dims() > 0) {
        shape.set_dim(0, batch_size);
      }
      status = output_tensors_list.allocate(i, shape, &output_tensor);
      if (!status.ok()) {
        context->CtxFailure(status);
        return;
      }
      status = SetTensorToZero(output_tensor);
      if (!status.ok()) {
        context->CtxFailure(status);
        return;
      }
    }
  }

 private:
  string model_name_;
  int64_t model_version_;
  bool fail_op_on_rpc_error_;
  int64_t max_rpc_deadline_millis_;
  string signature_name_;
  std::vector<TensorShape> output_shapes_;
  bool has_output_shapes_ = false;
  std::unique_ptr<PredictionServiceStubType> prediction_service_;
};

}  // namespace serving
}  // namespace tensorflow
#endif  // TENSORFLOW_SERVING_EXPERIMENTAL_TENSORFLOW_OPS_REMOTE_PREDICT_KERNELS_REMOTE_PREDICT_OP_KERNEL_H_
