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

#include "tensorflow_serving/servables/tensorflow/saved_model_warmup_test_util.h"

#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/cc/saved_model/signature_constants.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow_serving/apis/prediction_log.pb.h"

namespace tensorflow {
namespace serving {

void PopulateInferenceTask(const string& model_name,
                           const string& signature_name,
                           const string& method_name, InferenceTask* task) {
  ModelSpec model_spec;
  model_spec.set_name(model_name);
  model_spec.set_signature_name(signature_name);
  *task->mutable_model_spec() = model_spec;
  task->set_method_name(method_name);
}

void PopulateMultiInferenceRequest(MultiInferenceRequest* request) {
  request->mutable_input()->mutable_example_list()->add_examples();
  PopulateInferenceTask("test_model", kRegressMethodName, kRegressMethodName,
                        request->add_tasks());
  PopulateInferenceTask("test_model", kClassifyMethodName, kClassifyMethodName,
                        request->add_tasks());
}

void PopulatePredictRequest(PredictRequest* request) {
  request->mutable_model_spec()->set_signature_name(kPredictMethodName);
  TensorProto tensor_proto;
  tensor_proto.add_string_val("input_value");
  tensor_proto.set_dtype(tensorflow::DT_STRING);
  tensor_proto.mutable_tensor_shape()->add_dim()->set_size(1);
  (*request->mutable_inputs())[kPredictInputs] = tensor_proto;
}

void PopulateClassificationRequest(ClassificationRequest* request) {
  request->mutable_input()->mutable_example_list()->add_examples();
  request->mutable_model_spec()->set_signature_name(kClassifyMethodName);
}

void PopulateRegressionRequest(RegressionRequest* request) {
  request->mutable_input()->mutable_example_list()->add_examples();
  request->mutable_model_spec()->set_signature_name(kRegressMethodName);
}

void PopulatePredictionLog(PredictionLog* prediction_log,
                           PredictionLog::LogTypeCase log_type) {
  switch (log_type) {
    case PredictionLog::kRegressLog: {
      PopulateRegressionRequest(
          prediction_log->mutable_regress_log()->mutable_request());
    } break;
    case PredictionLog::kClassifyLog: {
      PopulateClassificationRequest(
          prediction_log->mutable_classify_log()->mutable_request());
    } break;
    case PredictionLog::kPredictLog: {
      PopulatePredictRequest(
          prediction_log->mutable_predict_log()->mutable_request());
    } break;
    case PredictionLog::kMultiInferenceLog: {
      PopulateMultiInferenceRequest(
          prediction_log->mutable_multi_inference_log()->mutable_request());
    } break;
    case PredictionLog::kSessionRunLog:
      prediction_log->mutable_session_run_log();
      TF_FALLTHROUGH_INTENDED;
    default:
      return;
  }
}

Status WriteWarmupData(const string& fname,
                       const std::vector<string>& warmup_records,
                       int num_warmup_records) {
  Env* env = Env::Default();
  std::unique_ptr<WritableFile> file;
  TF_RETURN_IF_ERROR(env->NewWritableFile(fname, &file));

  io::RecordWriterOptions options;
  io::RecordWriter writer(file.get(), options);
  for (int i = 0; i < num_warmup_records; ++i) {
    for (const string& warmup_record : warmup_records) {
      TF_RETURN_IF_ERROR(writer.WriteRecord(warmup_record));
    }
  }
  TF_RETURN_IF_ERROR(writer.Flush());
  return OkStatus();
}

Status WriteWarmupDataAsSerializedProtos(
    const string& fname, const std::vector<string>& warmup_records,
    int num_warmup_records) {
  Env* env = Env::Default();
  std::unique_ptr<WritableFile> file;
  TF_RETURN_IF_ERROR(env->NewWritableFile(fname, &file));
  for (int i = 0; i < num_warmup_records; ++i) {
    for (const string& warmup_record : warmup_records) {
      TF_RETURN_IF_ERROR(file->Append(warmup_record));
    }
  }
  TF_RETURN_IF_ERROR(file->Close());
  return OkStatus();
}

void AddMixedWarmupData(
    std::vector<string>* warmup_records,
    const std::vector<PredictionLog::LogTypeCase>& log_types) {
  for (auto& log_type : log_types) {
    PredictionLog prediction_log;
    PopulatePredictionLog(&prediction_log, log_type);
    warmup_records->push_back(prediction_log.SerializeAsString());
  }
}

// Creates a test SignatureDef with the given parameters
SignatureDef CreateSignatureDef(const string& method_name,
                                const std::vector<string>& input_names,
                                const std::vector<string>& output_names) {
  SignatureDef signature_def;
  signature_def.set_method_name(method_name);
  for (const string& input_name : input_names) {
    TensorInfo input;
    input.set_name(input_name);
    (*signature_def.mutable_inputs())[input_name] = input;
  }
  for (const string& output_name : output_names) {
    TensorInfo output;
    output.set_name(output_name);
    (*signature_def.mutable_outputs())[output_name] = output;
  }
  return signature_def;
}

void AddSignatures(MetaGraphDef* meta_graph_def) {
  (*meta_graph_def->mutable_signature_def())[kRegressMethodName] =
      CreateSignatureDef(kRegressMethodName, {kRegressInputs},
                         {kRegressOutputs});
  (*meta_graph_def->mutable_signature_def())[kClassifyMethodName] =
      CreateSignatureDef(kClassifyMethodName, {kClassifyInputs},
                         {kClassifyOutputClasses, kClassifyOutputScores});
  (*meta_graph_def->mutable_signature_def())[kPredictMethodName] =
      CreateSignatureDef(kPredictMethodName, {kPredictInputs},
                         {kPredictOutputs});
}

}  // namespace serving
}  // namespace tensorflow
