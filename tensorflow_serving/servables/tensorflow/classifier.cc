/* Copyright 2017 Google Inc. All Rights Reserved.

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

#include "tensorflow_serving/servables/tensorflow/classifier.h"

#include <stddef.h>
#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/cc/saved_model/signature_constants.h"
#include "tensorflow/contrib/session_bundle/signature.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_serving/apis/classification.pb.h"
#include "tensorflow_serving/apis/classifier.h"
#include "tensorflow_serving/apis/input.pb.h"
#include "tensorflow_serving/apis/model.pb.h"
#include "tensorflow_serving/servables/tensorflow/util.h"

namespace tensorflow {
namespace serving {
namespace {

// Implementation of the ClassificationService using the legacy SessionBundle
// ClassificationSignature signature format.
class TensorFlowClassifier : public ClassifierInterface {
 public:
  explicit TensorFlowClassifier(Session* session,
                                const ClassificationSignature* signature)
      : session_(session), signature_(signature) {}

  Status Classify(const ClassificationRequest& request,
                  ClassificationResult* result) override {
    TRACELITERAL("TensorFlowClassifier::Classify");
    TRACELITERAL("ConvertInputTFEXamplesToTensor");
    // Setup the input Tensor to be a vector of string containing the serialized
    // tensorflow.Example.
    Tensor input_tensor;
    TF_RETURN_IF_ERROR(
        InputToSerializedExampleTensor(request.input(), &input_tensor));

    const int num_examples = input_tensor.dim_size(0);
    if (num_examples == 0) {
      return errors::InvalidArgument("ClassificationRequest::input is empty.");
    }

    TRACELITERAL("RunClassification");
    // Support serving models that return classes, scores or both.
    std::unique_ptr<Tensor> classes;
    if (!signature_->classes().tensor_name().empty()) {
      classes.reset(new Tensor);
    }
    std::unique_ptr<Tensor> scores;
    if (!signature_->scores().tensor_name().empty()) {
      scores.reset(new Tensor);
    }

    TF_RETURN_IF_ERROR(RunClassification(*signature_, input_tensor, session_,
                                         classes.get(), scores.get()));

    // Validate classes output Tensor.
    if (classes) {
      if (classes->dims() != 2) {
        return errors::InvalidArgument(
            "Expected Tensor shape: [batch_size num_classes] but got ",
            classes->shape().DebugString());
      }
      if (classes->dtype() != DT_STRING) {
        return errors::Internal("Expected classes Tensor of DT_STRING.  Got: ",
                                DataType_Name(classes->dtype()));
      }
      if (classes->dim_size(0) != num_examples) {
        return errors::Internal("Expected output batch size of ", num_examples,
                                ". Got: ", classes->dim_size(0));
      }
    }
    // Validate scores output Tensor.
    if (scores) {
      if (scores->dims() != 2) {
        return errors::InvalidArgument(
            "Expected Tensor shape: [batch_size num_classes] but got ",
            scores->shape().DebugString());
      }
      if (scores->dtype() != DT_FLOAT) {
        return errors::Internal("Expected scores Tensor of DT_FLOAT.  Got: ",
                                DataType_Name(scores->dtype()));
      }
      if (scores->dim_size(0) != num_examples) {
        return errors::Internal("Expected output batch size of ", num_examples,
                                ". Got: ", scores->dim_size(0));
      }
    }
    // Extract the number of classes from either the class or score output
    // Tensor.
    int num_classes = 0;
    if (classes && scores) {
      // If we have both Tensors they should agree in the second dimmension.
      if (classes->dim_size(1) != scores->dim_size(1)) {
        return errors::Internal(
            "Tensors class and score should match in dim_size(1). Got ",
            classes->dim_size(1), " vs. ", scores->dim_size(1));
      }
      num_classes = classes->dim_size(1);
    } else if (classes) {
      num_classes = classes->dim_size(1);
    } else if (scores) {
      num_classes = scores->dim_size(1);
    }

    TRACELITERAL("ConvertToClassificationResult");
    // Convert the output to ClassificationResult format.
    for (int i = 0; i < num_examples; ++i) {
      serving::Classifications* classifications = result->add_classifications();
      for (int c = 0; c < num_classes; ++c) {
        serving::Class* cl = classifications->add_classes();
        if (classes) {
          cl->set_label((classes->matrix<string>())(i, c));
        }
        if (scores) {
          cl->set_score((scores->matrix<float>())(i, c));
        }
      }
    }

    return Status::OK();
  }

 private:
  Session* const session_;
  const ClassificationSignature* const signature_;

  TF_DISALLOW_COPY_AND_ASSIGN(TensorFlowClassifier);
};

// Implementation of the ClassifierInterface using SavedModel.
class SavedModelTensorFlowClassifier : public ClassifierInterface {
 public:
  explicit SavedModelTensorFlowClassifier(const RunOptions& run_options,
                                          Session* session,
                                          const SignatureDef* const signature)
      : run_options_(run_options), session_(session), signature_(signature) {}

  ~SavedModelTensorFlowClassifier() override = default;

  Status Classify(const ClassificationRequest& request,
                  ClassificationResult* result) override {
    TRACELITERAL("TensorFlowClassifier::Classify");

    string input_tensor_name;
    std::vector<string> output_tensor_names;
    TF_RETURN_IF_ERROR(PreProcessClassification(*signature_, &input_tensor_name,
                                                &output_tensor_names));

    std::vector<Tensor> outputs;
    int num_examples;
    TF_RETURN_IF_ERROR(PerformOneShotTensorComputation(
        run_options_, request.input(), input_tensor_name, output_tensor_names,
        session_, &outputs, &num_examples));

    TRACELITERAL("ConvertToClassificationResult");
    return PostProcessClassificationResult(
        *signature_, num_examples, output_tensor_names, outputs, result);
  }

 private:
  const RunOptions run_options_;
  Session* const session_;
  const SignatureDef* const signature_;

  TF_DISALLOW_COPY_AND_ASSIGN(SavedModelTensorFlowClassifier);
};

// Implementation of the ClassificationService.
class SessionBundleClassifier : public ClassifierInterface {
 public:
  explicit SessionBundleClassifier(std::unique_ptr<SessionBundle> bundle)
      : bundle_(std::move(bundle)) {}

  ~SessionBundleClassifier() override = default;

  Status Classify(const ClassificationRequest& request,
                  ClassificationResult* result) override {
    // Get the default signature of the graph.  Expected to be a
    // classification signature.
    // TODO(b/26220896): Move TensorFlowClassifier creation to construction
    // time.
    ClassificationSignature signature;
    TF_RETURN_IF_ERROR(
        GetClassificationSignature(bundle_->meta_graph_def, &signature));

    TensorFlowClassifier classifier(bundle_->session.get(), &signature);
    return classifier.Classify(request, result);
  }

 private:
  std::unique_ptr<SessionBundle> bundle_;

  TF_DISALLOW_COPY_AND_ASSIGN(SessionBundleClassifier);
};

class SavedModelClassifier : public ClassifierInterface {
 public:
  SavedModelClassifier(const RunOptions& run_options,
                       std::unique_ptr<SavedModelBundle> bundle)
      : run_options_(run_options), bundle_(std::move(bundle)) {}

  ~SavedModelClassifier() override = default;

  Status Classify(const ClassificationRequest& request,
                  ClassificationResult* result) override {
    // Get the default signature of the graph.  Expected to be a
    // classification signature.
    // TODO(b/26220896): Move TensorFlowClassifier creation to construction
    // time.
    SignatureDef signature;
    TF_RETURN_IF_ERROR(GetClassificationSignatureDef(
        request.model_spec(), bundle_->meta_graph_def, &signature));
    SavedModelTensorFlowClassifier classifier(
        run_options_, bundle_->session.get(), &signature);
    return classifier.Classify(request, result);
  }

 private:
  const RunOptions run_options_;
  std::unique_ptr<SavedModelBundle> bundle_;

  TF_DISALLOW_COPY_AND_ASSIGN(SavedModelClassifier);
};

}  // namespace

Status CreateClassifierFromBundle(
    std::unique_ptr<SessionBundle> bundle,
    std::unique_ptr<ClassifierInterface>* service) {
  service->reset(new SessionBundleClassifier(std::move(bundle)));
  return Status::OK();
}

Status CreateClassifierFromSavedModelBundle(
    const RunOptions& run_options, std::unique_ptr<SavedModelBundle> bundle,
    std::unique_ptr<ClassifierInterface>* service) {
  service->reset(new SavedModelClassifier(run_options, std::move(bundle)));
  return Status::OK();
}

Status CreateFlyweightTensorFlowClassifier(
    Session* session, const ClassificationSignature* const signature,
    std::unique_ptr<ClassifierInterface>* service) {
  service->reset(new TensorFlowClassifier(session, signature));
  return Status::OK();
}

Status CreateFlyweightTensorFlowClassifier(
    const RunOptions& run_options, Session* session,
    const SignatureDef* signature,
    std::unique_ptr<ClassifierInterface>* service) {
  service->reset(
      new SavedModelTensorFlowClassifier(run_options, session, signature));
  return Status::OK();
}

Status GetClassificationSignatureDef(const ModelSpec& model_spec,
                                     const MetaGraphDef& meta_graph_def,
                                     SignatureDef* signature) {
  const string signature_name = model_spec.signature_name().empty()
                                    ? kDefaultServingSignatureDefKey
                                    : model_spec.signature_name();
  auto iter = meta_graph_def.signature_def().find(signature_name);
  if (iter == meta_graph_def.signature_def().end()) {
    return errors::InvalidArgument(strings::StrCat(
        "No signature was found with the name: ", signature_name));
  }
  if (iter->second.method_name() != kClassifyMethodName) {
    return errors::InvalidArgument(strings::StrCat(
        "Expected classification signature method_name to be ",
        kClassifyMethodName, ". Was: ", iter->second.method_name()));
  }
  *signature = iter->second;
  return Status::OK();
}

Status PreProcessClassification(const SignatureDef& signature,
                                string* input_tensor_name,
                                std::vector<string>* output_tensor_names) {
  if (signature.method_name() != kClassifyMethodName) {
    return errors::InvalidArgument(strings::StrCat(
        "Expected classification signature method_name to be ",
        kClassifyMethodName, ". Was: ", signature.method_name()));
  }
  if (signature.inputs().size() != 1) {
    return errors::InvalidArgument(
        strings::StrCat("Expected one input Tensor."));
  }
  if (signature.outputs().size() != 1 && signature.outputs().size() != 2) {
    return errors::InvalidArgument(
        strings::StrCat("Expected one or two output Tensors, found ",
                        signature.outputs().size()));
  }

  auto input_iter = signature.inputs().find(kClassifyInputs);
  if (input_iter == signature.inputs().end()) {
    return errors::FailedPrecondition(
        "No classification inputs found in SignatureDef: ",
        signature.DebugString());
  }
  *input_tensor_name = input_iter->second.name();

  auto classes_iter = signature.outputs().find(kClassifyOutputClasses);
  auto scores_iter = signature.outputs().find(kClassifyOutputScores);
  if (classes_iter == signature.outputs().end() &&
      scores_iter == signature.outputs().end()) {
    return errors::FailedPrecondition(strings::StrCat(
        "Expected classification signature outputs to contain at least one of ",
        "\"", kClassifyOutputClasses, "\" or \"", kClassifyOutputScores,
        "\". Signature was: ", signature.DebugString()));
  }
  if (classes_iter != signature.outputs().end()) {
    output_tensor_names->push_back(classes_iter->second.name());
  }
  if (scores_iter != signature.outputs().end()) {
    output_tensor_names->push_back(scores_iter->second.name());
  }
  return Status::OK();
}

Status PostProcessClassificationResult(
    const SignatureDef& signature, int num_examples,
    const std::vector<string>& output_tensor_names,
    const std::vector<Tensor>& output_tensors, ClassificationResult* result) {
  if (output_tensors.size() != output_tensor_names.size()) {
    return errors::InvalidArgument(
        strings::StrCat("Expected ", output_tensor_names.size(),
                        " output tensor(s).  Got: ", output_tensors.size()));
  }

  auto classes_iter = signature.outputs().find(kClassifyOutputClasses);
  string classes_tensor_name;
  if (classes_iter != signature.outputs().end()) {
    classes_tensor_name = classes_iter->second.name();
  }
  auto scores_iter = signature.outputs().find(kClassifyOutputScores);
  string scores_tensor_name;
  if (scores_iter != signature.outputs().end()) {
    scores_tensor_name = scores_iter->second.name();
  }

  const Tensor* classes = nullptr;
  const Tensor* scores = nullptr;
  for (int i = 0; i < output_tensors.size(); ++i) {
    if (output_tensor_names[i] == classes_tensor_name) {
      classes = &output_tensors[i];
    } else if (output_tensor_names[i] == scores_tensor_name) {
      scores = &output_tensors[i];
    }
  }

  // Validate classes output Tensor.
  if (classes) {
    if (classes->dims() != 2) {
      return errors::InvalidArgument(
          "Expected Tensor shape: [batch_size num_classes] but got ",
          classes->shape().DebugString());
    }
    if (classes->dtype() != DT_STRING) {
      return errors::InvalidArgument(
          "Expected classes Tensor of DT_STRING. Got: ",
          DataType_Name(classes->dtype()));
    }
    if (classes->dim_size(0) != num_examples) {
      return errors::InvalidArgument("Expected classes output batch size of ",
                                     num_examples,
                                     ". Got: ", classes->dim_size(0));
    }
  }
  // Validate scores output Tensor.
  if (scores) {
    if (scores->dims() != 2) {
      return errors::InvalidArgument(
          "Expected Tensor shape: [batch_size num_classes] but got ",
          scores->shape().DebugString());
    }
    if (scores->dtype() != DT_FLOAT) {
      return errors::InvalidArgument(
          "Expected scores Tensor of DT_FLOAT. Got: ",
          DataType_Name(scores->dtype()));
    }
    if (scores->dim_size(0) != num_examples) {
      return errors::InvalidArgument("Expected scores output batch size of ",
                                     num_examples,
                                     ". Got: ", scores->dim_size(0));
    }
  }
  // Extract the number of classes from either the class or score output
  // Tensor.
  int num_classes = 0;
  if (classes && scores) {
    // If we have both Tensors they should agree in the second dimmension.
    if (classes->dim_size(1) != scores->dim_size(1)) {
      return errors::InvalidArgument(
          "Tensors class and score should match in dim_size(1). Got ",
          classes->dim_size(1), " vs. ", scores->dim_size(1));
    }
    num_classes = classes->dim_size(1);
  } else if (classes) {
    num_classes = classes->dim_size(1);
  } else if (scores) {
    num_classes = scores->dim_size(1);
  }

  // Convert the output to ClassificationResult format.
  for (int i = 0; i < num_examples; ++i) {
    serving::Classifications* classifications = result->add_classifications();
    for (int c = 0; c < num_classes; ++c) {
      serving::Class* cl = classifications->add_classes();
      if (classes) {
        cl->set_label((classes->matrix<string>())(i, c));
      }
      if (scores) {
        cl->set_score((scores->matrix<float>())(i, c));
      }
    }
  }
  return Status::OK();
}

}  // namespace serving
}  // namespace tensorflow
