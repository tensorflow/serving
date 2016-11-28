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

#include "tensorflow_serving/servables/caffe/caffe_signature.h"

#include <string>
#include <utility>
#include <vector>

#include "caffe/proto/caffe.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/protobuf_internal.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/contrib/session_bundle/manifest.pb.h"

#include "tensorflow_serving/servables/caffe/caffe_serving_session.h"

namespace tensorflow {
namespace serving {

Status GetSignatures(const CaffeMetaGraphDef& meta_graph_def,
                     Signatures* signatures) {
  const std::vector<string>& ins = meta_graph_def.resolved_inputs;
  const std::vector<string>& outs = meta_graph_def.resolved_outputs;

  if (ins.size() == 0)
    return errors::FailedPrecondition("Network has no inputs");

  if (outs.size() == 0)
    return errors::FailedPrecondition("Network has no outputs");

  // compute the default signature.
  if (ins.size() == 1 && outs.size() == 1) {
    if (meta_graph_def.classes.dtype() != DT_INVALID) {
      // classification sig.
      ClassificationSignature* s = signatures->mutable_default_signature()
                                       ->mutable_classification_signature();

      s->mutable_input()->set_tensor_name(ins[0]);
      s->mutable_scores()->set_tensor_name(outs[0]);
      s->mutable_classes()->set_tensor_name(kClassLabelTensorName);
    } else {
      // regression sig.
      RegressionSignature* s = signatures->mutable_default_signature()
                                   ->mutable_regression_signature();

      s->mutable_input()->set_tensor_name(ins[0]);
      s->mutable_output()->set_tensor_name(outs[0]);
    }
  }

  {  // compute generic named input signature
    GenericSignature* s = (*signatures->mutable_named_signatures())["inputs"]
                              .mutable_generic_signature();

    for (const auto& in : ins) {
      (*s->mutable_map())[in].set_tensor_name(in);
    }
  }
  {  // compute generic named output signature
    GenericSignature* s = (*signatures->mutable_named_signatures())["outputs"]
                              .mutable_generic_signature();

    for (const auto& out : outs) {
      (*s->mutable_map())[out].set_tensor_name(out);
    }
  }

  return Status::OK();
}

Status GetClassificationSignature(const CaffeMetaGraphDef& meta_graph_def,
                                  ClassificationSignature* signature) {
  Signatures signatures;
  TF_RETURN_IF_ERROR(GetSignatures(meta_graph_def, &signatures));
  if (!signatures.has_default_signature()) {
    return errors::FailedPrecondition(strings::StrCat(
        "Expected a default signature in: ", signatures.DebugString()));
  }
  if (!signatures.default_signature().has_classification_signature()) {
    return errors::FailedPrecondition(
        strings::StrCat("Expected a classification signature in: ",
                        signatures.default_signature().DebugString()));
  }
  *signature = signatures.default_signature().classification_signature();
  return Status::OK();
}

Status GetRegressionSignature(const CaffeMetaGraphDef& meta_graph_def,
                              RegressionSignature* signature) {
  Signatures signatures;
  TF_RETURN_IF_ERROR(GetSignatures(meta_graph_def, &signatures));
  if (!signatures.has_default_signature()) {
    return errors::FailedPrecondition(strings::StrCat(
        "Expected a default signature in: ", signatures.DebugString()));
  }
  if (!signatures.default_signature().has_regression_signature()) {
    return errors::FailedPrecondition(
        strings::StrCat("Expected a regression signature in: ",
                        signatures.default_signature().DebugString()));
  }
  *signature = signatures.default_signature().regression_signature();
  return Status::OK();
}

Status GetGenericSignature(const string& name,
                           const CaffeMetaGraphDef& meta_graph_def,
                           GenericSignature* signature) {
  Signatures signatures;
  TF_RETURN_IF_ERROR(GetSignatures(meta_graph_def, &signatures));
  const auto& it = signatures.named_signatures().find(name);
  if (it == signatures.named_signatures().end()) {
    return errors::InvalidArgument(
        strings::StrCat("Missing generic signature named \"", name, "\" in ",
                        signatures.DebugString()));
  }
  if (!it->second.has_generic_signature()) {
    return errors::InvalidArgument(strings::StrCat(
        "Expected a generic signature: ", it->second.DebugString()));
  }
  *signature = it->second.generic_signature();
  return Status::OK();
}

Status GetNamedSignature(const string& name,
                         const CaffeMetaGraphDef& meta_graph_def,
                         Signature* signature) {
  Signatures signatures;
  TF_RETURN_IF_ERROR(GetSignatures(meta_graph_def, &signatures));
  const auto& it = signatures.named_signatures().find(name);
  if (it == signatures.named_signatures().end()) {
    return errors::NotFound(
        strings::StrCat("Missing signature named \"", name, "\" in: ",
                        DebugStringIfAvailable(signatures)));
  }
  *signature = it->second;
  return Status::OK();
}

Status GetDefaultSignature(const CaffeMetaGraphDef& meta_graph_def,
                           Signature* default_signature) {
  Signatures signatures;
  TF_RETURN_IF_ERROR(GetSignatures(meta_graph_def, &signatures));
  *default_signature = signatures.default_signature();
  return Status::OK();
}

}  // namespace serving
}  // namespace tensorflow
