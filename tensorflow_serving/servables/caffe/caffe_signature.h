/*

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

#ifndef TENSORFLOW_SERVING_SERVABLES_CAFFE_SIGNATURE_H_
#define TENSORFLOW_SERVING_SERVABLES_CAFFE_SIGNATURE_H_

#include <string>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/contrib/session_bundle/manifest.pb.h"

#include "tensorflow_serving/servables/caffe/caffe_serving_session.h"

namespace tensorflow {
namespace serving {

// Get Signatures from a MetaGraphDef.
Status GetSignatures(const CaffeMetaGraphDef& meta_graph_def,
                     Signatures* signatures);

// Gets the default signature from a MetaGraphDef.
Status GetDefaultSignature(const CaffeMetaGraphDef& meta_graph_def,
                           Signature* default_signature);

Status GetNamedSignature(const string& name,
                         const CaffeMetaGraphDef& meta_graph_def,
                         Signature* signature);

// Gets a ClassificationSignature from a MetaGraphDef's default signature.
// Returns an error if the default signature is not a ClassificationSignature,
// or does not exist.
Status GetClassificationSignature(const CaffeMetaGraphDef& meta_graph_def,
                                  ClassificationSignature* signature);

// Gets a RegressionSignature from a MetaGraphDef's default signature.
// Returns an error if the default signature is not a RegressionSignature,
// or does not exist.
Status GetRegressionSignature(const CaffeMetaGraphDef& meta_graph_def,
                              RegressionSignature* signature);

}  // namespace
}  // namespace

#endif // TENSORFLOW_SERVING_SERVABLES_CAFFE_SIGNATURE_H_