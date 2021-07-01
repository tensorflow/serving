/*
 * Copyright 2021 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Definition of the kernels for the optimized inference (i.e. fast, low memory
// consumption) of Yggdrasil Decision Forest models.
//
// "tf_op.py" contains a utility class to use these ops.
//
// Synergy of the OPs:
//
// A "LoadModel*" OP loads (and possibly optimizing) a model in memory.
//
// An "InferenceOp*" OP applies a model (previously loaded by "LoadModel") on
// a set of examples, and returns the predictions. The "tf_op.py" library
// contains a utility class able to build the input arguments of the
// "InferenceOp*", as the signature of this OP is non trivial.

#include "tensorflow/core/framework/op.h"

#include "tensorflow/core/framework/common_shape_fns.h"

namespace tensorflow {

REGISTER_OP("SimpleMLLoadModelFromPath")
    .SetIsStateful()
    .Attr("model_identifier: string")
    .Input("path: string")
    .Doc(R"(
Loads (and possibly compiles/optimizes) an Yggdrasil model in memory.

The model is then accessible in the "kModelContainer/{model_identifier}" TF
resource. If a model with the same "model_identifier" exists when this OP is
called (either from the same OP instance, or from another instance with the same
"model_identifier"), the model is discarded and replaced with the new model.

model_identifier: Unique identifier of the model. Used to create the name of
  the tf resource containing the model.

path: Path to the Yggdrasil model. Note: a Yggdrasil model directory should
  contains a "header.pb" file.

Returns a type-less OP that loads the model when called.
)");

REGISTER_OP("SimpleMLLoadModelFromPathWithHandle")
    .SetIsStateful()
    .Input("model_handle: resource")
    .Input("path: string")
    .Doc(R"(
Applies a model and returns its predictions.

Similar to "SimpleMLLoadModelFromPath", but takes a resource handle instead of
a resource name.
)");

Status SimpleMLInferenceOpSetShape(shape_inference::InferenceContext* c) {
  // Check the rank of the input features.
  ::tensorflow::shape_inference::ShapeHandle tmp_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &tmp_shape));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &tmp_shape));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &tmp_shape));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &tmp_shape));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 1, &tmp_shape));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 1, &tmp_shape));

  // Get the output batch dimension from the batch dimension of the input
  // features. Input batch dimension can be set or unknown. The special
  // value (0 for all inputs; except for input 5 with value 1) are ignored.
  shape_inference::DimensionOrConstant batch_size(0);
  bool batch_size_was_found = false;
  // Last known batch size. Use to ensure that, if the batch size size known
  // (at graph construction time), it is consistent in between the input
  // features.
  int known_batch_size = -1;
  for (const auto input_idx : {0, 1, 2, 5}) {
    auto candidate = c->Dim(c->input(input_idx), 0);
    if (c->ValueKnown(candidate)) {
      auto value = c->Value(candidate);
      // Note: The 5-th input has a reserved empty dimension.
      // Note: The API does not allow to test if a dimension handle is set.
      if (input_idx == 5) {
        value--;
      }
      if (value == 0) {
        // The feature is empty and ignored.
        continue;
      }
      if (known_batch_size == -1) {
        known_batch_size = value;
      } else if (known_batch_size != value) {
        return Status(error::INVALID_ARGUMENT,
                      "The batch size of the input features are inconsistent");
      }
    }

    if (!batch_size_was_found) {
      batch_size = candidate;
      batch_size_was_found = true;
    }
  }

  int dense_output_dim;
  TF_RETURN_IF_ERROR(c->GetAttr("dense_output_dim", &dense_output_dim));

  // Check the tensor shapes.
  TF_RETURN_IF_ERROR(c->set_output("dense_predictions",
                                   {c->Matrix(batch_size, dense_output_dim)}));
  TF_RETURN_IF_ERROR(
      c->set_output("dense_col_representation", {c->Vector(dense_output_dim)}));
  return Status::OK();
}

REGISTER_OP("SimpleMLInferenceOp")
    .SetIsStateful()
    .Attr("model_identifier: string")
    .Attr("dense_output_dim: int >= 1")
    .Input("numerical_features: float")
    .Input("boolean_features: float")
    .Input("categorical_int_features: int32")
    .Input("categorical_set_int_features_values: int32")
    .Input("categorical_set_int_features_row_splits_dim_1: int64")
    .Input("categorical_set_int_features_row_splits_dim_2: int64")
    .Output("dense_predictions: float")
    .Output("dense_col_representation: string")
    .SetShapeFn(SimpleMLInferenceOpSetShape)
    .Doc(R"(
Applies a model and returns its predictions.

This OP expects for a model to be loaded (e.g. by "LoadModelFromPath") before it
is called.

This OP expects for the input features to be flatten together, by type, as done
by the "_InferenceArgsBuilder" utility class in "tf_op.py". For example,
"numerical_features[i,j]" is the "j-th" numerical feature input of the model for
the "i-th "example in the batch.

model_identifier: Unique identifier of the model corresponding to a previously
  loaded model.

numerical_features: Numerical feature values. Tensor of shape "batch x
  numerical_features_dim" and type float32. "Quiet Nan" represents missing
  values.

boolean_features: Boolean feature values. Tensor of shape "batch x
  boolean_features_dim" and type float32. "Quiet Nan" represents missing
  values.

categorical_int_features: Categorical features stored as int. Tensor of shape
  "batch x categorical_int_features_dim" and type int32. -1 represents a missing
  value. 0 represents an "out of vocabulary" value (when applicable).

categorical_set_int_features_{values,dim_1,dim_2}: The value and two dimension
  index set of a ragged tensor of shape "batch x num_categorical_set_features x
  num_items" i.e "x.values, x.values.row_splits and x.row_splits" respectively.
  For a given feature and example, [-1] represents a missing value.

dense_output_dim: Dimension of the model output. For regression,
  dense_output_dim is the output dimension (e.g. 1 for uni-dimensional
  regression). For classification, dense_output_dim is the number of classes.

dense_predictions: Tensor of shape [batch x dense_output_dim] of type float32.
  Contains a probability for classification, and a value for regression and
  ranking.

dense_col_representation: Tensor of shape [dense_output_dim] of type bytes.
  Contains the representation of the columns of the predictions output. For
  classification with string label, contains the name of the labels. For all
  the other cases, contains empty strings.
)");

// Similar to "SimpleMLInferenceOp", but takes a resource handle instead of a
// resource name.
REGISTER_OP("SimpleMLInferenceOpWithHandle")
    .SetIsStateful()
    .Attr("dense_output_dim: int >= 1")
    .Input("numerical_features: float")
    .Input("boolean_features: float")
    .Input("categorical_int_features: int32")
    .Input("categorical_set_int_features_values: int32")
    .Input("categorical_set_int_features_row_splits_dim_1: int64")
    .Input("categorical_set_int_features_row_splits_dim_2: int64")
    .Input("model_handle: resource")
    .Output("dense_predictions: float")
    .Output("dense_col_representation: string")
    .SetShapeFn(SimpleMLInferenceOpSetShape);

Status ScalarOutput(shape_inference::InferenceContext* c) {
  c->set_output(0, c->Scalar());
  return Status::OK();
}

REGISTER_OP("SimpleMLCreateModelResource")
    .SetIsStateful()
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Output("model_handle: resource")
    .SetShapeFn(ScalarOutput)
    .Doc(R"(
Creates a model resource and returns the handle.

container: Name of the container.

shared_name: Name of the possibly shared name.

model_handle: Boolean feature values. Tensor of shape "batch x
  boolean_features_dim" and type float32. "Quiet Nan" represents missing
  values.
)");

}  // namespace tensorflow
