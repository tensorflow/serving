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

// Inference of an Yggdrasil Decision Forests model in a TensorFlow OP.
//
// Overall description:
//
// A "loading" op (e.g. SimpleMLLoadModelFromPath) loads a simpleML model in
// memory and make it available for the inference OP.
//
// Details: Once loaded, the model is stored in a TF Resource of type
// "SimpleMLModelResource". The model is loaded in the simpleML
// abstract model format (i.e. a class deriving "AbstractModel"), and then
// possibly converted/compiled into a format suited for inference. This
// conversion, as well as the actual inference of the model, is handled by the
// "inference engine". An inference engine is instantiated (class extending
// "AbstractInferenceEngine").
//
// The inference OP "SimpleMLInferenceOp" then retrieves the model (i.e. the
// inference engine contained in the TF resource), generates a cache for the
// inference engine (an optional buffer that can be used by an inference engine
// to speed-up the computation / avoid memory allocation during inference). A
// cache can cannot be shared among different thread / inference ops). Finally,
// the inference OP runs the model.
//
// The op can run any simpleML model, included wrapped models. However, the
// dependencies to these model should be added manually.
//
// The "loading" ops and the "inference" ops need to share tf resources. Two
// systems are available to identify a resource: (1) a model_identifier (stored
// as string) or (2) a model_handle (stored as a resource handle).
//
#include <algorithm>

#include "absl/status/status.h"
#include "absl/strings/substitute.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/model_library.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"
#include "yggdrasil_decision_forests/utils/distribution.pb.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"
#include "yggdrasil_decision_forests/utils/tensorflow.h"

namespace tensorflow_decision_forests {
namespace ops {

namespace tf = ::tensorflow;
namespace model = ::yggdrasil_decision_forests::model;
namespace utils = ::yggdrasil_decision_forests::utils;
namespace dataset = ::yggdrasil_decision_forests::dataset;
namespace serving = ::yggdrasil_decision_forests::serving;

template <typename T>
using StatusOr = ::yggdrasil_decision_forests::utils::StatusOr<T>;

using Task = model::proto::Task;

using OpKernel = tf::OpKernel;
using OpKernelConstruction = tf::OpKernelConstruction;
using OpKernelContext = tf::OpKernelContext;
using TensorShape = tf::TensorShape;
using Tensor = tf::Tensor;

// Name of the "tf resource group" containing models loaded in memory.
constexpr char kModelContainer[] = "simple_ml_model_serving";

// Key of the attributes, inputs and outputs the OPs.
constexpr char kAttributeModelIdentifier[] = "model_identifier";
constexpr char kAttributeDenseOutputDim[] = "dense_output_dim";

constexpr char kInputPath[] = "path";
constexpr char kInputNumericalFeatures[] = "numerical_features";
constexpr char kInputBooleanFeatures[] = "boolean_features";
constexpr char kInputCategoricalIntFeatures[] = "categorical_int_features";
constexpr char kInputCategoricalSetIntFeaturesValues[] =
    "categorical_set_int_features_values";
constexpr char kInputCategoricalSetIntFeaturesRowSplitsDim1[] =
    "categorical_set_int_features_row_splits_dim_1";
constexpr char kInputCategoricalSetIntFeaturesRowSplitsDim2[] =
    "categorical_set_int_features_row_splits_dim_2";
constexpr char kInputModelHandle[] = "model_handle";

constexpr char kOutputDensePredictions[] = "dense_predictions";
constexpr char kOutputDenseColRepresentation[] = "dense_col_representation";

// Input tensor values of the model. Does not own the data.
struct InputTensors {
  InputTensors(
      const Tensor* numerical_features_tensor,
      const Tensor* boolean_features_tensor,
      const Tensor* categorical_int_features_tensor,
      const Tensor* categorical_set_int_features_values_tensor,
      const Tensor* categorical_set_int_features_row_splits_dim_1_tensor,
      const Tensor* categorical_set_int_features_row_splits_dim_2_tensor)
      : numerical_features(numerical_features_tensor->matrix<float>()),
        boolean_features(boolean_features_tensor->matrix<float>()),
        categorical_int_features(
            categorical_int_features_tensor->matrix<int32_t>()),
        categorical_set_int_features_values(
            categorical_set_int_features_values_tensor->vec<int32_t>()),
        categorical_set_int_features_row_splits_dim_1(
            categorical_set_int_features_row_splits_dim_1_tensor
                ->vec<int64_t>()),
        categorical_set_int_features_row_splits_dim_2(
            categorical_set_int_features_row_splits_dim_2_tensor
                ->vec<int64_t>()) {}

  const tf::TTypes<const float>::Matrix numerical_features;
  const tf::TTypes<const float>::Matrix boolean_features;
  const tf::TTypes<const int32_t>::Matrix categorical_int_features;

  const tf::TTypes<const int32_t>::Vec categorical_set_int_features_values;
  const tf::TTypes<const int64_t>::Vec
      categorical_set_int_features_row_splits_dim_1;
  const tf::TTypes<const int64_t>::Vec
      categorical_set_int_features_row_splits_dim_2;

  // This value is computed after the struct constructor. The value "-1" is a
  // holding value until the value is computed.
  int batch_size = -1;
};

// Output tensor values of the model. Does not own the data.
struct OutputTensors {
  OutputTensors(Tensor* dense_predictions_tensor,
                Tensor* dense_col_representation_tensor, const int output_dim)
      : dense_predictions(dense_predictions_tensor->matrix<float>()),
        dense_col_representation(
            dense_col_representation_tensor->flat<tf::tstring>()),
        output_dim(output_dim) {}

  tf::TTypes<float>::Matrix dense_predictions;
  tf::TTypes<tf::tstring>::Flat dense_col_representation;
  const int output_dim;
};

// Mapping between feature idx (the index used by simpleML to index features),
// and the column of these features in the input tensors.
//
// WARNING: this mapping should be aligned with the mapping created by
// :tf_op_py.
class GenericInferenceEngine;

class FeatureIndex {
 public:
  tf::Status Initialize(const std::vector<int>& input_features,
                        const dataset::proto::DataSpecification& data_spec) {
    numerical_features_.clear();
    boolean_features_.clear();
    categorical_int_features_.clear();
    categorical_set_int_features_.clear();

    for (const int feature_idx : input_features) {
      const auto& feature_spec = data_spec.columns(feature_idx);
      switch (feature_spec.type()) {
        case dataset::proto::ColumnType::NUMERICAL:
          numerical_features_.push_back(feature_idx);
          break;
        case dataset::proto::ColumnType::BOOLEAN:
          boolean_features_.push_back(feature_idx);
          break;
        case dataset::proto::ColumnType::CATEGORICAL:
          categorical_int_features_.push_back(feature_idx);
          break;
        case dataset::proto::ColumnType::CATEGORICAL_SET:
          categorical_set_int_features_.push_back(feature_idx);
          break;
        default:
          return tf::Status(
              tf::error::UNIMPLEMENTED,
              absl::Substitute(
                  "Non supported feature type \"$0\" for feature \"$1\".",
                  dataset::proto::ColumnType_Name(feature_spec.type()),
                  feature_spec.name()));
      }
    }
    return tf::Status::OK();
  }

  const std::vector<int>& numerical_features() const {
    return numerical_features_;
  }

  const std::vector<int>& boolean_features() const { return boolean_features_; }

  const std::vector<int>& categorical_int_features() const {
    return categorical_int_features_;
  }

  const std::vector<int>& categorical_set_int_features() const {
    return categorical_set_int_features_;
  }

 private:
  // For each feature type, a mapping between a column index (in the input
  // tensor) and a feature index. "numerical_features_[i]" is the feature index
  // of the "i-th" column for the "numerical_feature" tensor in "InputTensors".
  std::vector<int> numerical_features_;
  std::vector<int> boolean_features_;
  std::vector<int> categorical_int_features_;
  std::vector<int> categorical_set_int_features_;
};

// Extracts a categorical-set-int value (i.e. a set of ints) from the tensors
// into a std::vector<int> representation.
//
// By convention, a meeting value is represented as [-1].
//
// Args:
//   - inputs: All the input tensors.
//   - feature_index: Mapping between feature spec index and the internal
//     indexing used by "inputs".
//   - tensor_col_idx: Column, in "inputs", containing the feature.
//   - max_value: Maximum value of the items. Items above or equal to this value
//     will be considered out-of-vocabulary.
tf::Status ExtractCategoricalSetInt(const InputTensors& inputs,
                                    const FeatureIndex& feature_index,
                                    const int tensor_col_idx,
                                    const int max_value, const int example_idx,
                                    std::vector<int32_t>* values) {
  if (inputs.categorical_set_int_features_row_splits_dim_2(example_idx) !=
      example_idx * feature_index.categorical_set_int_features().size()) {
    return tf::Status(tf::error::INTERNAL,
                      "Unexpected features_row_splits_dim_2 size.");
  }

  const int d1_cell =
      example_idx * feature_index.categorical_set_int_features().size() +
      tensor_col_idx;
  if (d1_cell + 1 >=
      inputs.categorical_set_int_features_row_splits_dim_1.size()) {
    return tf::Status(tf::error::INTERNAL,
                      "Unexpected features_row_splits_dim_1 size.");
  }

  const int begin_idx =
      inputs.categorical_set_int_features_row_splits_dim_1(d1_cell);
  const int end_idx =
      inputs.categorical_set_int_features_row_splits_dim_1(d1_cell + 1);

  // Note: The items of the "example_idx"-th example and "col_idx"-th
  // categorical-set feature are "values[begin_idx:end_idx]".
  const int num_items = end_idx - begin_idx;
  values->resize(num_items);
  for (int item_idx = 0; item_idx < num_items; item_idx++) {
    auto value =
        inputs.categorical_set_int_features_values(item_idx + begin_idx);
    if (value < -1 || value >= max_value) {
      value = 0;
    }
    (*values)[item_idx] = value;
  }
  return tf::Status::OK();
}

// Wrapping around an inference engine able to run a model.
class AbstractInferenceEngine {
 public:
  virtual ~AbstractInferenceEngine() = default;

  // Cache data used by individual inference engines. Unlike the inference
  // engine
  // (i.e. "AbstractInferenceEngine"), the cache data can be modified during the
  // inference, and therefore, cannot be used by multiple threads in parallel.
  class AbstractCache {
   public:
    virtual ~AbstractCache() = default;
  };

  // Creates a cache: one per inference op instance.
  virtual StatusOr<std::unique_ptr<AbstractCache>> CreateCache() const = 0;

  // Run the inference of the model. The output tensors are already allocated.
  virtual tf::Status RunInference(const InputTensors& inputs,
                                  const FeatureIndex& feature_index,
                                  OutputTensors* outputs,
                                  AbstractCache* cache) const = 0;
};

// The generic engine uses the generic serving API
// (go/simple_ml/serving.md#c-generic-api). This solution is slow but has full
// model coverage.
class GenericInferenceEngine : public AbstractInferenceEngine {
 public:
  explicit GenericInferenceEngine(std::unique_ptr<model::AbstractModel> model)
      : model_(std::move(model)) {}

  class Cache : public AbstractCache {
   private:
    dataset::VerticalDataset dataset_;

    friend GenericInferenceEngine;
  };

  StatusOr<std::unique_ptr<AbstractCache>> CreateCache() const override {
    auto cache = absl::make_unique<GenericInferenceEngine::Cache>();
    cache->dataset_.set_data_spec(model_->data_spec());
    RETURN_IF_ERROR(cache->dataset_.CreateColumnsFromDataspec());
    return cache;
  }

  tf::Status RunInference(const InputTensors& inputs,
                          const FeatureIndex& feature_index,
                          OutputTensors* outputs,
                          AbstractCache* abstract_cache) const override {
    // Update the vertical dataset with the input tensors.
    LOG(INFO) << "-----GenericInferenceEngine";
    auto* cache = dynamic_cast<Cache*>(abstract_cache);
    if (cache == nullptr) {
      return tf::Status(tf::error::INTERNAL, "Unexpected cache type.");
    }
    TF_RETURN_IF_ERROR(SetVerticalDataset(inputs, feature_index, cache));

    // Run the model.
    model::proto::Prediction prediction;
    for (int example_idx = 0; example_idx < inputs.batch_size; example_idx++) {
      model_->Predict(cache->dataset_, example_idx, &prediction);

      // Copy the predictions to the output tensor.
      switch (model_->task()) {
        case Task::CLASSIFICATION: {
          const auto& pred = prediction.classification();
          // Note: "pred" contains a probability for each possible classes.
          // Because the label is categorical, the first label value (i.e. index
          // 0) is reserved for the Out-of-vocabulary value. As simpleML models
          // are not expected to output such value, we skip it (see the ".. - 1"
          // and ".. + 1" in the next part of the code).
          DCHECK_EQ(outputs->dense_predictions.dimension(1),
                    outputs->output_dim);
          DCHECK_EQ(outputs->dense_predictions.dimension(1),
                    pred.distribution().counts().size() - 1);
          for (int class_idx = 0; class_idx < outputs->output_dim;
               class_idx++) {
            const float probability =
                prediction.classification().distribution().counts(class_idx +
                                                                  1) /
                prediction.classification().distribution().sum();
            outputs->dense_predictions(example_idx, class_idx) =
                utils::clamp(probability, 0.f, 1.f);
          }
        } break;

        case Task::REGRESSION: {
          DCHECK_EQ(outputs->output_dim, 1);
          DCHECK_EQ(outputs->dense_predictions.dimension(1), 1);
          outputs->dense_predictions(example_idx, 0) =
              prediction.regression().value();
        } break;

        default:
          return tf::Status(tf::error::UNIMPLEMENTED,
                            absl::Substitute("Non supported task $0",
                                             Task_Name(model_->task())));
      }
    }

    return tf::Status::OK();
  }

 private:
  tf::Status SetVerticalDataset(const InputTensors& inputs,
                                const FeatureIndex& feature_index,
                                Cache* cache) const {
    cache->dataset_.set_nrow(inputs.batch_size);
    // Numerical features.
    for (int col_idx = 0; col_idx < feature_index.numerical_features().size();
         col_idx++) {
      const int feature_idx = feature_index.numerical_features()[col_idx];
      auto* col = cache->dataset_.MutableColumnWithCastOrNull<
          dataset::VerticalDataset::NumericalColumn>(feature_idx);
      if (col == nullptr) {
        return tf::Status(tf::error::INTERNAL, "Unexpected column type.");
      }
      col->Resize(inputs.batch_size);
      auto& dst = *col->mutable_values();
      for (int example_idx = 0; example_idx < inputs.batch_size;
           example_idx++) {
        // Missing represented as NaN.
        dst[example_idx] = inputs.numerical_features(example_idx, col_idx);
      }
    }

    // Boolean features.
    for (int col_idx = 0; col_idx < feature_index.boolean_features().size();
         col_idx++) {
      const int feature_idx = feature_index.boolean_features()[col_idx];
      auto* col = cache->dataset_.MutableColumnWithCastOrNull<
          dataset::VerticalDataset::BooleanColumn>(feature_idx);
      if (col == nullptr) {
        return tf::Status(tf::error::INTERNAL, "Unexpected column type.");
      }
      col->Resize(inputs.batch_size);
      auto& dst = *col->mutable_values();
      for (int example_idx = 0; example_idx < inputs.batch_size;
           example_idx++) {
        char bool_value;
        const float value = inputs.boolean_features(example_idx, col_idx);
        if (std::isnan(value)) {
          bool_value = dataset::VerticalDataset::BooleanColumn::kNaValue;
        } else if (value >= 0.5) {
          bool_value = dataset::VerticalDataset::BooleanColumn::kTrueValue;
        } else {
          bool_value = dataset::VerticalDataset::BooleanColumn::kFalseValue;
        }
        dst[example_idx] = bool_value;
      }
    }

    // Categorical int features.
    for (int col_idx = 0;
         col_idx < feature_index.categorical_int_features().size(); col_idx++) {
      const int feature_idx = feature_index.categorical_int_features()[col_idx];
      auto* col = cache->dataset_.MutableColumnWithCastOrNull<
          dataset::VerticalDataset::CategoricalColumn>(feature_idx);
      if (col == nullptr) {
        return tf::Status(tf::error::INTERNAL, "Unexpected column type.");
      }
      col->Resize(inputs.batch_size);
      const int max_value = cache->dataset_.data_spec()
                                .columns(feature_idx)
                                .categorical()
                                .number_of_unique_values();
      auto& dst = *col->mutable_values();
      for (int example_idx = 0; example_idx < inputs.batch_size;
           example_idx++) {
        auto value = inputs.categorical_int_features(example_idx, col_idx);
        if (value < -1 || value >= max_value) {
          value = 0;
        }
        dst[example_idx] = value;
      }
    }

    // Categorical set int features.
    //
    // Note: The categorical-set values are stored in a "two levels" ragged
    // tensor i.e. a ragged tensor inside of another one, shaped
    // "[batch_size, num_features, set_size]", where "set_size" is the only
    // ragged dimension.
    // In other words, "value[i,j,k]" is the "k-th" item, of the "j-th" feature,
    // of the "i-th" example.
    std::vector<int> tmp_values;
    for (int col_idx = 0;
         col_idx < feature_index.categorical_set_int_features().size();
         col_idx++) {
      const int feature_idx =
          feature_index.categorical_set_int_features()[col_idx];
      auto* col = cache->dataset_.MutableColumnWithCastOrNull<
          dataset::VerticalDataset::CategoricalSetColumn>(feature_idx);
      if (col == nullptr) {
        return tf::Status(tf::error::INTERNAL, "Unexpected column type.");
      }
      col->Resize(inputs.batch_size);

      const int max_value = cache->dataset_.data_spec()
                                .columns(feature_idx)
                                .categorical()
                                .number_of_unique_values();

      for (int example_idx = 0; example_idx < inputs.batch_size;
           example_idx++) {
        const auto status =
            ExtractCategoricalSetInt(inputs, feature_index, col_idx, max_value,
                                     example_idx, &tmp_values);
        if (!status.ok()) {
          return status;
        }

        if (!tmp_values.empty() && tmp_values.front() < 0) {
          col->SetNA(example_idx);
        } else {
          col->SetIter(example_idx, tmp_values.begin(), tmp_values.end());
        }
      }
    }

    return tf::Status::OK();
  }

  std::unique_ptr<model::AbstractModel> model_;
};

// The semi-fast generic engine uses the generic serving API
// (go/simple_ml/serving.md#c-generic-api). When available, this solution is
// significantly (e.g. up to 20x) faster than "GenericInferenceEngine".
class SemiFastGenericInferenceEngine : public AbstractInferenceEngine {
 public:
  static StatusOr<std::unique_ptr<SemiFastGenericInferenceEngine>> Create(
      std::unique_ptr<serving::FastEngine> engine,
      const model::AbstractModel& model, const FeatureIndex& feature_index) {
    auto engine_wrapper = absl::WrapUnique(
        new SemiFastGenericInferenceEngine(std::move(engine), model));
    RETURN_IF_ERROR(engine_wrapper->Initialize(feature_index));
    return engine_wrapper;
  }

  class Cache : public AbstractCache {
   private:

    // Cache of pre-allocated predictions.
    std::vector<float> predictions_;

    // Cache of pre-allocated examples.
    std::unique_ptr<serving::AbstractExampleSet> examples_;

    // Number of examples allocated in "examples_".
    int num_examples_in_cache_ = -1;

    friend SemiFastGenericInferenceEngine;
  };

  StatusOr<std::unique_ptr<AbstractCache>> CreateCache() const override {
    auto cache = absl::make_unique<SemiFastGenericInferenceEngine::Cache>();
    cache->examples_ = engine_->AllocateExamples(1);
    cache->num_examples_in_cache_ = 1;
    return cache;
  }

  tf::Status RunInference(const InputTensors& inputs,
                          const FeatureIndex& feature_index,
                          OutputTensors* outputs,
                          AbstractCache* abstract_cache) const override {
    // Update the vertical dataset with the input tensors.
    LOG(INFO) << "-----Before Cache";
    auto* cache = dynamic_cast<Cache*>(abstract_cache);
    if (cache == nullptr) {
      return tf::Status(tf::error::INTERNAL, "Unexpected cache type.");
    }
    LOG(INFO) << "-----step1";
    // Allocate a cache of examples.
    if (cache->num_examples_in_cache_ < inputs.batch_size) {
      cache->examples_ = engine_->AllocateExamples(inputs.batch_size);
      cache->num_examples_in_cache_ = inputs.batch_size;
    }
    LOG(INFO) << "-----step2";

    // Copy the example data in the format expected by the engine.
    TF_RETURN_IF_ERROR(
        SetExamples(inputs, feature_index, cache->examples_.get()));
LOG(INFO) << "-----step3";
    // Run the model.
    engine_->Predict(*cache->examples_, inputs.batch_size,
                     &cache->predictions_);
LOG(INFO) << "-----step4";
    // Export the predictions.
    if (decompact_probability_) {
      DCHECK_EQ(outputs->output_dim, 2);
      DCHECK_EQ(engine_->NumPredictionDimension(), 1);
      for (int example_idx = 0; example_idx < inputs.batch_size;
           example_idx++) {
	LOG(INFO) << "-----step5";
        const float proba =
            utils::clamp(cache->predictions_[example_idx], 0.f, 1.f);
        outputs->dense_predictions(example_idx, 0) = 1.f - proba;
        outputs->dense_predictions(example_idx, 1) = proba;
      }

    } else {
	    LOG(INFO) << "-----step6";
      DCHECK_EQ(outputs->output_dim, engine_->NumPredictionDimension());
      for (int example_idx = 0; example_idx < inputs.batch_size;
           example_idx++) {
        for (int class_idx = 0; class_idx < outputs->output_dim; class_idx++) {
          const float value =
              cache
                  ->predictions_[example_idx * outputs->output_dim + class_idx];
          outputs->dense_predictions(example_idx, class_idx) = value;
        }
      }
    }
    return tf::Status::OK();
  }

 private:
  explicit SemiFastGenericInferenceEngine(
      std::unique_ptr<serving::FastEngine> engine,
      const model::AbstractModel& model)
      : engine_(std::move(engine)) {
    decompact_probability_ = (model.task() == Task::CLASSIFICATION) &&
                             engine_->NumPredictionDimension() == 1;
  }

  absl::Status Initialize(FeatureIndex feature_index) {
    // Register numerical features.
    LOG(INFO) << "-----SemiFast initialise";
    for (int tensor_col = 0;
         tensor_col < feature_index.numerical_features().size(); tensor_col++) {
      const auto dataspec_idx = feature_index.numerical_features()[tensor_col];
      const auto feature_id =
          engine_->features().GetNumericalFeatureId(dataspec_idx);
      if (!feature_id.ok()) {
        // The feature is not used by the model.
        continue;
      }
      numerical_features_.push_back({/*.tensor_col =*/tensor_col,
                                     /*.dataspec_idx = */ dataspec_idx,
                                     /*.example_set_id =*/feature_id.value()});
    }

    // Register categorical int features.
    for (int tensor_col = 0;
         tensor_col < feature_index.categorical_int_features().size();
         tensor_col++) {
      const auto dataspec_idx =
          feature_index.categorical_int_features()[tensor_col];
      const auto feature_id =
          engine_->features().GetCategoricalFeatureId(dataspec_idx);
      if (!feature_id.ok()) {
        // The feature is not used by the model.
        continue;
      }
      categorical_int_features_.push_back(
          {/*.tensor_col =*/tensor_col,
           /*.dataspec_idx =*/dataspec_idx,
           /*.example_set_id =*/feature_id.value()});
    }

    // Register categorical set int features.
    for (int tensor_col = 0;
         tensor_col < feature_index.categorical_set_int_features().size();
         tensor_col++) {
      const auto dataspec_idx =
          feature_index.categorical_set_int_features()[tensor_col];
      const auto feature_id =
          engine_->features().GetCategoricalSetFeatureId(dataspec_idx);
      if (!feature_id.ok()) {
        // The feature is not used by the model.
        continue;
      }
      categorical_set_int_features_.push_back(
          {/*.tensor_col =*/tensor_col,
           /*.dataspec_idx =*/dataspec_idx,
           /*.example_set_id =*/feature_id.value()});
    }

    if (!feature_index.boolean_features().empty()) {
      return absl::InvalidArgumentError("Boolean features not supported.");
    }

    return absl::OkStatus();
  }

  // Copy the content of "inputs" into "examples".
  // "examples" is allocated with at least "inputs.batch_size" examples.
  tf::Status SetExamples(const InputTensors& inputs,
                         const FeatureIndex& feature_index,
                         serving::AbstractExampleSet* examples) const {
    const auto& features = engine_->features();
    examples->FillMissing(engine_->features());

    // Numerical features.
    for (const auto& feature : numerical_features_) {
      for (int example_idx = 0; example_idx < inputs.batch_size;
           example_idx++) {
        const float value =
            inputs.numerical_features(example_idx, feature.tensor_col);
        if (!std::isnan(value)) {
          examples->SetNumerical(example_idx, feature.example_set_id, value,
                                 features);
        } else {
          examples->SetMissingNumerical(example_idx, feature.example_set_id,
                                        features);
        }
      }
    }

    // Categorical int features.
    for (const auto& feature : categorical_int_features_) {
      const int max_value = engine_->features()
                                .data_spec()
                                .columns(feature.dataspec_idx)
                                .categorical()
                                .number_of_unique_values();
      for (int example_idx = 0; example_idx < inputs.batch_size;
           example_idx++) {
        int value =
            inputs.categorical_int_features(example_idx, feature.tensor_col);
        if (value == -1) {
          examples->SetMissingCategorical(example_idx, feature.example_set_id,
                                          features);
        } else {
          if (value < -1 || value >= max_value) {
            value = 0;
          }
          examples->SetCategorical(example_idx, feature.example_set_id, value,
                                   features);
        }
      }
    }

    // Categorical set int features.
    std::vector<int> tmp_values;
    for (const auto& feature : categorical_set_int_features_) {
      const int max_value = engine_->features()
                                .data_spec()
                                .columns(feature.dataspec_idx)
                                .categorical()
                                .number_of_unique_values();
      for (int example_idx = 0; example_idx < inputs.batch_size;
           example_idx++) {
        const auto status =
            ExtractCategoricalSetInt(inputs, feature_index, feature.tensor_col,
                                     max_value, example_idx, &tmp_values);
        if (!status.ok()) {
          return status;
        }

        if (!tmp_values.empty() && tmp_values.front() < 0) {
          examples->SetMissingCategoricalSet(example_idx,
                                             feature.example_set_id, features);
        } else {
          examples->SetCategoricalSet(example_idx, feature.example_set_id,
                                      tmp_values, features);
        }
      }
    }

    return tf::Status::OK();
  }

  // Inference engine. Contains the model data.
  std::unique_ptr<serving::FastEngine> engine_;

  // If true, the model is output the probability "p" of the positive class,
  // and the inference op is expected to return [1-p,p].
  bool decompact_probability_;

  // Different ids of the features.
  template <typename FeaturesDefinitionId>
  struct FeatureId {
    const int tensor_col;
    const int dataspec_idx;
    const FeaturesDefinitionId example_set_id;
  };

  // Features used by the model.
  std::vector<FeatureId<serving::FeaturesDefinition::NumericalFeatureId>>
      numerical_features_;
  std::vector<FeatureId<serving::FeaturesDefinition::CategoricalFeatureId>>
      categorical_int_features_;
  std::vector<FeatureId<serving::FeaturesDefinition::CategoricalSetFeatureId>>
      categorical_set_int_features_;
};

// TF resource containing the Yggdrasil model in memory.
class YggdrasilModelResource : public tf::ResourceBase {
 public:
  std::string DebugString() const override { return "YggdrasilModelResource"; }

  // Loads the model from disk.
  tf::Status LoadModelFromDisk(const absl::string_view model_path) {
    LOG(INFO) << "------YggdrasilModelResource";
    std::unique_ptr<model::AbstractModel> model;
    TF_RETURN_IF_ERROR(utils::FromUtilStatus(LoadModel(model_path, &model)));
    task_ = model->task();
    TF_RETURN_IF_ERROR(
        feature_index_.Initialize(model->input_features(), model->data_spec()));
    TF_RETURN_IF_ERROR(ComputeDenseColRepresentation(model.get()));

    // WARNING: After this function, the "model" might not be available anymore.
    TF_RETURN_IF_ERROR(CreateInferenceEngine(std::move(model)));
    return tf::Status::OK();
  }

  const AbstractInferenceEngine* engine() const {
    return inference_engine_.get();
  }

  const FeatureIndex& feature_index() const { return feature_index_; }

  Task task() const { return task_; }

  const std::vector<tf::tstring>& dense_col_representation() const {
    return dense_col_representation_;
  }

 private:
  // Creates an inference engine compatible with the model. The inference engine
  // can take ownership of the abstract model data.
  tf::Status CreateInferenceEngine(
      std::unique_ptr<model::AbstractModel> model) {
    auto semi_fast_engine = model->BuildFastEngine();
    if (semi_fast_engine.ok()) {
      // Semi-fast generic engine.
      auto inference_engine_or_status = SemiFastGenericInferenceEngine::Create(
          std::move(semi_fast_engine.value()), *model, feature_index());
      TF_RETURN_IF_ERROR(
          utils::FromUtilStatus(inference_engine_or_status.status()));
      inference_engine_ = std::move(inference_engine_or_status.value());
      LOG(INFO) << "Use fast generic engine test";
      return tf::Status::OK();
    } else {
      // Slow generic engine.
      LOG(INFO) << "Use slow generic engine";
      inference_engine_ =
          absl::make_unique<GenericInferenceEngine>(std::move(model));
      return tf::Status::OK();
    }
  }

  // Pre-compute the values returned in the "dense_col_representation" output of
  // the inference OPs.
  tf::Status ComputeDenseColRepresentation(
      const model::AbstractModel* const model) {
    if (task_ == Task::CLASSIFICATION) {
      const auto& label_spec =
          model->data_spec().columns(model->label_col_idx());
      // Note: We don't report the "OOV" class value.
      const int num_classes =
          label_spec.categorical().number_of_unique_values() - 1;
      dense_col_representation_.resize(num_classes);
      for (int class_idx = 0; class_idx < num_classes; class_idx++) {
        dense_col_representation_[class_idx] =
            dataset::CategoricalIdxToRepresentation(label_spec, class_idx + 1);
      }
    } else {
      dense_col_representation_.resize(1);
    }
    return tf::Status::OK();
  }

  // The engine responsible to run the model.
  std::unique_ptr<AbstractInferenceEngine> inference_engine_;

  // Index of the input features and the input tensors.
  FeatureIndex feature_index_;

  // Task solved by the model.
  Task task_;

  // Pre-computed values to send in the "dense_col_representation" output
  // tensor.
  std::vector<tf::tstring> dense_col_representation_;
};

tf::Status GetModel(OpKernelContext* ctx,
                    YggdrasilModelResource** model_resource) {
  LOG(INFO) << "----- Get model";
  const Tensor* handle_tensor;
  TF_RETURN_IF_ERROR(ctx->input(kInputModelHandle, &handle_tensor));
  const tf::ResourceHandle& handle =
      handle_tensor->scalar<tf::ResourceHandle>()();
  return LookupResource(ctx, handle, model_resource);
}

// Get the model path at execution time.
tf::Status GetModelPath(OpKernelContext* ctx, std::string* model_path) {
  const Tensor* model_path_tensor;
  TF_RETURN_IF_ERROR(ctx->input(kInputPath, &model_path_tensor));

  const auto model_paths = model_path_tensor->flat<tf::tstring>();
  if (model_paths.size() != 1) {
    return tf::errors::InvalidArgument(absl::Substitute(
        "The \"$0\" attribute is expected to contains exactly one entry.",
        kInputPath));
  }
  *model_path = model_paths(0);
  return tf::Status::OK();
}

// Load the model from disk into a resource specified as resource name.
class SimpleMLLoadModelFromPath : public OpKernel {
 public:
  explicit SimpleMLLoadModelFromPath(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr(kAttributeModelIdentifier, &model_identifier_));
  }

  void Compute(OpKernelContext* ctx) override {
    {
      // Skip loading the model if a model with the target identifier is already
      // loaded in the session's resource set.
      YggdrasilModelResource* maybe_resource;
      if (ctx->resource_manager()
              ->Lookup(kModelContainer, model_identifier_, &maybe_resource)
              .ok()) {
        maybe_resource->Unref();
        LOG(WARNING) << "Model " << model_identifier_ << " already loaded";
        return;
      }
    }

    std::string model_path;
    OP_REQUIRES_OK(ctx, GetModelPath(ctx, &model_path));
    LOG(INFO) << "-----SimpleMLLoadModelFromPath";
    auto* model_container = new YggdrasilModelResource();
    const auto load_status = model_container->LoadModelFromDisk(model_path);
    if (!load_status.ok()) {
      model_container->Unref();  // Call delete on "model_container".
      OP_REQUIRES_OK(ctx, load_status);
    }

    // Note: "Create" takes ownership of "model_container".
    OP_REQUIRES_OK(
        ctx, ctx->resource_manager()->Create(kModelContainer, model_identifier_,
                                             model_container));
  }

 private:
  // Identifier of the model. Copy of the "model_identifier" attribute.
  std::string model_identifier_;
};

REGISTER_KERNEL_BUILDER(
    Name("SimpleMLLoadModelFromPath").Device(tf::DEVICE_CPU),
    SimpleMLLoadModelFromPath);

// Load the model from disk into a resource specified as a resource handle.
class SimpleMLLoadModelFromPathWithHandle : public OpKernel {
 public:
  explicit SimpleMLLoadModelFromPathWithHandle(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    std::string model_path;
    OP_REQUIRES_OK(ctx, GetModelPath(ctx, &model_path));

    YggdrasilModelResource* model_container;
    OP_REQUIRES_OK(ctx, GetModel(ctx, &model_container));
    tf::core::ScopedUnref unref_me(model_container);

    LOG(INFO) << "Loading model from path";
    OP_REQUIRES_OK(ctx, model_container->LoadModelFromDisk(model_path));
  }
};

REGISTER_KERNEL_BUILDER(
    Name("SimpleMLLoadModelFromPathWithHandle").Device(tf::DEVICE_CPU),
    SimpleMLLoadModelFromPathWithHandle);

// Runs the inference of the model on packed tensors.
// The input model is specified as a resource string name.
class SimpleMLInferenceOp : public OpKernel {
 public:
  explicit SimpleMLInferenceOp(OpKernelConstruction* ctx,
                               const bool read_model_identifier = true)
      : OpKernel(ctx) {
    if (read_model_identifier) {
      OP_REQUIRES_OK(
          ctx, ctx->GetAttr(kAttributeModelIdentifier, &model_identifier_));
    }

    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr(kAttributeDenseOutputDim, &dense_output_dim_));
  }

  ~SimpleMLInferenceOp() override {
    if (model_container_) {
      model_container_->Unref();
      model_container_ = nullptr;
    }
  }
  void Compute(OpKernelContext* ctx) override {
    // Make sure the model is available.
    LOG(INFO) << "-----InferneceOP Compute";
    if (!model_container_) {
      OP_REQUIRES_OK(ctx, LinkModelResource(ctx));
    }

    // Collect the input signals.
    LOG(INFO) << "CALL 971" ; 
    tf::Status io_status;
    const auto input_tensors =
        LinkInputTensors(ctx, model_container_->feature_index(), &io_status);
    OP_REQUIRES_OK(ctx, io_status);

    // Allocate the output predictions memory.
    LOG(INFO) << "CALL 977 prediction memory" ;
    auto output_tensors =
        LinkOutputTensors(ctx, input_tensors.batch_size, &io_status);
    OP_REQUIRES_OK(ctx, io_status);

    // Set the output representation.
    const auto& reps = model_container_->dense_col_representation();
    for (int rep_idx = 0; rep_idx < reps.size(); rep_idx++) {
      output_tensors.dense_col_representation(rep_idx) = reps[rep_idx];
    }

    // Run the model.
    LOG(INFO) << "Call line 988" ;
    tf::mutex_lock lock_engine_mutex(engine_mutex_);
    OP_REQUIRES_OK(ctx, model_container_->engine()->RunInference(
                            input_tensors, model_container_->feature_index(),
                            &output_tensors, engine_cache_.get()));
  }

 protected:
  // Creates a cache for the engine in model_container_.
  tf::Status CreateEngineCache() {
    LOG(INFO) << "-----Create Cache";
    auto cache_or_status = model_container_->engine()->CreateCache();
    TF_RETURN_IF_ERROR(utils::FromUtilStatus(cache_or_status.status()));
    engine_cache_ = std::move(cache_or_status).value();
    return tf::Status::OK();
  }

  // Links the model and set "model_container_" accordingly.
  virtual tf::Status LinkModelResource(OpKernelContext* ctx) {
    const auto lookup_status = ctx->resource_manager()->Lookup(
        kModelContainer, model_identifier_, &model_container_);
    if (!lookup_status.ok()) {
      return tf::Status(
          lookup_status.code(),
          absl::StrCat(lookup_status.error_message(),
                       ". This error caused the simpleML model not to be "
                       "available for inference. This error is likely due to "
                       "the \"LoadModel*\" not having been run before."));
    }

    return CreateEngineCache();
  }

  // Computes the batch size from the input feature tensors. Returns an error if
  // the size of the input feature tensors is inconsistent.
  //
  // All the input feature are expected to have a first dimension of zero
  // (unused) or equal to the batch size.
  tf::Status ComputeBatchSize(const InputTensors& input_tensors,
                              int* batch_size) {
    LOG(INFO) << "-----Computing Batch Size";
    int max_size = 0;
    for (const int size :
         {input_tensors.numerical_features.dimension(0),
          input_tensors.boolean_features.dimension(0),
          input_tensors.categorical_int_features.dimension(0),
          input_tensors.categorical_set_int_features_row_splits_dim_2.dimension(
              0) -
              1}) {
      if (size > 0) {
        if (max_size == 0) {
          max_size = size;
        } else if (max_size != size) {
          return tf::Status(
              tf::error::INVALID_ARGUMENT,
              absl::StrCat(
                  "The batch size of the input features are inconsistent: ",
                  max_size, " vs ", size, "."));
        }
      }
    }
    *batch_size = max_size;
    return tf::Status::OK();
  }

  // Gets the c++ references on all the input tensor values of the inference op.
  // In other words, get the input tensor and cast them to the expected type.
  InputTensors LinkInputTensors(OpKernelContext* ctx,
                                const FeatureIndex& feature_index,
                                tf::Status* status) {
    const Tensor* numerical_features_tensor = nullptr;
    const Tensor* boolean_features_tensor = nullptr;
    const Tensor* categorical_int_features_tensor = nullptr;
    const Tensor* categorical_set_int_features_values_tensor = nullptr;
    const Tensor* categorical_set_int_features_row_splits_dim_1_tensor =
        nullptr;
    const Tensor* categorical_set_int_features_row_splits_dim_2_tensor =
        nullptr;

    auto build_return = [&]() -> InputTensors {
      return {numerical_features_tensor,
              boolean_features_tensor,
              categorical_int_features_tensor,
              categorical_set_int_features_values_tensor,
              categorical_set_int_features_row_splits_dim_1_tensor,
              categorical_set_int_features_row_splits_dim_2_tensor};
    };

    for (const auto& tensor_def :
         std::vector<std::pair<const char*, const Tensor**>>{
             {kInputNumericalFeatures, &numerical_features_tensor},

             {kInputBooleanFeatures, &boolean_features_tensor},

             {kInputCategoricalIntFeatures, &categorical_int_features_tensor},

             {kInputCategoricalSetIntFeaturesValues,
              &categorical_set_int_features_values_tensor},

             {kInputCategoricalSetIntFeaturesRowSplitsDim1,
              &categorical_set_int_features_row_splits_dim_1_tensor},

             {kInputCategoricalSetIntFeaturesRowSplitsDim2,
              &categorical_set_int_features_row_splits_dim_2_tensor},
         }) {
      *status = ctx->input(tensor_def.first, tensor_def.second);
      if (!status->ok()) {
        return build_return();
      }
    }

    // Set the batch size from the tensors.
    auto tensors = build_return();
    *status = ComputeBatchSize(tensors, &tensors.batch_size);

    // Check number of dimensions of inputs.
    // Note: The user cannot impact those if using the wrapper.
    if (tensors.numerical_features.dimension(1) !=
        feature_index.numerical_features().size()) {
      *status = tf::Status(tf::error::INVALID_ARGUMENT,
                           "Unexpected dimension of numerical_features bank.");
      return build_return();
    }

    if (tensors.boolean_features.dimension(1) !=
        feature_index.boolean_features().size()) {
      *status = tf::Status(tf::error::INVALID_ARGUMENT,
                           "Unexpected dimension of boolean_features bank.");
      return build_return();
    }

    if (tensors.categorical_int_features.dimension(1) !=
        feature_index.categorical_int_features().size()) {
      *status =
          tf::Status(tf::error::INVALID_ARGUMENT,
                     "Unexpected dimension of categorical_int_features bank.");
      return build_return();
    }

    return tensors;
  }

  // Allocates and gets the c++ references to all the output tensor values of
  // the inference op.
  OutputTensors LinkOutputTensors(OpKernelContext* ctx, const int batch_size,
                                  tf::Status* status) {
    Tensor* dense_predictions_tensor = nullptr;
    Tensor* dense_col_representation_tensor = nullptr;

    *status = ctx->allocate_output(kOutputDensePredictions,
                                   TensorShape({batch_size, dense_output_dim_}),
                                   &dense_predictions_tensor);
    if (!status->ok()) {
      return {dense_predictions_tensor, dense_col_representation_tensor,
              dense_output_dim_};  // Eigen maps cannot be empty.
    }
    *status = ctx->allocate_output(kOutputDenseColRepresentation,
                                   TensorShape({dense_output_dim_}),
                                   &dense_col_representation_tensor);
    if (!status->ok()) {
      return {dense_predictions_tensor, dense_col_representation_tensor,
              dense_output_dim_};  // Eigen maps cannot be empty.
    }
    return {dense_predictions_tensor, dense_col_representation_tensor,
            dense_output_dim_};
  }

  // Identifier of the model. Copy of the "model_identifier" attribute.
  std::string model_identifier_;

  // Possibly shared, non-owning pointer to the model. Is set during the first
  // call to the OP.
  YggdrasilModelResource* model_container_ = nullptr;

  // Cache data to re-use in between inference calls.
  std::unique_ptr<AbstractInferenceEngine::AbstractCache> engine_cache_;

  // Protects calls to the engine inference.
  tensorflow::mutex engine_mutex_;

  // Copy of the attributes of the same name.
  int dense_output_dim_;
};

REGISTER_KERNEL_BUILDER(Name("SimpleMLInferenceOp").Device(tf::DEVICE_CPU),
                        SimpleMLInferenceOp);

class SimpleMLInferenceOpWithHandle : public SimpleMLInferenceOp {
 public:
  explicit SimpleMLInferenceOpWithHandle(OpKernelConstruction* ctx)
      : SimpleMLInferenceOp(ctx, /*read_model_identifier=*/false) {}

  ~SimpleMLInferenceOpWithHandle() override {}

  tf::Status LinkModelResource(OpKernelContext* ctx) override {
    TF_RETURN_IF_ERROR(GetModel(ctx, &model_container_));
    LOG(INFO) << "------SIMPLEMLInferenceOpWithHandle";
    return CreateEngineCache();
  }
};

REGISTER_KERNEL_BUILDER(
    Name("SimpleMLInferenceOpWithHandle").Device(tf::DEVICE_CPU),
    SimpleMLInferenceOpWithHandle);

// Implementation inspired from "LookupTableOp" in:
// google3/third_party/tensorflow/core/kernels/lookup_table_op.h
class SimpleMLCreateModelResource : public OpKernel {
 public:
  explicit SimpleMLCreateModelResource(OpKernelConstruction* ctx)
      : OpKernel(ctx), model_handle_set_(false) {
    // The model_handle_ object will outlive the constructor function.
    // It will live until the resource is freed manually or by
    // explicit ~Tensor() destructor.
    OP_REQUIRES_OK(
        ctx, ctx->allocate_temp(tensorflow::DT_RESOURCE,
                                tensorflow::TensorShape({}), &model_handle_));
  }

  ~SimpleMLCreateModelResource() override {
    if (model_handle_set_ && cinfo_.resource_is_private_to_kernel()) {
      if (!cinfo_.resource_manager()
               ->template Delete<YggdrasilModelResource>(cinfo_.container(),
                                                         cinfo_.name())
               .ok()) {
        // Do nothing; the resource can have been deleted by session resets.
      }
    }
  }
  void Compute(OpKernelContext* ctx) override {
    tf::mutex_lock l(mu_);
    if (!model_handle_set_) {
      OP_REQUIRES_OK(ctx, cinfo_.Init(ctx->resource_manager(), def(), false));
    }
    LOG(INFO) << "-----SimpleMLCreateModelResource";
    auto creator =
        [ctx, this](YggdrasilModelResource** ret)
            TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
              auto* container = new YggdrasilModelResource();
              if (!ctx->status().ok()) {
                container->Unref();
                return ctx->status();
              }
              if (ctx->track_allocations()) {
                ctx->record_persistent_memory_allocation(
                    container->MemoryUsed() + model_handle_.AllocatedBytes());
              }
              *ret = container;
              return tf::Status::OK();
            };

    YggdrasilModelResource* model = nullptr;
    OP_REQUIRES_OK(ctx,
                   cinfo_.resource_manager()
                       ->template LookupOrCreate<YggdrasilModelResource>(
                           cinfo_.container(), cinfo_.name(), &model, creator));
    tf::core::ScopedUnref unref_me(model);

    if (!model_handle_set_) {
      auto h = model_handle_.template scalar<tf::ResourceHandle>();
      h() = tf::MakeResourceHandle<YggdrasilModelResource>(
          ctx, cinfo_.container(), cinfo_.name());
    }
    ctx->set_output(0, model_handle_);
    model_handle_set_ = true;
  }

 private:
  tf::mutex mu_;
  tf::Tensor model_handle_ TF_GUARDED_BY(mu_);
  bool model_handle_set_ TF_GUARDED_BY(mu_);
  tf::ContainerInfo cinfo_;

  TF_DISALLOW_COPY_AND_ASSIGN(SimpleMLCreateModelResource);
};

REGISTER_KERNEL_BUILDER(
    Name("SimpleMLCreateModelResource").Device(tf::DEVICE_CPU),
    SimpleMLCreateModelResource);

}  // namespace ops
}  // namespace tensorflow_decision_forests
