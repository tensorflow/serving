# SavedModel Warmup

## Introduction

The TensorFlow runtime has components that are lazily initialized,
which can cause high latency for the first request/s sent to a model after it is
loaded. This latency can be several orders of magnitude higher than that of a
single inference request.

To reduce the impact of lazy initialization on request latency, it's possible to
trigger the initialization of the sub-systems and components at model load time
by providing a sample set of inference requests along with the SavedModel. This
process is known as "warming up" the model.

## Usage

SavedModel Warmup is supported for Regress, Classify, MultiInference and
Predict. To trigger warmup of the model at load time, attach a warmup data file
under the assets.extra subfolder of the SavedModel directory.

Requirements for model warmup to work correctly:

*   Warmup file name: 'tf_serving_warmup_requests'
*   File location: assets.extra/
*   File format:
    [TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord#tfrecords_format_details)
    with each record as a
    [PredictionLog](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/apis/prediction_log.proto#:~:text=message-,PredictionLog,-%7B).
*   Number of warmup records <= 1000.
*   The warmup data must be representative of the inference requests used at
    serving.

## Warm-up data generation

Warmup data can be added in two ways:

*   By directly populating the warmup requests into your exported Saved Model.
    This could be done by creating a script reading a list of sample
    inference requests, converting each request into
    [PredictionLog](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/apis/prediction_log.proto#:~:text=message-,PredictionLog,-%7B)
    (if it's originally in a different format) and using
    [TFRecordWriter](https://www.tensorflow.org/api_docs/python/tf/io/TFRecordWriter)
    to write the PredictionLog entries into
    `YourSavedModel/assets.extra/tf_serving_warmup_requests`.
*   By using TFX Infra Validator
    [option to export a Saved Model with warmup](https://tensorflow.github.io/tfx/guide/infra_validator/#producing-a-savedmodel-with-warmup).
    With this option the TFX Infa Validator will populate
    `YourSavedModel/assets.extra/tf_serving_warmup_requests` based on the
    validation requests provided via
    [RequestSpec](https://tensorflow.github.io/tfx/guide/infra_validator/#requestspec).


