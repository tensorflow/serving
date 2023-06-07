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

Example code snippet producing warmup data:

```python
import tensorflow as tf
from tensorflow_serving.apis import classification_pb2
from tensorflow_serving.apis import inference_pb2
from tensorflow_serving.apis import model_pb2
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_log_pb2
from tensorflow_serving.apis import regression_pb2

def main():
    with tf.io.TFRecordWriter("tf_serving_warmup_requests") as writer:
        # replace <request> with one of:
        # predict_pb2.PredictRequest(..)
        # classification_pb2.ClassificationRequest(..)
        # regression_pb2.RegressionRequest(..)
        # inference_pb2.MultiInferenceRequest(..)
        log = prediction_log_pb2.PredictionLog(
            predict_log=prediction_log_pb2.PredictLog(request=<request>))
        writer.write(log.SerializeToString())

if __name__ == "__main__":
    main()
```
