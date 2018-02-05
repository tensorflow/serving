
#!/usr/bin/env python2.7

"""A client that talks to tensorflow_model_server loaded with twitter sentiment data.

    #Author: Rabindra Nath Nandi
"""

from __future__ import print_function

import sys
import threading

# This is a placeholder for a Google-internal import.

from grpc.beta import implementations
import numpy
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
#from tensorflow_serving.example import mnist_input_data
from data_helpers import load_data
from data_helpers import batch_iter
import numpy as np
tf.app.flags.DEFINE_integer('concurrency', 1,
                            'maximum number of concurrent inference requests')
tf.app.flags.DEFINE_integer('num_tests', 100, 'Number of test images')
tf.app.flags.DEFINE_string('server', '', 'PredictionService host:port')
tf.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory. ')
tf.flags.DEFINE_integer('test_data_ratio', 10,
                        'Percentual of the dataset to be used for validation '
                        '(default: 10)')
FLAGS = tf.app.flags.FLAGS
tf.flags.DEFINE_integer('batch_size', 100, 'Batch Size (default: 100)')


class _ResultCounter(object):
  """Counter for the prediction results."""

  def __init__(self, num_tests, concurrency):
    self._num_tests = num_tests
    self._concurrency = concurrency
    self._error = 0
    self._done = 0
    self._active = 0
    self._condition = threading.Condition()

  def inc_error(self):
    with self._condition:
      self._error += 1

  def inc_done(self):
    with self._condition:
      self._done += 1
      self._condition.notify()

  def dec_active(self):
    with self._condition:
      self._active -= 1
      self._condition.notify()

  def get_error_rate(self):
    with self._condition:
      while self._done != self._num_tests:
        self._condition.wait()
      return self._error / float(self._num_tests)

  def throttle(self):
    with self._condition:
      while self._active == self._concurrency:
        self._condition.wait()
      self._active += 1


def _create_rpc_callback(label, result_counter):
  """Creates RPC callback function.

  Args:
    label: The correct label for the predicted example.
    result_counter: Counter for the prediction result.
  Returns:
    The callback function.
  """
  def _callback(result_future):
    """Callback function.

    Calculates the statistics for the prediction result.

    Args:
      result_future: Result future of the RPC.
    """
    exception = result_future.exception()
    if exception:
      result_counter.inc_error()
      print(exception)
    else:
      sys.stdout.write('.')
      sys.stdout.flush()
      response = numpy.array(
          result_future.result().outputs['scores'].float_val)
      prediction = numpy.argmax(response)



      print('label->argmax: ')
      print(np.argmax(label))
      print('prediction: ')
      print(prediction)
      print('label: ')
      print(label)
      if np.argmax(label) != prediction:
        result_counter.inc_error()
    result_counter.inc_done()
    result_counter.dec_active()
  return _callback


def do_inference(hostport, work_dir, concurrency, num_tests):
  """Tests PredictionService with concurrent requests.

  Args:
    hostport: Host:port address of the PredictionService.
    work_dir: The full path of working directory for test data set.
    concurrency: Maximum number of concurrent requests.
    num_tests: Number of test images to use.

  Returns:
    The classification error rate.

  Raises:
    IOError: An error occurred processing test data set.
  """
  x, y, vocabulary, vocabulary_inv = load_data(1)
  # Randomly shuffle datas
  np.random.seed(123)
  shuffle_indices = np.random.permutation(np.arange(len(y)))
  x_shuffled = x[shuffle_indices]
  y_shuffled = y[shuffle_indices]

# Split train/test set
  text_percent = FLAGS.test_data_ratio / 100.0
  test_index = int(len(x) * text_percent)
  x_train, x_test = x_shuffled[:-test_index], x_shuffled[-test_index:]
  y_train, y_test = y_shuffled[:-test_index], y_shuffled[-test_index:]
 # batches = batch_iter(zip(x_train, y_train), FLAGS.batch_size, FLAGS.epochs)
  test_batches = list(batch_iter(zip(x_test, y_test), FLAGS.batch_size, 1))
  host, port = hostport.split(':')
  channel = implementations.insecure_channel(host, int(port))
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
  result_counter = _ResultCounter(num_tests, concurrency)


  print("Testing start................: batch size: "+str(len(test_batches)))
  for batch in test_batches:
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'twitter-sentiment'
    request.model_spec.signature_name = 'predict_text'
    test_data, label = zip(*batch)
    desired_array = [int(numeric_string) for numeric_string in test_data[0]]
    desired_array = np.array(desired_array, dtype=np.int64)
    print('test_data:')
    print(test_data[0])
    print('converted tes data:')
    print(desired_array)
    request.inputs['text'].CopyFrom(
        tf.contrib.util.make_tensor_proto(desired_array, shape=[1, test_data[0].size]))
    request.inputs['dropout'].CopyFrom(
        tf.contrib.util.make_tensor_proto(1.0,shape=[1,1]))
    result_counter.throttle()
    result_future = stub.Predict.future(request, 5.0)  # 5 seconds
    result_future.add_done_callback(
        _create_rpc_callback(label[0], result_counter))
  return result_counter.get_error_rate()


def main(_):
  if FLAGS.num_tests > 10000:
    print('num_tests should not be greater than 10k')
    return
  if not FLAGS.server:
    print('please specify server host:port')
    return
  error_rate = do_inference(FLAGS.server, FLAGS.work_dir,
                            FLAGS.concurrency, FLAGS.num_tests)
  print('\nInference error rate: %s%%' % (error_rate * 100))


if __name__ == '__main__':
  tf.app.run()
