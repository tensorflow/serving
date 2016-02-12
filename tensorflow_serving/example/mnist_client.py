# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#!/usr/grte/v4/bin/python2.7

"""A client that talks to mnist_inference service.

The client downloads test images of mnist data set, queries the service with
such test images to get classification, and calculates the inference error rate.
Please see mnist_inference.proto for details.

Typical usage example:

    mnist_client.py --num_tests=100 --server=localhost:9000
"""

import sys
import threading

# This is a placeholder for a Google-internal import.

from grpc.beta import implementations
import numpy
import tensorflow as tf

from tensorflow_serving.example import mnist_inference_pb2
from tensorflow_serving.example import mnist_input_data


tf.app.flags.DEFINE_integer('concurrency', 1,
                            'maximum number of concurrent inference requests')
tf.app.flags.DEFINE_integer('num_tests', 100, 'Number of test images')
tf.app.flags.DEFINE_string('server', '', 'mnist_inference service host:port')
tf.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory. ')
FLAGS = tf.app.flags.FLAGS


def do_inference(hostport, work_dir, concurrency, num_tests):
  """Tests mnist_inference service with concurrent requests.

  Args:
    hostport: Host:port address of the mnist_inference service.
    work_dir: The full path of working directory for test data set.
    concurrency: Maximum number of concurrent requests.
    num_tests: Number of test images to use.

  Returns:
    The classification error rate.

  Raises:
    IOError: An error occurred processing test data set.
  """
  test_data_set = mnist_input_data.read_data_sets(work_dir).test
  host, port = hostport.split(':')
  channel = implementations.insecure_channel(host, int(port))
  stub = mnist_inference_pb2.beta_create_MnistService_stub(channel)
  cv = threading.Condition()
  result = {'active': 0, 'error': 0, 'done': 0}
  def done(result_future, label):
    with cv:
      exception = result_future.exception()
      if exception:
        result['error'] += 1
        print exception
      else:
        sys.stdout.write('.')
        sys.stdout.flush()
        response = numpy.array(result_future.result().value)
        prediction = numpy.argmax(response)
        if label != prediction:
          result['error'] += 1
      result['done'] += 1
      result['active'] -= 1
      cv.notify()
  for _ in range(num_tests):
    request = mnist_inference_pb2.MnistRequest()
    image, label = test_data_set.next_batch(1)
    for pixel in image[0]:
      request.image_data.append(pixel.item())
    with cv:
      while result['active'] == concurrency:
        cv.wait()
      result['active'] += 1
    result_future = stub.Classify.future(request, 5.0)  # 5 seconds
    result_future.add_done_callback(
        lambda result_future, l=label[0]: done(result_future, l))  # pylint: disable=cell-var-from-loop
  with cv:
    while result['done'] != num_tests:
      cv.wait()
    return result['error'] / float(num_tests)


def main(_):
  if FLAGS.num_tests > 10000:
    print 'num_tests should not be greater than 10k'
    return
  if not FLAGS.server:
    print 'please specify server host:port'
    return
  error_rate = do_inference(FLAGS.server, FLAGS.work_dir,
                            FLAGS.concurrency, FLAGS.num_tests)
  print '\nInference error rate: %s%%' % (error_rate * 100)


if __name__ == '__main__':
  tf.app.run()
