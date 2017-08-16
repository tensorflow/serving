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

#!/usr/bin/env python2.7

from __future__ import print_function

# This is a placeholder for a Google-internal import.

from grpc.beta import implementations
import tensorflow as tf

from syntaxnet import sentence_pb2
from tensorflow_serving.apis import syntaxnet_service_pb2


tf.app.flags.DEFINE_string('server', 'localhost:8500',
                           'SyntaxNetService host:port')
FLAGS = tf.app.flags.FLAGS


def main(_):
  host, port = FLAGS.server.split(':')
  channel = implementations.insecure_channel(host, int(port))

  stub = syntaxnet_service_pb2.beta_create_SyntaxNetService_stub(channel)

  request = syntaxnet_service_pb2.SyntaxNetRequest()
  request.model_spec.name = 'default'

  sentence = sentence_pb2.Sentence()
  sentence.text = 'Привет меня зовут Алексей'
  sentence.token = [sentence_pb2.Token(word='Привет', start=0, end=6),
                    sentence_pb2.Token(word='меня', start=7, end=11),
                    sentence_pb2.Token(word='зовут', start=12, end=17),
                    sentence_pb2.Token(word='Алексей', start=18, end=26),]

  request.inputs = [sentence]
  result = stub.Predict(request, 10.0)
  print(result)


if __name__ == '__main__':
  tf.app.run()
