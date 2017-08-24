# !/usr/bin/env python2.7
# coding=utf-8

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

  # text = u'П р и в е т '
  # tokens = [sentence_pb2.Token(word=word, start=-1, end=-1) for word in text.split()]
  tokens = [sentence_pb2.Token(word=u"В", start=0, end=0, break_level=0),
            sentence_pb2.Token(word=u"линии", start=2, end=6, break_level=1),
            sentence_pb2.Token(word=u"её", start=8, end=9, break_level=1),
            sentence_pb2.Token(word=u"кузова", start=11, end=16, break_level=1),
            sentence_pb2.Token(word=u"я", start=17, end=18, break_level=1),
            sentence_pb2.Token(word=u"влюбился", start=20, end=27, break_level=1),
            sentence_pb2.Token(word=u"с", start=29, end=30, break_level=1),
            sentence_pb2.Token(word=u"первого", start=32, end=38, break_level=1),
            sentence_pb2.Token(word=u"взгляда", start=40, end=46, break_level=1),
            sentence_pb2.Token(word=u".", start=48, end=48, break_level=1),
            ]
  sentence = sentence_pb2.Sentence()
  sentence.text = u'В линии её кузова я влюбился с первого взгляда'
  sentence.token.extend(tokens)

  request.inputs.extend([sentence])
  result = stub.Parse(request, 10)
  print(result)


if __name__ == '__main__':
  tf.app.run()
