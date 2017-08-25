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
  tokens = [sentence_pb2.Token(word=u"В", start=0, end=1, break_level=0),
            sentence_pb2.Token(word=u"линии", start=3, end=12, break_level=1),
            sentence_pb2.Token(word=u"её", start=14, end=17, break_level=1),
            sentence_pb2.Token(word=u"кузова", start=19, end=30, break_level=1),
            sentence_pb2.Token(word=u"я", start=32, end=33, break_level=1),
            sentence_pb2.Token(word=u"влюбился", start=35, end=50, break_level=1),
            sentence_pb2.Token(word=u"с", start=52, end=53, break_level=1),
            sentence_pb2.Token(word=u"первого", start=55, end=68, break_level=1),
            sentence_pb2.Token(word=u"взгляда", start=70, end=83, break_level=1),
            sentence_pb2.Token(word=u".", start=84, end=84, break_level=0),
            ]
  sentence = sentence_pb2.Sentence()
  sentence.text = u'В линии её кузова я влюбился с первого взгляда'
  sentence.token.extend(tokens)

  request.inputs.extend([sentence])
  result = stub.Parse(request, 10)
  print(result)


if __name__ == '__main__':
  tf.app.run()
