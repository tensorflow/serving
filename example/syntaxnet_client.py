# !/usr/bin/env python2.7
# coding=utf-8

from __future__ import print_function

# This is a placeholder for a Google-internal import.
import time
import grpc
import random
import tensorflow as tf

from syntaxnet import sentence_pb2
from tensorflow_serving.apis import syntaxnet_service_pb2

tf.app.flags.DEFINE_string('server', 'localhost:8500',
  'SyntaxNetService host:port')
tf.app.flags.DEFINE_bool('interactive', False,
  'Interactivity for choosing batch size during every request')
FLAGS = tf.app.flags.FLAGS


def main(_):
  channel = grpc.insecure_channel(FLAGS.server,
    options=[('grpc.max_send_message_length', -1), (
    'grpc.max_receive_message_length', -1)])

  stub = syntaxnet_service_pb2.SyntaxNetServiceStub(channel)

  request = syntaxnet_service_pb2.SyntaxNetRequest()
  request.model_spec.name = 'russian'
  request.model_spec.signature_name = 'parse_sentences'

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
  sentence.text = u'В линии её кузова я влюбился с первого взгляда.'
  sentence.token.extend(tokens)

  batch_sizes = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]

  i = 1
  while True:
    sentences = [sentence] * (random.choice(batch_sizes) if not FLAGS.interactive else int(input("Enter batch size number: ")))
    del request.inputs[:]
    request.inputs.extend(sentences)

    start_time = time.time()

    result = stub.Parse(request, 60)

    elapsed_time = time.time() - start_time
    print('Completed #{}, time: {}, count: {} '.format(str(i), str(elapsed_time), str(len(result.outputs))))
    i += 1


if __name__ == '__main__':
  tf.app.run()
