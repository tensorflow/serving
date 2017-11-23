# !/usr/bin/env python3
# coding=utf-8

# This is a placeholder for a Google-internal import.
import time
import grpc
import random
import os
import tensorflow as tf

from google.protobuf.internal.encoder import _VarintBytes
from google.protobuf.internal.decoder import _DecodeVarint32

from syntaxnet import sentence_pb2
from tensorflow_serving.apis import syntaxnet_service_pb2

tf.app.flags.DEFINE_string('server', 'localhost:8500',
                           'SyntaxNetService host:port')
tf.app.flags.DEFINE_string('proto_dir', '/workspace',
                           'Input can be a folder with sentence objects as protobufs (data preprocessed with segmenter)')
FLAGS = tf.app.flags.FLAGS


def abs_filepath(directory):
  for dirpath, _, filenames in os.walk(directory):
    for f in filenames:
      yield os.path.abspath(os.path.join(dirpath, f))


def read_all_sentences(file):
  buf = file.read()
  n = 0
  sentences = []
  while n < len(buf):
    msg_len, new_pos = _DecodeVarint32(buf, n)
    n = new_pos
    msg_buf = buf[n:n + msg_len]
    n += msg_len
    sentence = sentence_pb2.Sentence()
    sentence.ParseFromString(msg_buf)
    sentences.append(sentence)
  print('Amount of sentences in a file: {}'.format(len(sentences)))
  return sentences


def parse(stub, request, sentences):
  del request.inputs[:]
  request.inputs.extend(sentences)
  start_time = time.time()
  result = stub.Parse(request, 60)
  elapsed_time = time.time() - start_time
  print(
    'Finished parsing request, time: {}, count: {} '.format(str(elapsed_time),
                                                            str(len(
                                                                result.outputs))))


def main(_):
  channel = grpc.insecure_channel(FLAGS.server,
                                  options=[('grpc.max_send_message_length', -1),
                                           (
                                             'grpc.max_receive_message_length',
                                             -1)])

  stub = syntaxnet_service_pb2.SyntaxNetServiceStub(channel)

  request = syntaxnet_service_pb2.SyntaxNetRequest()
  request.model_spec.name = 'russian'
  request.model_spec.signature_name = 'parse_sentences'

  for file in abs_filepath(FLAGS.proto_dir):
    with open(file, 'rb') as f:
      sentences = read_all_sentences(f)
    parse(stub, request, sentences)


if __name__ == '__main__':
  tf.app.run()
