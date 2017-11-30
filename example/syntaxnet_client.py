# !/usr/bin/env python
# coding=utf-8

# This is a placeholder for a Google-internal import.
import time
import grpc
import random
import os
import argparse

import multiprocessing
from multiprocessing import Pool, Lock

from google.protobuf.internal.encoder import _VarintBytes
from google.protobuf.internal.decoder import _DecodeVarint32

from syntaxnet import sentence_pb2
from tensorflow_serving.apis import syntaxnet_service_pb2


def parse_cmd_line_args():
  """Define and handle command-line args, computing defaults as needed."""
  parser = argparse.ArgumentParser(
      description="Sample client for the server, that works with preprocessed (segmented) data.")
  parser.add_argument("--server", default='localhost:8500',
                      help="DRAGNN server host:port.")
  parser.add_argument("--proto_dir", default='/workspace',
                      help="Input can be a folder with sentence objects as protobufs (data preprocessed with segmenter).")
  parser.add_argument("--processes", default=multiprocessing.cpu_count(),
                      help="Amount of processses to spawn (by default, equals to number of CPUs in the system).")

  return parser.parse_args()


FLAGS = parse_cmd_line_args()

LOCK_OUTPUT = Lock()


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
  return sentences


def parse_request(stub, request, sentences):
  del request.inputs[:]
  request.inputs.extend(sentences)
  start_time = time.time()
  result = stub.Parse(request, 60)
  elapsed_time = time.time() - start_time
  return result, elapsed_time


def parse_file(file):
  channel = grpc.insecure_channel(FLAGS.server,
                                  options=[('grpc.max_send_message_length', -1),
                                           (
                                             'grpc.max_receive_message_length',
                                             -1)])

  stub = syntaxnet_service_pb2.SyntaxNetServiceStub(channel)

  request = syntaxnet_service_pb2.SyntaxNetRequest()
  request.model_spec.name = 'russian'
  request.model_spec.signature_name = 'parse_sentences'

  with open(file, 'rb') as f:
    sentences = read_all_sentences(f)
  result, elapsed_time = parse_request(stub, request, sentences)

  LOCK_OUTPUT.acquire()
  print('ID# {}, Parsed sentences: {}, Elapsed time: {}'.format(
      multiprocessing.current_process().pid,
      str(len(result.outputs)),
      str(elapsed_time)))
  LOCK_OUTPUT.release()


def main():
  print('Number of processes: {}'.format(FLAGS.processes))
  pool = Pool(processes=int(FLAGS.processes))

  print('Started processing.')
  pool.map(parse_file, abs_filepath(FLAGS.proto_dir))
  print('Finished processing.')


if __name__ == '__main__':
  main()
