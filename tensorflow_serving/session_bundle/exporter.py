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

"""Export a TensorFlow model.

See: go/tf-exporter
"""

import os
import re
import six

import tensorflow as tf

from google.protobuf import any_pb2

from tensorflow.python.framework import ops
from tensorflow.python.platform import gfile
from tensorflow.python.training import training_util

from tensorflow_serving.session_bundle import gc
from tensorflow_serving.session_bundle import manifest_pb2

# See: go/tf-exporter for these constants and directory structure.
VERSION_FORMAT_SPECIFIER = "%08d"
ASSETS_DIRECTORY = "assets"
EXPORT_BASE_NAME = "export"
EXPORT_SUFFIX_NAME = "meta"
META_GRAPH_DEF_FILENAME = EXPORT_BASE_NAME + "." + EXPORT_SUFFIX_NAME
VARIABLES_FILENAME = EXPORT_BASE_NAME
VARIABLES_FILENAME_PATTERN = VARIABLES_FILENAME + "-?????-of-?????"
INIT_OP_KEY = "serving_init_op"
SIGNATURES_KEY = "serving_signatures"
ASSETS_KEY = "serving_assets"
GRAPH_KEY = "serving_graph"


def regression_signature(input_tensor, output_tensor):
  """Creates a regression signature.

  Args:
    input_tensor: Tensor specifying the input to a graph.
    output_tensor: Tensor specifying the output of a graph.

  Returns:
    A Signature message.
  """
  signature = manifest_pb2.Signature()
  signature.regression_signature.input.tensor_name = input_tensor.name
  signature.regression_signature.output.tensor_name = output_tensor.name
  return signature


def classification_signature(input_tensor,
                             classes_tensor=None,
                             scores_tensor=None):
  """Creates a classification signature.

  Args:
    input_tensor: Tensor specifying the input to a graph.
    classes_tensor: Tensor specifying the output classes of a graph.
    scores_tensor: Tensor specifying the scores of the output classes.

  Returns:
    A Signature message.
  """
  signature = manifest_pb2.Signature()
  signature.classification_signature.input.tensor_name = input_tensor.name
  if classes_tensor is not None:
    signature.classification_signature.classes.tensor_name = classes_tensor.name
  if scores_tensor is not None:
    signature.classification_signature.scores.tensor_name = scores_tensor.name
  return signature


def generic_signature(name_tensor_map):
  """Creates a generic signature of name to Tensor name.

  Args:
    name_tensor_map: Map from logical name to Tensor.

  Returns:
    A Signature message.
  """
  signature = manifest_pb2.Signature()
  for name, tensor in six.iteritems(name_tensor_map):
    signature.generic_signature.map[name].tensor_name = tensor.name
  return signature


class Exporter(object):
  """Exporter helps package a TensorFlow model for serving.

  Args:
    saver: Saver object.
  """

  def __init__(self, saver):
    self._saver = saver
    self._has_init = False

  def init(self,
           graph_def=None,
           init_op=None,
           clear_devices=False,
           default_graph_signature=None,
           named_graph_signatures=None,
           assets=None,
           assets_callback=None):
    """Initialization.

    Args:
      graph_def: A GraphDef message of the graph to be used in inference.
        GraphDef of default graph is used when None.
      init_op: Op to be used in initialization.
      clear_devices: If device info of the graph should be cleared upon export.
      default_graph_signature: Default signature of the graph.
      named_graph_signatures: Map of named input/output signatures of the graph.
      assets: A list of tuples of asset files with the first element being the
        filename (string) and the second being the Tensor.
      assets_callback: callback with a single string argument; called during
        export with the asset path.

    Raises:
      RuntimeError: if init is called more than once.
      TypeError: if init_op is not an Operation or None.
    """
    # Avoid Dangerous default value []
    if named_graph_signatures is None:
      named_graph_signatures = {}
    if assets is None:
      assets = {}

    if self._has_init:
      raise RuntimeError("init should be called only once")
    self._has_init = True

    if graph_def or clear_devices:
      copy = tf.GraphDef()
      if graph_def:
        copy.CopyFrom(graph_def)
      else:
        copy.CopyFrom(tf.get_default_graph().as_graph_def())
      if clear_devices:
        for node in copy.node:
          node.device = ""
      graph_any_buf = any_pb2.Any()
      graph_any_buf.Pack(copy)
      tf.add_to_collection(GRAPH_KEY, graph_any_buf)

    if init_op:
      if not isinstance(init_op, ops.Operation):
        raise TypeError("init_op needs to be an Operation: %s" % init_op)
      tf.add_to_collection(INIT_OP_KEY, init_op)

    signatures_proto = manifest_pb2.Signatures()
    if default_graph_signature:
      signatures_proto.default_signature.CopyFrom(default_graph_signature)
    for signature_name, signature in six.iteritems(named_graph_signatures):
      signatures_proto.named_signatures[signature_name].CopyFrom(signature)
    signatures_any_buf = any_pb2.Any()
    signatures_any_buf.Pack(signatures_proto)
    tf.add_to_collection(SIGNATURES_KEY, signatures_any_buf)

    for filename, tensor in assets:
      asset = manifest_pb2.AssetFile()
      asset.filename = filename
      asset.tensor_binding.tensor_name = tensor.name
      asset_any_buf = any_pb2.Any()
      asset_any_buf.Pack(asset)
      tf.add_to_collection(ASSETS_KEY, asset_any_buf)

    self._assets_callback = assets_callback

  def export(self,
             export_dir_base,
             global_step_tensor,
             sess=None,
             exports_to_keep=None):
    """Exports the model.

    Args:
      export_dir_base: A string path to the base export dir.
      global_step_tensor: An Tensor or tensor name providing the
        global step counter to append to the export directory path and set
        in the manifest version.
      sess: A Session to use to save the parameters.
      exports_to_keep: a gc.Path filter function used to determine the set of
        exports to keep. If set to None, all versions will be kept.

    Raises:
      RuntimeError: if init is not called.
      RuntimeError: if the export would overwrite an existing directory.
    """
    if not self._has_init:
      raise RuntimeError("init must be called first")

    global_step = training_util.global_step(sess, global_step_tensor)
    export_dir = os.path.join(export_dir_base,
                              VERSION_FORMAT_SPECIFIER % global_step)

    # Prevent overwriting on existing exports which could lead to bad/corrupt
    # storage and loading of models. This is an important check that must be
    # done before any output files or directories are created.
    if gfile.Exists(export_dir):
      raise RuntimeError("Overwriting exports can cause corruption and are "
                         "not allowed. Duplicate export dir: %s" % export_dir)

    # Output to a temporary directory which is atomically renamed to the final
    # directory when complete.
    tmp_export_dir = export_dir + "-tmp"
    gfile.MakeDirs(tmp_export_dir)

    self._saver.save(sess,
                     os.path.join(tmp_export_dir, EXPORT_BASE_NAME),
                     meta_graph_suffix=EXPORT_SUFFIX_NAME)

    # Run the asset callback.
    if self._assets_callback:
      assets_dir = os.path.join(tmp_export_dir, ASSETS_DIRECTORY)
      gfile.MakeDirs(assets_dir)
      self._assets_callback(assets_dir)

    # TODO(b/27794910): Delete *checkpoint* file before rename.
    gfile.Rename(tmp_export_dir, export_dir)

    if exports_to_keep:
      # create a simple parser that pulls the export_version from the directory.
      def parser(path):
        match = re.match("^" + export_dir_base + "/(\\d{8})$", path.path)
        if not match:
          return None
        return path._replace(export_version=int(match.group(1)))

      paths_to_delete = gc.negation(exports_to_keep)
      for p in paths_to_delete(gc.get_paths(export_dir_base, parser=parser)):
        gfile.DeleteRecursively(p.path)
