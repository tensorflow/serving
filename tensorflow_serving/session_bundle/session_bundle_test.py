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

"""Tests for session_bundle.py."""

import os.path
# This is a placeholder for a Google-internal import.
import numpy as np
import tensorflow as tf
from tensorflow_serving.session_bundle import constants
from tensorflow_serving.session_bundle import manifest_pb2
from tensorflow_serving.session_bundle import session_bundle
from tensorflow_serving.test_util import test_util


class SessionBundleLoadTest(tf.test.TestCase):

  def testBasic(self):
    base_path = test_util.TestSrcDirPath(
        "session_bundle/example/half_plus_two/00000123")
    tf.reset_default_graph()
    sess, meta_graph_def = session_bundle.LoadSessionBundleFromPath(
        base_path, target="", config=tf.ConfigProto(device_count={"CPU": 2}))

    self.assertTrue(sess)
    asset_path = os.path.join(base_path, constants.ASSETS_DIRECTORY)
    with sess.as_default():
      path1, path2 = sess.run(["filename1:0", "filename2:0"])
      self.assertEqual(os.path.join(asset_path, "hello1.txt"), path1)
      self.assertEqual(os.path.join(asset_path, "hello2.txt"), path2)

      collection_def = meta_graph_def.collection_def

      signatures_any = collection_def[constants.SIGNATURES_KEY].any_list.value
      self.assertEquals(len(signatures_any), 1)

      signatures = manifest_pb2.Signatures()
      signatures_any[0].Unpack(signatures)
      default_signature = signatures.default_signature
      input_name = default_signature.regression_signature.input.tensor_name
      output_name = default_signature.regression_signature.output.tensor_name
      y = sess.run([output_name], {input_name: np.array([[0], [1], [2], [3]])})
      # The operation is y = 0.5 * x + 2
      self.assertEqual(y[0][0], 2)
      self.assertEqual(y[0][1], 2.5)
      self.assertEqual(y[0][2], 3)
      self.assertEqual(y[0][3], 3.5)

  def testBadPath(self):
    base_path = test_util.TestSrcDirPath("/no/such/a/dir")
    tf.reset_default_graph()
    with self.assertRaises(RuntimeError) as cm:
      _, _ = session_bundle.LoadSessionBundleFromPath(
          base_path, target="local",
          config=tf.ConfigProto(device_count={"CPU": 2}))
    self.assertTrue("Expected meta graph file missing" in cm.exception.message)

if __name__ == "__main__":
  tf.test.main()
