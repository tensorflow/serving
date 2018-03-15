# Copyright 2017 Google Inc. All Rights Reserved.
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

import argparse
import tensorflow as tf


def _generate_saved_model_for_matrix_half_plus_two(export_dir):
  """Creates SavedModel for half plus two model that accepts batches of
       3*3 matrices.
       The model divides all elements in each matrix by 2 and adds 2 to them.
       So, for one input matrix [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
       the result will be [[2.5, 3, 3.5], [4, 4.5, 5], [5.5, 6, 6.5]].
    Args:
      export_dir: The directory where to write SavedModel files.
    """
  builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
  with tf.Session() as session:
    x = tf.placeholder(tf.float32, shape=[None, 3, 3], name="x")
    a = tf.constant(0.5)
    b = tf.constant(2.0)
    y = tf.add(tf.multiply(a, x), b, name="y")
    predict_signature_def = (
        tf.saved_model.signature_def_utils.predict_signature_def({
            "x": x
        }, {"y": y}))
    signature_def_map = {
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            predict_signature_def
    }
    session.run(tf.global_variables_initializer())
    builder.add_meta_graph_and_variables(
        session, [tf.saved_model.tag_constants.SERVING],
        signature_def_map=signature_def_map)
    builder.save()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--output_dir",
      type=str,
      default="/tmp/matrix_half_plus_two/1",
      help="The directory where to write SavedModel files.")
  args = parser.parse_args()
  _generate_saved_model_for_matrix_half_plus_two(args.output_dir)
