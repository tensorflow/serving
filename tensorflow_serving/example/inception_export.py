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
"""Export inception model given existing training checkpoints.

The model is exported with proper signatures that can be loaded by standard
tensorflow_model_server.
"""

from __future__ import print_function

import os.path

# This is a placeholder for a Google-internal import.

import tensorflow as tf

from tensorflow.contrib.session_bundle import exporter
from inception import inception_model


tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/inception_train',
                           """Directory where to read training checkpoints.""")
tf.app.flags.DEFINE_string('export_dir', '/tmp/inception_export',
                           """Directory where to export inference model.""")
tf.app.flags.DEFINE_integer('image_size', 299,
                            """Needs to provide same value as in training.""")
FLAGS = tf.app.flags.FLAGS


NUM_CLASSES = 1000
NUM_TOP_CLASSES = 5

WORKING_DIR = os.path.dirname(os.path.realpath(__file__))
SYNSET_FILE = os.path.join(WORKING_DIR, 'imagenet_lsvrc_2015_synsets.txt')
METADATA_FILE = os.path.join(WORKING_DIR, 'imagenet_metadata.txt')


def export():
  # Create index->synset mapping
  synsets = []
  with open(SYNSET_FILE) as f:
    synsets = f.read().splitlines()
  # Create synset->metadata mapping
  texts = {}
  with open(METADATA_FILE) as f:
    for line in f.read().splitlines():
      parts = line.split('\t')
      assert len(parts) == 2
      texts[parts[0]] = parts[1]

  with tf.Graph().as_default():
    # Build inference model.
    # Please refer to Tensorflow inception model for details.

    # Input transformation.
    serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
    feature_configs = {
        'image/encoded': tf.FixedLenFeature(shape=[], dtype=tf.string),
    }
    tf_example = tf.parse_example(serialized_tf_example, feature_configs)
    jpegs = tf_example['image/encoded']
    images = tf.map_fn(preprocess_image, jpegs, dtype=tf.float32)

    # Run inference.
    logits, _ = inception_model.inference(images, NUM_CLASSES + 1)

    # Transform output to topK result.
    values, indices = tf.nn.top_k(logits, NUM_TOP_CLASSES)

    # Create a constant string Tensor where the i'th element is
    # the human readable class description for the i'th index.
    # Note that the 0th index is an unused background class
    # (see inception model definition code).
    class_descriptions = ['unused background']
    for s in synsets:
      class_descriptions.append(texts[s])
    class_tensor = tf.constant(class_descriptions)

    table = tf.contrib.lookup.index_to_string_table_from_tensor(class_tensor)
    classes = table.lookup(tf.to_int64(indices))

    # Restore variables from training checkpoint.
    variable_averages = tf.train.ExponentialMovingAverage(
        inception_model.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    with tf.Session() as sess:
      # Restore variables from training checkpoints.
      ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
      if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        # Assuming model_checkpoint_path looks something like:
        #   /my-favorite-path/imagenet_train/model.ckpt-0,
        # extract global_step from it.
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        print('Successfully loaded model from %s at step=%s.' %
              (ckpt.model_checkpoint_path, global_step))
      else:
        print('No checkpoint file found at %s' % FLAGS.checkpoint_dir)
        return

      # Export inference model.
      init_op = tf.group(tf.tables_initializer(), name='init_op')
      classification_signature = exporter.classification_signature(
          input_tensor=serialized_tf_example,
          classes_tensor=classes,
          scores_tensor=values)
      named_graph_signature = {
          'inputs': exporter.generic_signature({'images': jpegs}),
          'outputs': exporter.generic_signature({
              'classes': classes,
              'scores': values
          })}
      model_exporter = exporter.Exporter(saver)
      model_exporter.init(
          init_op=init_op,
          default_graph_signature=classification_signature,
          named_graph_signatures=named_graph_signature)
      model_exporter.export(FLAGS.export_dir, tf.constant(global_step), sess)
      print('Successfully exported model to %s' % FLAGS.export_dir)


def preprocess_image(image_buffer):
  """Preprocess JPEG encoded bytes to 3D float Tensor."""

  # Decode the string as an RGB JPEG.
  # Note that the resulting image contains an unknown height and width
  # that is set dynamically by decode_jpeg. In other words, the height
  # and width of image is unknown at compile-time.
  image = tf.image.decode_jpeg(image_buffer, channels=3)
  # After this point, all image pixels reside in [0,1)
  # until the very end, when they're rescaled to (-1, 1).  The various
  # adjust_* ops all require this range for dtype float.
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  # Crop the central region of the image with an area containing 87.5% of
  # the original image.
  image = tf.image.central_crop(image, central_fraction=0.875)
  # Resize the image to the original height and width.
  image = tf.expand_dims(image, 0)
  image = tf.image.resize_bilinear(image,
                                   [FLAGS.image_size, FLAGS.image_size],
                                   align_corners=False)
  image = tf.squeeze(image, [0])
  # Finally, rescale to [-1,1] instead of [0, 1)
  image = tf.subtract(image, 0.5)
  image = tf.multiply(image, 2.0)
  return image


def main(unused_argv=None):
  export()


if __name__ == '__main__':
  tf.app.run()
