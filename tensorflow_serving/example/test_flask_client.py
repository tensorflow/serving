#this can be run from outside of the docker

import requests
import tensorflow as tf
from tensorflow.contrib.session_bundle import exporter

def create_and_export_model(export_path):
    x = tf.placeholder(tf.int32, shape=[3])
    z = tf.Variable([2])
    y = tf.mul(x, z)

    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)
    feed_dict = {x: [3, 4, 5]}
    print(sess.run(y, feed_dict=feed_dict))

    version = 1

    print 'Exporting trained model to', export_path
    saver = tf.train.Saver(sharded=True)
    model_exporter = exporter.Exporter(saver)
    model_exporter.init(
        sess.graph.as_graph_def(),
        named_graph_signatures={
            'inputs': exporter.generic_signature({'x': x}),
            'outputs': exporter.generic_signature({'y': y})})
    model_exporter.export(export_path, tf.constant(version), sess)

def test_flask_client():
    create_and_export_model("/tmp/models")

    URL = "http://localhost:6000/model_prediction"
    DATA = {"model_name": "default",
            "input": [1, 2, 3, 4],
            "input_name": "x",
            "input_type": "int32"}

    r = requests.post(URL, data=DATA)

    print r.status_code
    print r.text
