workspace(name = "tf_serving")

# Please add all new TensorFlow Serving dependencies in workspace.bzl.
load('//tensorflow_serving:workspace.bzl', 'tf_serving_workspace')
tf_serving_workspace(__workspace_dir__)

# Specify the minimum required bazel version.
load("@org_tensorflow//tensorflow:tensorflow.bzl", "check_version")
check_version("0.3.0")