# TensorFlow Serving external dependencies that can be loaded in WORKSPACE
# files.

load('//tensorflow/tensorflow:workspace.bzl', 'tf_workspace')

# All TensorFlow Serving external dependencies.
# workspace_dir is the absolute path to the TensorFlow Serving repo. If linked
# as a submodule, it'll likely be '__workspace_dir__ + "/serving"'
def tf_serving_workspace(workspace_dir):
  native.local_repository(
    name = "org_tensorflow",
    path = workspace_dir + "/tensorflow",
  )

  native.local_repository(
    name = "inception_model",
    path = workspace_dir + "/tf_models/inception",
  )

  tf_workspace("tensorflow/", "@org_tensorflow")

  # ===== gRPC dependencies =====
  native.bind(
    name = "libssl",
    actual = "@boringssl_git//:ssl",
  )

  native.bind(
      name = "zlib",
      actual = "@zlib_archive//:zlib",
  )
