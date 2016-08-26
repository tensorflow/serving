# TensorFlow Serving external dependencies that can be loaded in WORKSPACE
# files.

load('@org_tensorflow//tensorflow:workspace.bzl', 'tf_workspace')

# All TensorFlow Serving external dependencies.
# workspace_dir is the absolute path to the TensorFlow Serving repo. If linked
# as a submodule, it'll likely be '__workspace_dir__ + "/serving"'
def tf_serving_workspace():
  native.local_repository(
    name = "inception_model",
    path = "tf_models/inception",
  )

  tf_workspace()

  # ===== gRPC dependencies =====
  native.bind(
    name = "libssl",
    actual = "@boringssl//:ssl",
  )

  native.bind(
      name = "zlib",
      actual = "@zlib_archive//:zlib",
  )
