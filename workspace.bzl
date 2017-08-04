# TensorFlow Serving external dependencies that can be loaded in WORKSPACE
# files.

# All TensorFlow Serving external dependencies.
# workspace_dir is the absolute path to the TensorFlow Serving repo. If linked
# as a submodule, it'll likely be '__workspace_dir__ + "/serving"'
def tf_serving_workspace():
  if (not path.exists('core/')):
    return

  native.local_repository(
    name = "inception",
    path = "tf_models/inception",
  )

  # ===== gRPC dependencies =====
  native.bind(
    name = "libssl",
    actual = "@boringssl//:ssl",
  )

  native.bind(
      name = "zlib",
      actual = "@zlib_archive//:zlib",
  )
