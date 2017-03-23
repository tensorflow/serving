# TensorFlow Serving external dependencies that can be loaded in WORKSPACE
# files.

load('@org_tensorflow//tensorflow:workspace.bzl', 'tf_workspace')

# All TensorFlow Serving external dependencies.
# workspace_dir is the absolute path to the TensorFlow Serving repo. If linked
# as a submodule, it'll likely be '__workspace_dir__ + "/serving"'
def tf_serving_workspace():
  native.new_local_repository(
    name = "inception_model",
    path = "tf_models/inception",
    build_file = "tf_models/inception/inception/BUILD",
  )

  tf_workspace(path_prefix = "", tf_repo_name = "org_tensorflow")

  # ===== gRPC dependencies =====
  native.bind(
    name = "libssl",
    actual = "@boringssl//:ssl",
  )

  native.bind(
      name = "zlib",
      actual = "@zlib_archive//:zlib",
  )
