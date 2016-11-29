# TensorFlow Serving external dependencies that can be loaded in WORKSPACE
# files.

load('@org_tensorflow//tensorflow:workspace.bzl', 'tf_workspace')

# All TensorFlow Serving external dependencies.
# workspace_dir is the absolute path to the TensorFlow Serving repo. If linked
# as a submodule, it'll likely be '__workspace_dir__ + "/serving"'
def tf_serving_workspace(workspace_dir):
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

  # ===== caffe =====
  native.new_git_repository(
    name = "caffe",
    remote = "https://github.com/BVLC/caffe",
    commit = "50c9a0fc8eed0101657e9f8da164d88f66242aeb",
    init_submodules = True,
    build_file = workspace_dir + "/third_party/caffe/caffe.BUILD",
  )

  # ===== caffe build/integration tools =====
  native.local_repository(
    name = "caffe_tools",
    path = workspace_dir + "/third_party/caffe"
  )
