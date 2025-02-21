workspace(name = "tf_serving")

# TODO(b/269515133): We temporarily remove remote_predict from our builds for
# 2.12 due to a breakage caused by
# github.com/tensorflow/tensorflow/commit/6147c03eb9af1e5d2ae155045b33e909ef96944e
# This will be removed in a subsequent release.
local_repository(
    name = "ignore_remote_predict",
    path = "tensorflow_serving/experimental/tensorflow/ops/remote_predict/",
)

# ===== TensorFlow dependency =====
#
# TensorFlow is imported here instead of in tf_serving_workspace() because
# existing automation scripts that bump the TF commit hash expect it here.
#
# To update TensorFlow to a new revision.
# 1. Update the 'git_commit' args below to include the new git hash.
# 2. Get the sha256 hash of the archive with a command such as...
#    curl -L https://github.com/tensorflow/tensorflow/archive/<git hash>.tar.gz | sha256sum
#    and update the 'sha256' arg with the result.
# 3. Request the new archive to be mirrored on mirror.bazel.build for more
#    reliable downloads.
load("//tensorflow_serving:repo.bzl", "tensorflow_http_archive")
tensorflow_http_archive(
    name = "org_tensorflow",
    sha256 = "c7c0c03f7f520124bb99a8cea2b58d8d0cb35c98c23bb96b939d31914841e44f",
    git_commit = "03902aa28a830bdc7477cf0e60960afd8c15ead3",
    patch = "//third_party/tensorflow:tensorflow.patch",
)

# Import all of TensorFlow Serving's external dependencies.
# Downstream projects (projects importing TensorFlow Serving) need to
# duplicate all code below in their WORKSPACE file in order to also initialize
# those external dependencies.
load("//tensorflow_serving:workspace.bzl", "tf_serving_workspace")
tf_serving_workspace()

# Check bazel version requirement, which is stricter than TensorFlow's.
load("@bazel_skylib//lib:versions.bzl", "versions")
versions.check("6.5.0")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "bazel_skylib",
    sha256 = "74d544d96f4a5bb630d465ca8bbcfe231e3594e5aae57e1edbf17a6eb3ca2506",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/bazel-skylib/releases/download/1.3.0/bazel-skylib-1.3.0.tar.gz",
        "https://github.com/bazelbuild/bazel-skylib/releases/download/1.3.0/bazel-skylib-1.3.0.tar.gz",
    ],
)

http_archive(
    name = "rules_python",
    sha256 = "84aec9e21cc56fbc7f1335035a71c850d1b9b5cc6ff497306f84cced9a769841",
    strip_prefix = "rules_python-0.23.1",
    url = "https://github.com/bazelbuild/rules_python/releases/download/0.23.1/rules_python-0.23.1.tar.gz",
)

load("@rules_python//python:repositories.bzl", "python_register_toolchains")

load(
    "@rules_python//python:repositories.bzl",
    "py_repositories",
    "python_register_toolchains",
)
py_repositories()

load(
    "@org_tensorflow//tensorflow/tools/toolchains/python:python_repo.bzl",
    "python_repository",
)
python_repository(name = "python_version_repo")
load("@python_version_repo//:py_version.bzl", "HERMETIC_PYTHON_VERSION")

python_register_toolchains(
    name = "python",
    ignore_root_user_error = True,
    python_version = HERMETIC_PYTHON_VERSION,
)

# Initialize TensorFlow's external dependencies.
load("@org_tensorflow//tensorflow:workspace3.bzl", "workspace")
workspace()
load("@org_tensorflow//tensorflow:workspace2.bzl", "workspace")
workspace()
load("@org_tensorflow//tensorflow:workspace1.bzl", "workspace")
workspace()
load("@org_tensorflow//tensorflow:workspace0.bzl", "workspace")
workspace()

# Initialize bazel package rules' external dependencies.
load("@rules_pkg//:deps.bzl", "rules_pkg_dependencies")
rules_pkg_dependencies()

load("@rules_proto//proto:repositories.bzl", "rules_proto_dependencies", "rules_proto_toolchains")

rules_proto_dependencies()
rules_proto_toolchains()

load(
    "@local_tsl//third_party/gpus/cuda/hermetic:cuda_json_init_repository.bzl",
    "cuda_json_init_repository",
)

cuda_json_init_repository()

load(
    "@cuda_redist_json//:distributions.bzl",
    "CUDA_REDISTRIBUTIONS",
    "CUDNN_REDISTRIBUTIONS",
)
load(
    "@local_tsl//third_party/gpus/cuda/hermetic:cuda_redist_init_repositories.bzl",
    "cuda_redist_init_repositories",
    "cudnn_redist_init_repository",
)

cuda_redist_init_repositories(
    cuda_redistributions = CUDA_REDISTRIBUTIONS,
)

cudnn_redist_init_repository(
    cudnn_redistributions = CUDNN_REDISTRIBUTIONS,
)

load(
    "@local_tsl//third_party/gpus/cuda/hermetic:cuda_configure.bzl",
    "cuda_configure",
)

cuda_configure(name = "local_config_cuda")

load(
    "@local_tsl//third_party/nccl/hermetic:nccl_redist_init_repository.bzl",
    "nccl_redist_init_repository",
)

nccl_redist_init_repository()

load(
    "@local_tsl//third_party/nccl/hermetic:nccl_configure.bzl",
    "nccl_configure",
)

nccl_configure(name = "local_config_nccl")

