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
    sha256 = "23c0bdb76f3c72214ab7ed9ec98011e11a95eb768770061332ddb88e266552b8",
    git_commit = "10baba2ed8c97c9c997218288dfac0fcfe9ca5b5",
)

# Import all of TensorFlow Serving's external dependencies.
# Downstream projects (projects importing TensorFlow Serving) need to
# duplicate all code below in their WORKSPACE file in order to also initialize
# those external dependencies.
load("//tensorflow_serving:workspace.bzl", "tf_serving_workspace")
tf_serving_workspace()

# Check bazel version requirement, which is stricter than TensorFlow's.
load("@bazel_skylib//lib:versions.bzl", "versions")
versions.check("6.1.0")

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

