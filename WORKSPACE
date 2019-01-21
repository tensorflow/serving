workspace(name = "tf_serving")

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
    sha256 = "7206ec5dd05b19ad634846af0e4a8e95799c5123dc7f7c8f75d5aa700033de37",
    git_commit = "4bfa2359152e9d106c2c20e9fff67643c8578c81",
)

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# TensorFlow depends on "io_bazel_rules_closure" so we need this here.
# Needs to be kept in sync with the same target in TensorFlow's WORKSPACE file.
http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "a38539c5b5c358548e75b44141b4ab637bba7c4dc02b46b1f62a96d6433f56ae",
    strip_prefix = "rules_closure-dbb96841cc0a5fb2664c37822803b06dab20c7d1",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_closure/archive/dbb96841cc0a5fb2664c37822803b06dab20c7d1.tar.gz",
        "https://github.com/bazelbuild/rules_closure/archive/dbb96841cc0a5fb2664c37822803b06dab20c7d1.tar.gz",  # 2018-04-13
    ],
)

# Please add all new TensorFlow Serving dependencies in workspace.bzl.
load("//tensorflow_serving:workspace.bzl", "tf_serving_workspace")

tf_serving_workspace()

# Specify the minimum required bazel version.
load("@org_tensorflow//tensorflow:version_check.bzl", "check_bazel_version_at_least")

check_bazel_version_at_least("0.20.0")
