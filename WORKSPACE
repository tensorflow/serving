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
    sha256 = "fc56025a6739b2cbb8b7f7b0f14f8188c841b99cd06ef32e0c8db726ec135adb",
    git_commit = "d2e24b6039433bd83478da8c8c2d6c58034be607",
)

# TensorFlow depends on "io_bazel_rules_closure" so we need this here.
# Needs to be kept in sync with the same target in TensorFlow's WORKSPACE file.
http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "fc56025a6739b2cbb8b7f7b0f14f8188c841b99cd06ef32e0c8db726ec135adb",
    strip_prefix = "rules_closure-d2e24b6039433bd83478da8c8c2d6c58034be607",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_closure/archive/d2e24b6039433bd83478da8c8c2d6c58034be607.tar.gz",
        "https://github.com/bazelbuild/rules_closure/archive/d2e24b6039433bd83478da8c8c2d6c58034be607.tar.gz",  # 2017-10-31
    ],
)

# Please add all new TensorFlow Serving dependencies in workspace.bzl.
load("//tensorflow_serving:workspace.bzl", "tf_serving_workspace")

tf_serving_workspace()

# Specify the minimum required bazel version.
load("@org_tensorflow//tensorflow:workspace.bzl", "check_bazel_version_at_least")

check_bazel_version_at_least("0.5.4")
