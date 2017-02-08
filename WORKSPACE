workspace(name = "tf_serving")

local_repository(
    name = "org_tensorflow",
    path = "tensorflow",
)

# TensorFlow depends on "io_bazel_rules_closure" so we need this here.
# Needs to be kept in sync with the same target in TensorFlow's WORKSPACE file.
http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "60fc6977908f999b23ca65698c2bb70213403824a84f7904310b6000d78be9ce",
    strip_prefix = "rules_closure-5ca1dab6df9ad02050f7ba4e816407f88690cf7d",
    urls = [
        "http://bazel-mirror.storage.googleapis.com/github.com/bazelbuild/rules_closure/archive/5ca1dab6df9ad02050f7ba4e816407f88690cf7d.tar.gz",  # 2017-02-03
        "https://github.com/bazelbuild/rules_closure/archive/5ca1dab6df9ad02050f7ba4e816407f88690cf7d.tar.gz",
    ],
)

# Please add all new TensorFlow Serving dependencies in workspace.bzl.
load('//tensorflow_serving:workspace.bzl', 'tf_serving_workspace')
tf_serving_workspace()

# Specify the minimum required bazel version.
load("@org_tensorflow//tensorflow:workspace.bzl", "check_version")
check_version("0.4.2")
