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
    sha256 = "21d6ac553adcfc9d089925f6d6793fee6a67264a0ce717bc998636662df4ca7e",
    git_commit = "bc69c4ceed6544c109be5693eb40ddcf3a4eb95d",
)

# TensorFlow depends on "io_bazel_rules_closure" so we need this here.
# Needs to be kept in sync with the same target in TensorFlow's WORKSPACE file.
http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "dbe0da2cca88194d13dc5a7125a25dd7b80e1daec7839f33223de654d7a1bcc8",
    strip_prefix = "rules_closure-ba3e07cb88be04a2d4af7009caa0ff3671a79d06",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_closure/archive/ba3e07cb88be04a2d4af7009caa0ff3671a79d06.tar.gz",
        "https://github.com/bazelbuild/rules_closure/archive/ba3e07cb88be04a2d4af7009caa0ff3671a79d06.tar.gz",  # 2017-10-31
    ],
)

# Please add all new TensorFlow Serving dependencies in workspace.bzl.
load("//tensorflow_serving:workspace.bzl", "tf_serving_workspace")

tf_serving_workspace()

# Specify the minimum required bazel version.
load("@org_tensorflow//tensorflow:workspace.bzl", "check_bazel_version_at_least")

check_bazel_version_at_least("0.5.4")
