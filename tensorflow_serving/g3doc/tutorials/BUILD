# Description: Test targets for tutorial notebooks.

package(default_visibility = ["//tensorflow_serving:__subpackages__"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//research/colab:build_defs.bzl", "colab_notebook_test")
load("//research/colab:build_defs.bzl", "colab_binary")

filegroup(
    name = "files",
    srcs = [
    ] + glob(["*"]),
)

colab_binary(
    name = "lite_notebook",
    # Use hermetic python. This is necessary until c/31549734 is resolved
    launcher = "//devtools/python/launcher",
    deps = [
        # We don't use:
        #    ".../client:notebook_deps",
        # in order to minimize deps. However this is a good place
        # from which you can identify additional dependencies to add.
        "//file/base",
        "//file/liball",
        "//file/localfile",
        "//third_party/py/matplotlib",
        "//third_party/py/numpy",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)
