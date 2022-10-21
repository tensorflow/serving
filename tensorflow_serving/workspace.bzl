"""Provides a macro to import all TensorFlow Serving dependencies.

Some of the external dependencies need to be initialized. To do this, duplicate
the initialization code from TensorFlow Serving's WORKSPACE file.
"""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

def tf_serving_workspace():
    """All TensorFlow Serving external dependencies."""

    # ===== Bazel skylib dependency =====
    http_archive(
        name = "bazel_skylib",
        sha256 = "74d544d96f4a5bb630d465ca8bbcfe231e3594e5aae57e1edbf17a6eb3ca2506",
        url = "https://github.com/bazelbuild/bazel-skylib/releases/download/1.3.0/bazel-skylib-1.3.0.tar.gz",
    )

    # ===== Bazel package rules dependency =====
    http_archive(
        name = "rules_pkg",
        sha256 = "352c090cc3d3f9a6b4e676cf42a6047c16824959b438895a76c2989c6d7c246a",
        url = "https://github.com/bazelbuild/rules_pkg/releases/download/0.2.5/rules_pkg-0.2.5.tar.gz",
    )

    # ===== RapidJSON (rapidjson.org) dependency =====
    http_archive(
        name = "com_github_tencent_rapidjson",
        url = "https://github.com/Tencent/rapidjson/archive/v1.1.0.zip",
        sha256 = "8e00c38829d6785a2dfb951bb87c6974fa07dfe488aa5b25deec4b8bc0f6a3ab",
        strip_prefix = "rapidjson-1.1.0",
        build_file = "@//third_party/rapidjson:BUILD",
    )

    # ===== libevent (libevent.org) dependency =====
    http_archive(
        name = "com_github_libevent_libevent",
        url = "https://github.com/libevent/libevent/archive/release-2.1.8-stable.zip",
        sha256 = "70158101eab7ed44fd9cc34e7f247b3cae91a8e4490745d9d6eb7edc184e4d96",
        strip_prefix = "libevent-release-2.1.8-stable",
        build_file = "@//third_party/libevent:BUILD",
    )

    # ===== ICU dependency =====
    # Note: This overrides the dependency from TensorFlow with a version
    # that contains all data.
    http_archive(
        name = "icu",
        strip_prefix = "icu-release-64-2",
        sha256 = "dfc62618aa4bd3ca14a3df548cd65fe393155edd213e49c39f3a30ccd618fc27",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/unicode-org/icu/archive/release-64-2.zip",
            "https://github.com/unicode-org/icu/archive/release-64-2.zip",
        ],
        build_file = "//third_party/icu:BUILD",
        patches = ["//third_party/icu:data.patch"],
        patch_args = ["-p1", "-s"],
    )

    # ===== TF.Text dependencies
    # NOTE: Before updating this version, you must update the test model
    # and double check all custom ops have a test:
    # https://github.com/tensorflow/text/blob/master/oss_scripts/model_server/save_models.py
    http_archive(
        name = "org_tensorflow_text",
        sha256 = "774af1b75c6af53f0ee1be28a8d2d75f2fc60c112f615418a0ee76f8c3d59c0f",
        strip_prefix = "text-2.8.2",
        url = "https://github.com/tensorflow/text/archive/v2.8.2.zip",
        patches = ["@//third_party/tf_text:tftext.patch"],
        patch_args = ["-p1"],
        repo_mapping = {"@com_google_re2": "@com_googlesource_code_re2"},
    )

    http_archive(
        name = "com_google_sentencepiece",
        strip_prefix = "sentencepiece-0.1.96",
        sha256 = "8409b0126ebd62b256c685d5757150cf7fcb2b92a2f2b98efb3f38fc36719754",
        urls = [
            "https://github.com/google/sentencepiece/archive/refs/tags/v0.1.96.zip",
        ],
        build_file = "//third_party/sentencepiece:BUILD",
        patches = ["//third_party/sentencepiece:sentencepiece.patch"],
        patch_args = ["-p1"],
    )

    http_archive(
        name = "darts_clone",
        build_file = "//third_party/darts_clone:BUILD",
        sha256 = "c97f55d05c98da6fcaf7f9ecc6a6dc6bc5b18b8564465f77abff8879d446491c",
        strip_prefix = "darts-clone-e40ce4627526985a7767444b6ed6893ab6ff8983",
        urls = [
            "https://github.com/s-yata/darts-clone/archive/e40ce4627526985a7767444b6ed6893ab6ff8983.zip",
        ],
    )

    http_archive(
        name = "com_google_glog",
        sha256 = "1ee310e5d0a19b9d584a855000434bb724aa744745d5b8ab1855c85bff8a8e21",
        strip_prefix = "glog-028d37889a1e80e8a07da1b8945ac706259e5fd8",
        urls = [
            "https://mirror.bazel.build/github.com/google/glog/archive/028d37889a1e80e8a07da1b8945ac706259e5fd8.tar.gz",
            "https://github.com/google/glog/archive/028d37889a1e80e8a07da1b8945ac706259e5fd8.tar.gz",
        ],
    )

    # ==== TensorFlow Decision Forests ===
    http_archive(
        name = "org_tensorflow_decision_forests",
        sha256 = "7a6b8187341da782ca93ee6e61ded14fba23f4bc0750beac291ae15277b1257e",
        strip_prefix = "decision-forests-1.0.1",
        url = "https://github.com/tensorflow/decision-forests/archive/refs/tags/1.0.1.zip",
    )

    http_archive(
        name = "ydf",
        sha256 = "ba8e69ef3889c58cde58ce201ff88ea3acf376bcb9996a55c070959d5fa0a51b",
        strip_prefix = "yggdrasil-decision-forests-144b139d688381beea31100e75e5d621127f8ee4",
        urls = ["https://github.com/google/yggdrasil-decision-forests/archive/144b139d688381beea31100e75e5d621127f8ee4.zip"],
    )

    # The Boost repo is organized into git sub-modules (see the list at
    # https://github.com/boostorg/boost/tree/master/libs), which requires "new_git_repository".
    new_git_repository(
        name = "org_boost",
        commit = "b7b1371294b4bdfc8d85e49236ebced114bc1d8f",  # boost-1.75.0
        build_file = "//third_party/boost:BUILD",
        init_submodules = True,
        recursive_init_submodules = True,
        remote = "https://github.com/boostorg/boost",
    )
