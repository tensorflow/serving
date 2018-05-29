# TensorFlow Serving external dependencies that can be loaded in WORKSPACE
# files.

load('@org_tensorflow//tensorflow:workspace.bzl', 'tf_workspace')

def tf_serving_workspace():
  '''All TensorFlow Serving external dependencies.'''
  # The inception model's BUILD file is written as if it is the root BUILD
  # file. We use strip_prefix to make the inception model directory the root.
  native.http_archive(
      name = "inception_model",
      urls = [
          "https://mirror.bazel.build/github.com/tensorflow/models/archive/6fc65ee60ac39be0445e5a311b40dc7ccce214d0.tar.gz",
          "https://github.com/tensorflow/models/archive/6fc65ee60ac39be0445e5a311b40dc7ccce214d0.tar.gz",
      ],
      sha256 = "7a908017d60fca54c80405527576f08dbf8d130efe6a53791639ff3b26afffbc",
      strip_prefix = "models-6fc65ee60ac39be0445e5a311b40dc7ccce214d0/research/inception",
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

  # gRPC wants the existence of a cares dependence but its contents are not
  # actually important since we have set GRPC_ARES=0 in tools/bazel.rc
  native.bind(
      name = "cares",
      actual = "@grpc//third_party/nanopb:nanopb",
  )

  # ===== RapidJSON (rapidjson.org) dependencies =====
  native.new_http_archive(
      name = "com_github_tencent_rapidjson",
      urls = [
          "https://github.com/Tencent/rapidjson/archive/v1.1.0.zip",
      ],
      sha256 = "8e00c38829d6785a2dfb951bb87c6974fa07dfe488aa5b25deec4b8bc0f6a3ab",
      strip_prefix = "rapidjson-1.1.0",
      build_file = "third_party/rapidjson.BUILD"
  )

  # ===== libevent (libevent.org) dependencies =====
  native.new_http_archive(
      name = "com_github_libevent_libevent",
      urls = [
          "https://github.com/libevent/libevent/archive/release-2.1.8-stable.zip"
      ],
      sha256 = "70158101eab7ed44fd9cc34e7f247b3cae91a8e4490745d9d6eb7edc184e4d96",
      strip_prefix = "libevent-release-2.1.8-stable",
      build_file = "third_party/libevent.BUILD"
  )
