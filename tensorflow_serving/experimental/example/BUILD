# Description: Tensorflow Serving examples.

# Placeholder: load py_binary
load("@org_tensorflow//tensorflow:tensorflow.bzl", "if_oss")

package(features = ["no_layering_check"])

licenses(["notice"])

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
)

py_binary(
    name = "remote_predict_client",
    srcs = ["remote_predict_client.py"],
    exec_properties = if_oss(
        None,
        select({
            "//tools/cpp:asan_build": {"cpp_link.mem": "20g"},
            "//conditions:default": None,
        }),
    ),
    tags = [
        "manual",
    ],
    deps = ["//tensorflow_serving/experimental/tensorflow/ops/remote_predict:remote_predict_py"],
)

py_binary(
    name = "half_plus_two_with_rpop",
    srcs = [
        "half_plus_two_with_rpop.py",
    ],
    exec_properties = if_oss(
        None,
        select({
            "//tools/cpp:asan_build": {"cpp_link.mem": "20g"},
            "//conditions:default": None,
        }),
    ),
    srcs_version = "PY3",
    tags = [
        "manual",
    ],
    deps = ["//tensorflow_serving/experimental/tensorflow/ops/remote_predict:remote_predict_py"],
)

py_binary(
    name = "half_plus_two_with_rpop_client",
    srcs = ["half_plus_two_with_rpop_client.py"],
    exec_properties = if_oss(
        None,
        select({
            "//tools/cpp:asan_build": {"cpp_link.mem": "20g"},
            "//conditions:default": None,
        }),
    ),
    tags = [
        "manual",
    ],
    deps = [
        "//tensorflow_serving/apis:predict_proto_py_pb2",
        "//tensorflow_serving/apis:prediction_service_proto_py_pb2",
        "//tensorflow_serving/experimental/tensorflow/ops/remote_predict:remote_predict_py",
    ],
)
