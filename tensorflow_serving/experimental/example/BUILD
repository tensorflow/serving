# Description: Tensorflow Serving examples.

package(
    default_visibility = ["//tensorflow_serving:internal"],
    features = ["no_layering_check"],
)

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
    python_version = "PY3",
    deps = ["//tensorflow_serving/experimental/tensorflow/ops/remote_predict:remote_predict_py"],
)

py_binary(
    name = "half_plus_two_with_rpop",
    srcs = [
        "half_plus_two_with_rpop.py",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = ["//tensorflow_serving/experimental/tensorflow/ops/remote_predict:remote_predict_py"],
)

py_binary(
    name = "half_plus_two_with_rpop_client",
    srcs = ["half_plus_two_with_rpop_client.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow_serving/apis:predict_proto_py_pb2",
        "//tensorflow_serving/apis:prediction_service_proto_py_pb2",
        "//tensorflow_serving/experimental/tensorflow/ops/remote_predict:remote_predict_py",
    ],
)
