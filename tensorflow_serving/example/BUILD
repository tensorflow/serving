# Description: Tensorflow Serving examples.

package(
    default_visibility = ["//tensorflow_serving:internal"],
    features = [
        "-parse_headers",
        "no_layering_check",
    ],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

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

cc_library(
    name = "mnist_grpc_cc",
    srcs = [
        "mnist_inference.grpc.pb.cc",
        "mnist_inference.pb.cc",
    ],
    hdrs = [
        "mnist_inference.grpc.pb.h",
        "mnist_inference.pb.h",
    ],
    deps = [
        "@grpc//:grpc++",
    ],
)

py_library(
    name = "mnist_input_data",
    srcs = ["mnist_input_data.py"],
)

py_binary(
    name = "mnist_export",
    srcs = [
        "mnist_export.py",
    ],
    deps = [
        "@tf//tensorflow:tensorflow_py",
        ":mnist_input_data",
        "//tensorflow_serving/session_bundle:exporter",
    ],
)

cc_binary(
    name = "mnist_inference",
    srcs = [
        "mnist_inference.cc",
    ],
    linkopts = ["-lm"],
    deps = [
        "@tf//tensorflow/core:core_cpu",
        "@tf//tensorflow/core:framework",
        "@tf//tensorflow/core:lib",
        "@tf//tensorflow/core:protos_all_cc",
        "@tf//tensorflow/core:tensorflow",
        ":mnist_grpc_cc",
        "@grpc//:grpc",
        "//tensorflow_serving/session_bundle",
        "//tensorflow_serving/session_bundle:manifest_proto",
        "//tensorflow_serving/session_bundle:signature",
    ],
)

cc_binary(
    name = "mnist_inference_2",
    srcs = [
        "mnist_inference_2.cc",
    ],
    linkopts = ["-lm"],
    deps = [
        "@tf//tensorflow/core:framework",
        "@tf//tensorflow/core:lib",
        "@tf//tensorflow/core:protos_all_cc",
        "@tf//tensorflow/core:tensorflow",
        ":mnist_grpc_cc",
        "@grpc//:grpc",
        "//tensorflow_serving/batching:batch_scheduler",
        "//tensorflow_serving/batching:batch_scheduler_retrier",
        "//tensorflow_serving/batching:streaming_batch_scheduler",
        "//tensorflow_serving/core:manager",
        "//tensorflow_serving/core:servable_handle",
        "//tensorflow_serving/core:servable_id",
        "//tensorflow_serving/servables/tensorflow:simple_servers",
        "//tensorflow_serving/session_bundle",
        "//tensorflow_serving/session_bundle:manifest_proto",
        "//tensorflow_serving/session_bundle:signature",
        "//tensorflow_serving/util:unique_ptr_with_deps",
    ],
)

py_binary(
    name = "mnist_client",
    srcs = [
        "mnist_client.py",
        "mnist_inference_pb2.py",
    ],
    deps = [
        "@tf//tensorflow:tensorflow_py",
        ":mnist_input_data",
    ],
)
