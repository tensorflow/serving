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

load("//tensorflow_serving:serving.bzl", "serving_proto_library")

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

serving_proto_library(
    name = "mnist_inference_proto",
    srcs = ["mnist_inference.proto"],
    has_services = 1,
    cc_api_version = 2,
    cc_grpc_version = 1,
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
        ":mnist_input_data",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/contrib/session_bundle:exporter",
    ],
)

cc_binary(
    name = "mnist_inference",
    srcs = [
        "mnist_inference.cc",
    ],
    linkopts = ["-lm"],
    deps = [
        ":mnist_inference_proto",
        "//tensorflow_serving/servables/tensorflow:session_bundle_config_proto",
        "//tensorflow_serving/servables/tensorflow:session_bundle_factory",
        "@grpc//:grpc++",
        "@org_tensorflow//tensorflow/contrib/session_bundle",
        "@org_tensorflow//tensorflow/contrib/session_bundle:manifest_proto_cc",
        "@org_tensorflow//tensorflow/contrib/session_bundle:signature",
        "@org_tensorflow//tensorflow/core:core_cpu",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:tensorflow",
    ],
)

cc_binary(
    name = "mnist_inference_2",
    srcs = [
        "mnist_inference_2.cc",
    ],
    linkopts = ["-lm"],
    deps = [
        ":mnist_inference_proto",
        "//tensorflow_serving/batching:basic_batch_scheduler",
        "//tensorflow_serving/batching:batch_scheduler",
        "//tensorflow_serving/core:manager",
        "//tensorflow_serving/core:servable_handle",
        "//tensorflow_serving/core:servable_id",
        "//tensorflow_serving/servables/tensorflow:simple_servers",
        "@grpc//:grpc++",
        "@org_tensorflow//tensorflow/contrib/session_bundle",
        "@org_tensorflow//tensorflow/contrib/session_bundle:manifest_proto_cc",
        "@org_tensorflow//tensorflow/contrib/session_bundle:signature",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:tensorflow",
    ],
)

py_binary(
    name = "mnist_client",
    srcs = [
        "mnist_client.py",
        "mnist_inference_pb2.py",
    ],
    deps = [
        ":mnist_input_data",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

serving_proto_library(
    name = "inception_inference_proto",
    srcs = ["inception_inference.proto"],
    has_services = 1,
    cc_api_version = 2,
    cc_grpc_version = 1,
)

py_binary(
    name = "inception_export",
    srcs = [
        "inception_export.py",
    ],
    deps = [
        "@inception_model//inception",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/contrib/session_bundle:exporter",
    ],
)

cc_binary(
    name = "inception_inference",
    srcs = [
        "inception_inference.cc",
    ],
    linkopts = ["-lm"],
    deps = [
        ":inception_inference_proto",
        "//tensorflow_serving/batching:basic_batch_scheduler",
        "//tensorflow_serving/batching:batch_scheduler",
        "//tensorflow_serving/core:manager",
        "//tensorflow_serving/core:servable_handle",
        "//tensorflow_serving/core:servable_id",
        "//tensorflow_serving/servables/tensorflow:simple_servers",
        "@grpc//:grpc++",
        "@org_tensorflow//tensorflow/contrib/session_bundle",
        "@org_tensorflow//tensorflow/contrib/session_bundle:manifest_proto_cc",
        "@org_tensorflow//tensorflow/contrib/session_bundle:signature",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:tensorflow",
    ],
)

py_binary(
    name = "inception_client",
    srcs = [
        "inception_client.py",
        "inception_inference_pb2.py",
    ],
    deps = ["@org_tensorflow//tensorflow:tensorflow_py"],
)
