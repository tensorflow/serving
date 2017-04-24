# Description: Tensorflow Serving examples.

package(
    default_visibility = ["//tensorflow_serving:internal"],
    features = ["no_layering_check"],
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

py_library(
    name = "mnist_input_data",
    srcs = ["mnist_input_data.py"],
)

# TODO(b/32628014): remove mnist_export after we no longer support
# SessionBundle.
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

py_binary(
    name = "mnist_saved_model",
    srcs = [
        "mnist_saved_model.py",
    ],
    deps = [
        ":mnist_input_data",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/python/saved_model:builder",
        "@org_tensorflow//tensorflow/python/saved_model:constants",
        "@org_tensorflow//tensorflow/python/saved_model:loader",
        "@org_tensorflow//tensorflow/python/saved_model:signature_constants",
        "@org_tensorflow//tensorflow/python/saved_model:signature_def_utils",
        "@org_tensorflow//tensorflow/python/saved_model:tag_constants",
        "@org_tensorflow//tensorflow/python/saved_model:utils",
    ],
)

py_binary(
    name = "mnist_client",
    srcs = [
        "mnist_client.py",
    ],
    deps = [
        ":mnist_input_data",
        "//tensorflow_serving/apis:predict_proto_py_pb2",
        "//tensorflow_serving/apis:prediction_service_proto_py_pb2",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

# TODO(b/32628014): remove inception_export after we no longer support
# SessionBundle.
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

py_binary(
    name = "inception_saved_model",
    srcs = [
        "inception_saved_model.py",
    ],
    deps = [
        "@inception_model//inception",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/python/saved_model:builder",
        "@org_tensorflow//tensorflow/python/saved_model:constants",
        "@org_tensorflow//tensorflow/python/saved_model:loader",
        "@org_tensorflow//tensorflow/python/saved_model:signature_constants",
        "@org_tensorflow//tensorflow/python/saved_model:signature_def_utils",
        "@org_tensorflow//tensorflow/python/saved_model:tag_constants",
        "@org_tensorflow//tensorflow/python/saved_model:utils",
    ],
)

py_binary(
    name = "inception_client",
    srcs = [
        "inception_client.py",
    ],
    deps = [
        "//tensorflow_serving/apis:predict_proto_py_pb2",
        "//tensorflow_serving/apis:prediction_service_proto_py_pb2",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

cc_binary(
    name = "inception_client_cc",
    srcs = [
        "inception_client.cc",
    ],
    deps = [
        "//tensorflow_serving/apis:prediction_service_proto",
        "@grpc//:grpc++",
        "@org_tensorflow//tensorflow/core:framework",
        "@protobuf//:protobuf_lite",
    ],
)
