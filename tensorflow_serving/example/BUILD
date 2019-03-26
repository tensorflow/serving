# Description: Tensorflow Serving examples.

package(
    default_visibility = ["//tensorflow_serving:internal"],
    features = ["no_layering_check"],
)

licenses(["notice"])  # Apache 2.0

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

cc_binary(
    name = "resnet_client_cc",
    srcs = [
        "resnet_client.cc",
    ],
    deps = [
        "//tensorflow_serving/apis:prediction_service_proto",
        "@grpc//:grpc++",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@protobuf_archive//:protobuf_lite",
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
    ],
)

py_library(
    name = "mnist_input_data",
    srcs = ["mnist_input_data.py"],
    deps = [
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)
