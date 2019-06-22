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
