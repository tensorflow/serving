# Description: Tensorflow Serving test utils.

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = [
    "//tensorflow_serving:internal",
])

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

cc_library(
    name = "test_util",
    testonly = 1,
    srcs = ["test_util.cc"],
    hdrs = ["test_util.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "@com_google_googletest//:gtest",
        "@com_google_protobuf//:protobuf",
        "@local_tsl//tsl/platform:protobuf",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:session_options",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_binary(
    name = "grpc_client",
    srcs = ["grpc_client.cc"],
    deps = [
        "//tensorflow_serving/apis:prediction_service_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core/platform:env",
    ],
)
