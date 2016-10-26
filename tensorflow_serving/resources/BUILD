package(
    default_visibility = [
        "//tensorflow_serving:internal",
    ],
    features = ["-layering_check"],
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
            "g3doc/sitemap.md",
        ],
    ),
)

load("//tensorflow_serving:serving.bzl", "serving_proto_library")

serving_proto_library(
    name = "resources_proto",
    srcs = ["resources.proto"],
    cc_api_version = 2,
    visibility = ["//visibility:public"],
    deps = ["@protobuf//:cc_wkt_protos"],
)

cc_library(
    name = "resource_values",
    srcs = ["resource_values.cc"],
    hdrs = ["resource_values.h"],
    visibility = ["//visibility:public"],
    deps = [
    ],
)

cc_library(
    name = "resource_util",
    srcs = ["resource_util.cc"],
    hdrs = ["resource_util.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":resources_proto",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "resource_util_test",
    size = "small",
    srcs = ["resource_util_test.cc"],
    deps = [
        ":resource_util",
        "//tensorflow_serving/core/test_util:test_main",
        "//tensorflow_serving/test_util",
    ],
)

cc_library(
    name = "resource_tracker",
    srcs = ["resource_tracker.cc"],
    hdrs = ["resource_tracker.h"],
    deps = [
        ":resource_util",
        ":resources_proto",
        "//tensorflow_serving/core:loader",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "resource_tracker_test",
    size = "small",
    srcs = ["resource_tracker_test.cc"],
    deps = [
        ":resource_tracker",
        "//tensorflow_serving/core/test_util:mock_loader",
        "//tensorflow_serving/core/test_util:test_main",
        "//tensorflow_serving/test_util",
    ],
)
