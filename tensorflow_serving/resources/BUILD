# Placeholder: load py_proto_library
load("//tensorflow_serving:serving.bzl", "serving_proto_library")
# Placeholder: load go_proto_library

package(
    default_visibility = [
        "//tensorflow_serving:internal",
    ],
    features = ["-layering_check"],
)

licenses(["notice"])

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

serving_proto_library(
    name = "resources_proto",
    srcs = ["resources.proto"],
    visibility = ["//visibility:public"],
    deps = ["@com_google_protobuf//:cc_wkt_protos"],
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
        ":resources_cc_proto",
        "@com_google_absl//absl/container:flat_hash_set",
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
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "resource_tracker",
    srcs = ["resource_tracker.cc"],
    hdrs = ["resource_tracker.h"],
    deps = [
        ":resource_util",
        ":resources_cc_proto",
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
