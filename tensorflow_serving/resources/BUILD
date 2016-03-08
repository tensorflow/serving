package(
    default_visibility = [
        "//tensorflow_serving:internal",
    ],
    features = [
        "-layering_check",
        "-parse_headers",
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
    deps = ["@tf//google/protobuf:cc_wkt_protos"],
)

cc_library(
    name = "resource_values",
    srcs = ["resource_values.cc"],
    hdrs = ["resource_values.h"],
    visibility = ["//visibility:public"],
    deps = [
    ],
)
